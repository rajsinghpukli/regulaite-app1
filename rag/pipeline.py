from __future__ import annotations
import os, json, re
from typing import Dict, Any, List, Optional
from openai import OpenAI
from pydantic import ValidationError
from .schema import RegulAIteAnswer, DEFAULT_EMPTY
from .agents import build_system_instruction
from .router import normalize_mode
from .websearch import ddg_search

client = OpenAI()

# ---------- helpers ----------
def _history_to_brief(history: List[Dict[str, str]], max_pairs: int = 8) -> str:
    """Convert chat history to a compact prompt string."""
    if not history:
        return ""
    turns = history[-(max_pairs * 2):]
    lines: List[str] = []
    for h in turns:
        role = "User" if h["role"] == "user" else "Assistant"
        text = h["content"].replace("\n", " ")
        lines.append(f"{role}: {text}")
    return "\n".join(lines)


def _maybe_json_parse(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        return {}


# ---------- main ----------
def ask(
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
    mode: str = "default",
) -> RegulAIteAnswer:
    """Main pipeline: query enrichment, retrieval, answer drafting."""
    history = history or []
    mode = normalize_mode(mode)

    # Build system instruction
    sys_msg = build_system_instruction(mode=mode)

    # History context
    history_brief = _history_to_brief(history)

    # Step 1: Enrichment
    enrich_prompt = f"""
You are assisting with regulatory Q&A.
Query: {query}
History: {history_brief}

Rewrite into 2-3 enriched search queries (precise, regulatory terms, synonyms).
Output as JSON list.
"""
    enr = client.responses.create(
        model=os.getenv("RESPONSES_MODEL", "gpt-4o-mini"),
        input=[{"role": "system", "content": sys_msg},
               {"role": "user", "content": enrich_prompt}],
    )
    enr_text = enr.output_text.strip()
    try:
        search_queries = json.loads(enr_text)
        if not isinstance(search_queries, list):
            search_queries = [query]
    except Exception:
        search_queries = [query]

    # Step 2: Retrieval (vector + web always)
    evidence_chunks: List[str] = []
    try:
        from openai import VectorStore
        vs_id = os.getenv("OPENAI_VECTOR_STORE_ID")
        if vs_id:
            vs = VectorStore(vs_id)
            for sq in search_queries:
                results = vs.similarity_search(sq, k=12)
                for r in results:
                    snippet = r["document"]["text"][:500]
                    evidence_chunks.append(snippet)
    except Exception:
        pass

    # Always add web search
    web_snippets = []
    for sq in search_queries[:2]:
        web_snippets.extend(ddg_search(sq))
    evidence_chunks.extend(web_snippets[:5])

    evidence_text = "\n\n".join(evidence_chunks[:15])

    # Step 3: Draft answer
    draft_prompt = f"""
Answer the user's question in long, structured, ChatGPT-style form.

User question: {query}

Relevant evidence:
{evidence_text}

Instructions:
- Provide a detailed narrative answer with headings, bullet points, and tables where useful.
- Always include: (a) structured overview per framework, (b) comparison table, (c) short approval workflow + reporting matrix if relevant, and (d) recommendation for Khaleeji Bank.
- If evidence is missing, gracefully infer from general knowledge — never return 'no answer'.
- Maintain professional explanatory style.
- Suggest 3-6 follow-up questions at the end.
Output in Markdown.
"""

    draft = client.responses.create(
        model=os.getenv("RESPONSES_MODEL", "gpt-4o-mini"),
        input=[{"role": "system", "content": sys_msg},
               {"role": "user", "content": draft_prompt}],
    )
    text = draft.output_text.strip()

    # Step 4: Secondary structuring pass
    struct_prompt = f"""
Take the following draft answer and reformat into structured JSON.

Draft answer:
{text}

Output JSON with fields:
- raw_markdown: full Markdown version of the answer (narrative, tables, workflows).
- summary: short executive summary.
- per_source: mapping of framework → list of 2-5 short evidence quotes.
- comparison_table_md: Markdown table if available.
- follow_up_suggestions: list of 3-6 follow-up questions.
"""
    struct = client.responses.create(
        model=os.getenv("RESPONSES_MODEL", "gpt-4o-mini"),
        input=[{"role": "system", "content": sys_msg},
               {"role": "user", "content": struct_prompt}],
    )
    struct_text = struct.output_text.strip()
    data = _maybe_json_parse(struct_text)

    # ---- Graceful fallback ----
    markdown = ""
    if isinstance(data, dict) and "raw_markdown" in data:
        markdown = data["raw_markdown"]
    else:
        markdown = text or "_Answer failed, but no blank output._"

    return RegulAIteAnswer(
        raw_markdown=markdown,
        summary=data.get("summary", "") if isinstance(data, dict) else "",
        per_source=data.get("per_source", {}) if isinstance(data, dict) else {},
        comparison_table_md=data.get("comparison_table_md", "") if isinstance(data, dict) else "",
        follow_up_suggestions=(
            data.get("follow_up_suggestions")
            if isinstance(data, dict) and data.get("follow_up_suggestions")
            else [
                f"What are approval thresholds and board oversight for {query}?",
                f"Draft a closure checklist for {query} with controls and required evidence.",
                f"What reporting pack fields should be in the monthly board pack for {query}?",
                f"How should breaches/exceptions for {query} be escalated and documented?",
                f"What stress-test scenarios are relevant for {query} and how to calibrate them?",
            ]
        ),
    )
