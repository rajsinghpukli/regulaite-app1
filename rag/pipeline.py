from __future__ import annotations
import os, json
from typing import Dict, Any, List, Optional
from openai import OpenAI
from .schema import RegulAIteAnswer
from .agents import build_system_instruction
from .router import normalize_mode
from .websearch import ddg_search

client = OpenAI()


# ---------- helpers ----------
def _history_to_brief(history: List[Dict[str, str]], max_pairs: int = 8) -> str:
    """Convert chat history to compact form for context injection."""
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
    user_id: Optional[str] = None,   # ✅ added back
    history: Optional[List[Dict[str, str]]] = None,
    mode: str = "default",
) -> RegulAIteAnswer:
    """Main pipeline: enrich → retrieve → draft → structure → return."""
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
    try:
        search_queries = json.loads(enr.output_text.strip())
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

    # Web search
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
- Provide a detailed narrative with headings, bullet points, and tables where useful.
- Always include: (a) structured overview per framework, (b) comparison table, 
  (c) short approval workflow + reporting matrix if relevant, (d) recommendation for Khaleeji Bank.
- If evidence is missing, gracefully infer from general knowledge — never return 'no answer'.
- Suggest 3-6 follow-up questions at the end.
Output in Markdown.
"""
    draft = client.responses.create(
        model=os.getenv("RESPONSES_MODEL", "gpt-4o-mini"),
        input=[{"role": "system", "content": sys_msg},
               {"role": "user", "content": draft_prompt}],
    )
    text = draft.output_text.strip()

    # Step 4: Structuring pass
    struct_prompt = f"""
Take the following draft answer and reformat into structured JSON.

Draft answer:
{text}

Output JSON with fields:
- raw_markdown
- summary
- per_source
- comparison_table_md
- follow_up_suggestions
"""
    struct = client.responses.create(
        model=os.getenv("RESPONSES_MODEL", "gpt-4o-mini"),
        input=[{"role": "system", "content": sys_msg},
               {"role": "user", "content": struct_prompt}],
    )
    data = _maybe_json_parse(struct.output_text.strip())

    # ✅ Graceful fallback
    markdown = data.get("raw_markdown", text or "_Answer failed, but no blank output._")

    return RegulAIteAnswer(
        raw_markdown=markdown,
        summary=data.get("summary", ""),
        per_source=data.get("per_source", {}),
        comparison_table_md=data.get("comparison_table_md", ""),
        follow_up_suggestions=(
            data.get("follow_up_suggestions")
            if data.get("follow_up_suggestions")
            else [
                f"What are approval thresholds and board oversight for {query}?",
                f"Draft a closure checklist for {query} with controls and required evidence.",
                f"What reporting pack fields should be in the monthly board pack for {query}?",
                f"How should breaches/exceptions for {query} be escalated and documented?",
                f"What stress-test scenarios are relevant for {query} and how to calibrate them?",
            ]
        ),
    )
