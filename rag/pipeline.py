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

def _history_to_brief(history: List[Dict[str, str]], max_pairs: int = 8) -> str:
    if not history:
        return ""
    turns = history[-(max_pairs * 2):]
    lines: List[str] = []
    for h in turns:
        role = h.get("role")
        content = (h.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            lines.append(f"User asked: {content}")
        else:
            lines.append(f"Assistant replied (extract): {content[:700]}")
    return "\n".join(lines[-(max_pairs * 2):])

def _json_schema() -> Dict[str, Any]:
    # Pydantic schema mirror (kept minimal/strict)
    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "per_source": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "notes": {"type": "string"},
                        "quotes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "framework": {"type": "string", "enum": ["IFRS","AAOIFI","CBB","InternalPolicy"]},
                                    "snippet": {"type": "string"},
                                    "citation": {"type": "string"},
                                },
                                "required": ["framework", "snippet"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "additionalProperties": False,
                },
            },
            "comparative_analysis": {"type": "string"},
            "recommendation": {"type": "string"},
            "general_knowledge": {"type": "string"},
            "gaps_or_next_steps": {"type": "string"},
            "citations": {"type": "array", "items": {"type": "string"}},
            "ai_opinion": {"type": "string"},
            "follow_up_suggestions": {"type": "array", "items": {"type": "string"}},
            "comparison_table_md": {"type": "string"},
        },
        "required": ["summary", "per_source"],
        "additionalProperties": False,
    }

def _schema_block() -> str:
    return (
        "Return a SINGLE JSON object that exactly matches this JSON Schema. "
        "No markdown outside string fields; no extra keys; do not include analysis text before or after the JSON.\n\n"
        + json.dumps(_json_schema(), ensure_ascii=False)
    )

def _parse_first_json(text: str) -> Dict[str, Any]:
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {}
    raw = m.group(0)
    try:
        return json.loads(raw)
    except Exception:
        raw = re.sub(r",\s*}", "}", raw)
        raw = re.sub(r",\s*]", "]", raw)
        try:
            return json.loads(raw)
        except Exception:
            return {}

def _mode_tokens(mode: str) -> int:
    return {"short": 1000, "long": 3000, "research": 4200}.get(mode, 1800)

def ask(
    query: str,
    *,
    user_id: str,
    history: List[Dict[str, str]],
    k_hint: int = 5,
    evidence_mode: bool = True,
    mode_hint: str | None = "auto",
    web_enabled: bool = False,
    vec_id: Optional[str] = None,
    model: Optional[str] = None,
) -> RegulAIteAnswer:
    """
    Responses API with vector-store attachments ONLY.
    If attachments are not supported or the vector store id is missing, this will raise.
    """
    if not vec_id:
        # Hard requirement per your request: no fallback.
        return DEFAULT_EMPTY

    mode = normalize_mode(mode_hint)
    responses_model = (model or os.getenv("RESPONSES_MODEL") or "gpt-4.1-mini").strip()
    convo_brief = _history_to_brief(history)
    max_out = _mode_tokens(mode)

    sys_inst = build_system_instruction(k_hint=k_hint, evidence_mode=evidence_mode, mode=mode)

    # Web context (secondary to VS)
    web_context = ""
    if web_enabled:
        results = ddg_search(query, max_results=min(8, max(4, k_hint)))
        if results:
            lines = ["Relevant web snippets (secondary to attached docs):"]
            for i, r in enumerate(results, 1):
                snippet = (r.get("snippet") or "").strip()[:350]
                title = r.get("title") or ""
                url = r.get("url") or ""
                lines.append(f"{i}. {title} — {url}\n   Snippet: {snippet}")
            web_context = "\n".join(lines)

    attachments = [{"vector_store_id": vec_id, "file_search": {"max_num_results": int(k_hint)}}]

    resp = client.responses.create(
        model=responses_model,
        input=[
            {"role": "system", "content": sys_inst},
            {"role": "system", "content": _schema_block()},
            {"role": "user", "content": f"Conversation so far (brief):\n{convo_brief}"},
            {"role": "user", "content": web_context} if web_context else None,
            {"role": "user", "content": query},
        ],
        # Strict schema forces long, structured answers that our renderer understands
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "RegulAIteAnswer",
                "schema": _json_schema(),
                "strict": True,
            },
        },
        tools=[{"type": "web_search"}] if web_enabled else None,
        attachments=attachments,
        max_output_tokens=max_out,
        metadata={"app": "RegulAIte", "user": user_id},
    )

    # Extract text from Responses API objects
    text = getattr(resp, "output_text", None)
    if not text:
        parts = []
        for block in getattr(resp, "output", []):
            for c in getattr(block, "content", []):
                if getattr(c, "type", None) == "output_text":
                    parts.append(getattr(c, "text", {}).get("value", ""))
        text = "\n".join(parts)

    data = _parse_first_json(text or "")
    if not data:
        return DEFAULT_EMPTY
    try:
        return RegulAIteAnswer(**data)
    except ValidationError:
        return DEFAULT_EMPTY
