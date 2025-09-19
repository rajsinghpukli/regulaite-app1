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

# ----------------- Helpers -----------------
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
    # Mirrors RegulAIteAnswer (no "status"; frameworks omitted if empty)
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
        "No markdown outside string fields; do not include any analysis before/after the JSON.\n\n"
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
    # Generous budgets for long/research so it feels like ChatGPT
    return {"short": 900, "long": 2600, "research": 3800}.get(mode, 1600)

# ----------------- Main -----------------
def ask(
    query: str,
    *,
    user_id: str,
    history: List[Dict[str, str]],
    k_hint: int = 5,
    evidence_mode: bool = True,
    mode_hint: str | None = "auto",
    web_enabled: bool = False,
    vec_id: Optional[str] = None,   # Ignored on this SDK path (attachments not supported)
    model: Optional[str] = None,
) -> RegulAIteAnswer:
    """
    Chat Completions path (no attachments). This avoids SDK keyword errors and
    produces long, ChatGPT-style answers with a comparison table in long/research.
    Frameworks with no evidence should be omitted (not marked 'not_found').
    """
    mode = normalize_mode(mode_hint)
    convo_brief = _history_to_brief(history)
    max_out = _mode_tokens(mode)

    # Build the main system instruction from agents.py (no soft_evidence kwarg).
    sys_inst = build_system_instruction(
        k_hint=k_hint,
        evidence_mode=evidence_mode,
        mode=mode,
    )

    # Inline patch: emphasize NO "not_found" spam and “ChatGPT-style” depth.
    soft_evidence_patch = (
        "Important: Do NOT say 'not found'. If you lack evidence for a framework, "
        "omit that framework key from per_source. Provide long, flowing, natural guidance "
        "like a senior CRO would. In long/research modes, include a substantial comparison "
        "table (comparison_table_md) and an implementation checklist."
    )

    # Optional web context (gives snippets + links to anchor facts)
    web_context = ""
    if web_enabled:
        results = ddg_search(query, max_results=min(10, max(6, k_hint)))
        if results:
            lines = ["Relevant web snippets (use prudently for evidence):"]
            for i, r in enumerate(results, 1):
                snippet = (r.get("snippet") or "").strip()[:400]
                title = r.get("title") or ""
                url = r.get("url") or ""
                lines.append(f"{i}. {title} — {url}\n   Snippet: {snippet}")
            web_context = "\n".join(lines)

    # Build messages
    messages = [
        {"role": "system", "content": sys_inst},
        {"role": "system", "content": soft_evidence_patch},
        {"role": "system", "content": _schema_block()},
        {"role": "user", "content": f"Conversation so far (brief):\n{convo_brief}"},
    ]
    if web_context:
        messages.append({"role": "user", "content": web_context})
    messages.append({"role": "user", "content": query})

    chat_model = (os.getenv("CHAT_MODEL") or os.getenv("RESPONSES_MODEL") or "gpt-4o-mini").strip()

    resp = client.chat.completions.create(
        model=chat_model,
        temperature=0.2,
        max_tokens=max_out,
        messages=messages,
    )

    text = ""
    try:
        text = resp.choices[0].message.content or ""
    except Exception:
        text = ""

    data = _parse_first_json(text or "")
    if not data:
        return DEFAULT_EMPTY
    try:
        return RegulAIteAnswer(**data)
    except ValidationError:
        return DEFAULT_EMPTY
