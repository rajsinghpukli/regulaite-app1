from __future__ import annotations
import os, json, re
from typing import Dict, Any, List, Optional
from openai import OpenAI
from pydantic import ValidationError
from .schema import RegulAIteAnswer, DEFAULT_EMPTY
from .agents import build_system_instruction, history_to_brief
from .router import normalize_mode
from .websearch import ddg_search

client = OpenAI()

# ---------- JSON schema + helpers -------------------------------------------
def _per_source_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["addressed", "not_found"]},
            "notes": {"type": "string"},
            "quotes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "framework": {"type": "string"},
                        "snippet": {"type": "string"},
                        "citation": {"type": "string"},
                    },
                    "required": ["framework", "snippet"],
                    "additionalProperties": False,
                },
            },
        },
        "additionalProperties": False,
    }

def _schema_dict() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "per_source": {
                "type": "object",
                "properties": {
                    "IFRS": _per_source_schema(),
                    "AAOIFI": _per_source_schema(),
                    "CBB": _per_source_schema(),
                    "InternalPolicy": _per_source_schema(),
                },
                "additionalProperties": False,
            },
            "comparative_analysis": {"type": "string"},
            "recommendation": {"type": "string"},
            "general_knowledge": {"type": "string"},
            "gaps_or_next_steps": {"type": "string"},
            "citations": {"type": "array", "items": {"type": "string"}},
            "ai_opinion": {"type": "string"},
            "follow_up_suggestions": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["summary", "per_source"],
        "additionalProperties": False,
    }

def _schema_prompt_block() -> str:
    # We enforce JSON via prompt so it works on any SDK version.
    return (
        "You MUST return a SINGLE JSON object that exactly matches this JSON Schema. "
        "Do not include any prose, markdown, or backticks—only the JSON object.\n\n"
        + json.dumps(_schema_dict(), ensure_ascii=False)
    )

def _parse_json_loose(text: str) -> Dict[str, Any]:
    # Extract the first top-level JSON object from the text
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {}
    raw = m.group(0)
    try:
        return json.loads(raw)
    except Exception:
        # Gentle cleanup in case of trailing commas
        raw = re.sub(r",\s*}", "}", raw)
        raw = re.sub(r",\s*]", "]", raw)
        try:
            return json.loads(raw)
        except Exception:
            return {}

def _pick_chat_model(model: str | None) -> str:
    # Some older SDKs/models only support Chat Completions well
    m = (model or os.getenv("RESPONSES_MODEL") or "gpt-4o-mini").strip()
    # If a 4.1* model is provided (responses-first), fall back to 4o-mini for chat
    if "4.1" in m:
        return "gpt-4o-mini"
    return m

# ---------- Main entry -------------------------------------------------------
def ask(
    query: str,
    *,
    user_id: str,
    history: List[Dict[str, str]],
    k_hint: int = 5,
    evidence_mode: bool = True,
    mode_hint: str | None = "auto",
    web_enabled: bool = False,
    vec_id: Optional[str] = None,   # kept for API compatibility, not used on old SDKs
    model: Optional[str] = None,
) -> RegulAIteAnswer:
    """
    Chat-completions based implementation (no attachments/tools) for maximum SDK compatibility.
    - Web search (beta): uses DuckDuckGo and passes top links as context.
    - Vector store: requires OpenAI 'attachments' on Responses API; since your SDK rejects it,
      we skip it here. Once your SDK is upgraded, we can re-enable that path.
    """
    mode = normalize_mode(mode_hint)
    sys_inst = build_system_instruction(k_hint=k_hint, evidence_mode=evidence_mode, mode=mode)
    convo_brief = history_to_brief(history)

    web_context = ""
    if web_enabled:
        results = ddg_search(query, max_results=min(6, max(3, k_hint)))
        if results:
            lines = [f"{i+1}. {t} — {u}" for i, (t, u) in enumerate(results)]
            web_context = "Relevant web links:\n" + "\n".join(lines)

    # Build messages for Chat Completions
    messages = [
        {"role": "system", "content": sys_inst},
        {"role": "system", "content": _schema_prompt_block()},
        {"role": "user", "content": f"Conversation so far (brief):\n{convo_brief}"},
    ]
    if web_context:
        messages.append({"role": "user", "content": web_context})
    messages.append({"role": "user", "content": query})

    chat_model = _pick_chat_model(model)

    resp = client.chat.completions.create(
        model=chat_model,
        temperature=0.2,
        messages=messages,
    )

    text = ""
    try:
        text = resp.choices[0].message.content or ""
    except Exception:
        text = ""

    data = _parse_json_loose(text or "")
    if not data:
        return DEFAULT_EMPTY
    try:
        return RegulAIteAnswer(**data)
    except ValidationError:
        return DEFAULT_EMPTY
