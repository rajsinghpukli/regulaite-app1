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

# ---- History brief kept local to avoid import drift -------------------------
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

# ---- JSON schema helpers ----------------------------------------------------
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
    return (
        "You MUST return a SINGLE JSON object that exactly matches this JSON Schema. "
        "Do not include any prose, markdown, or backticks—only the JSON object.\n\n"
        + json.dumps(_schema_dict(), ensure_ascii=False)
    )

def _parse_json_loose(text: str) -> Dict[str, Any]:
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
    return {"short": 900, "long": 2600, "research": 3500}.get(mode, 1500)

# ---- Main orchestrator ------------------------------------------------------
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
    Preferred path: OpenAI Responses API with vector-store attachments (if supported by SDK).
    Fallback: Chat Completions with web snippets only (soft evidence).
    """
    mode = normalize_mode(mode_hint)
    convo_brief = _history_to_brief(history)
    max_out = _mode_tokens(mode)

    # Build web context (used in both paths; in Responses it helps the model too)
    web_context = ""
    if web_enabled:
        results = ddg_search(query, max_results=min(8, max(4, k_hint)))
        if results:
            lines = ["Relevant web snippets (use for evidence if reliable):"]
            for i, r in enumerate(results, 1):
                snippet = (r.get("snippet") or "").strip()[:350]
                title = r.get("title") or ""
                url = r.get("url") or ""
                lines.append(f"{i}. {title} — {url}\n   Snippet: {snippet}")
            web_context = "\n".join(lines)

    # ---- Try Responses API with attachments (vector store) ------------------
    responses_model = (model or os.getenv("RESPONSES_MODEL") or "gpt-4.1-mini").strip()
    try:
        sys_inst = build_system_instruction(
            k_hint=k_hint,
            evidence_mode=evidence_mode,
            mode=mode,
            soft_evidence=False if vec_id else True,  # if no VS, soften evidence rules
        )

        tools = [{"type": "web_search"}] if web_enabled else None
        attachments = [{"vector_store_id": vec_id, "file_search": {"max_num_results": int(k_hint)}}] if vec_id else None

        resp = client.responses.create(
            model=responses_model,
            input=[
                {"role": "system", "content": sys_inst},
                {"role": "system", "content": _schema_prompt_block()},
                {"role": "user", "content": f"Conversation so far (brief):\n{convo_brief}"},
                {"role": "user", "content": web_context} if web_context else None,
                {"role": "user", "content": query},
            ],
            response_format={"type": "json_schema", "json_schema": {"name": "RegulAIteAnswer", "schema": _schema_dict(), "strict": True}},
            tools=tools,
            attachments=attachments,
            max_output_tokens=max_out,
            metadata={"app": "RegulAIte", "user": user_id},
        )

        text = getattr(resp, "output_text", None)
        if not text:
            parts = []
            for block in getattr(resp, "output", []):
                for c in getattr(block, "content", []):
                    if getattr(c, "type", None) == "output_text":
                        parts.append(getattr(c, "text", {}).get("value", ""))
            text = "\n".join(parts)

        data = _parse_json_loose(text or "")
        if data:
            return RegulAIteAnswer(**data)
    except Exception:
        # If anything goes wrong (older SDK, param mismatch, etc.), fall through to chat path
        pass

    # ---- Fallback: Chat Completions (no VS) --------------------------------
    chat_model = (os.getenv("CHAT_MODEL") or "gpt-4o-mini").strip()
    sys_inst = build_system_instruction(
        k_hint=k_hint,
        evidence_mode=evidence_mode,
        mode=mode,
        soft_evidence=True,  # soften because we don't have VS here
    )
    messages = [
        {"role": "system", "content": sys_inst},
        {"role": "system", "content": _schema_prompt_block()},
        {"role": "user", "content": f"Conversation so far (brief):\n{convo_brief}"},
    ]
    if web_context:
        messages.append({"role": "user", "content": web_context})
    messages.append({"role": "user", "content": query})

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

    data = _parse_json_loose(text or "")
    if not data:
        return DEFAULT_EMPTY
    try:
        return RegulAIteAnswer(**data)
    except ValidationError:
        return DEFAULT_EMPTY
