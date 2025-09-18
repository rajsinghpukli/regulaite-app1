from __future__ import annotations
import os, json, re
from typing import Dict, Any, List, Optional
from openai import OpenAI
from pydantic import ValidationError
from .schema import RegulAIteAnswer, DEFAULT_EMPTY
from .agents import build_system_instruction   # ← only this import from agents
from .router import normalize_mode
from .websearch import ddg_search

client = OpenAI()

# ---------- Local helper to avoid cross-file import issues -------------------
def _history_to_brief(history: List[Dict[str, str]], max_pairs: int = 8) -> str:
    """
    Compress recent turns into a short brief fed to the model.
    Keeps up to `max_pairs` Q/A pairs (2*max_pairs turns).
    """
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
    # Enforce JSON via prompt (keeps compatibility with older SDKs)
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

def _pick_chat_model(model: str | None) -> str:
    # Use Chat Completions for max SDK compatibility
    m = (model or os.getenv("RESPONSES_MODEL") or "gpt-4o-mini").strip()
    if "4.1" in m:
        return "gpt-4o-mini"
    return m

def _mode_tokens(mode: str) -> int:
    if mode == "short":
        return 800
    if mode == "long":
        return 1800
    if mode == "research":
        return 3000
    return 1200

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
    vec_id: Optional[str] = None,  # kept for compatibility; not used on older SDKs
    model: Optional[str] = None,
) -> RegulAIteAnswer:
    """
    Chat-completions implementation (SDK-safe). Vector store is skipped until your
    SDK supports Responses attachments. Web search injects real snippets/links.
    Mode (short/long/research) meaningfully changes length and structure.
    """
    mode = normalize_mode(mode_hint)
    sys_inst = build_system_instruction(k_hint=k_hint, evidence_mode=evidence_mode, mode=mode)
    convo_brief = _history_to_brief(history)

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

    messages = [
        {"role": "system", "content": sys_inst},
        {"role": "system", "content": _schema_prompt_block()},
        {"role": "user", "content": f"Conversation so far (brief):\n{convo_brief}"},
    ]
    if web_context:
        messages.append({"role": "user", "content": web_context})
    messages.append({"role": "user", "content": query})

    chat_model = _pick_chat_model(model)
    max_tokens = _mode_tokens(mode)

    resp = client.chat.completions.create(
        model=chat_model,
        temperature=0.2,
        max_tokens=max_tokens,
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
