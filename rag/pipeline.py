from __future__ import annotations
import os, json, re
from typing import Dict, Any, List, Optional, Union
from openai import OpenAI
from pydantic import ValidationError
from .schema import RegulAIteAnswer, DEFAULT_EMPTY
from .agents import build_system_instruction
from .router import normalize_mode
from .websearch import ddg_search
from .prompts import STYLE_GUIDE, FEW_SHOT_EXAMPLE

client = OpenAI()

# ---------- helpers ----------
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

def _schema_dict() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "raw_markdown": {"type": "string"},
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
            "comparison_table_md": {"type": "string"},
            "follow_up_suggestions": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["raw_markdown", "summary", "per_source", "follow_up_suggestions"],
        "additionalProperties": True,
    }

def _schema_prompt() -> str:
    return (
        "Return ONE JSON object that exactly matches this JSON Schema. "
        "No analysis before/after the JSON; no markdown outside string fields.\n\n"
        + json.dumps(_schema_dict(), ensure_ascii=False)
    )

def _parse_json(text: str) -> Dict[str, Any]:
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
    return {"short": 900, "long": 2600, "research": 3800}.get(mode, 2000)

# ---------- main ----------
def ask(
    query: str,
    *,
    user_id: str,
    history: List[Dict[str, str]],
    k_hint: int = 12,
    evidence_mode: bool = True,
    mode_hint: str | None = "long",
    web_enabled: Union[bool, str] = True,
    vec_id: Optional[str] = None,
    model: Optional[str] = None,
) -> RegulAIteAnswer:
    mode = normalize_mode(mode_hint)
    convo_brief = _history_to_brief(history)
    max_out = _mode_tokens(mode)

    sys_inst = build_system_instruction(k_hint=k_hint, evidence_mode=evidence_mode, mode=mode)

    # --- Web context always on ---
    web_context = ""
    results = ddg_search(query, max_results=k_hint)
    if results:
        lines = ["Web snippets (vector store takes precedence):"]
        for i, r in enumerate(results, 1):
            snippet = (r.get("snippet") or "").strip()[:400]
            title = r.get("title") or ""
            url = r.get("url") or ""
            lines.append(f"{i}. {title} — {url}\n   Snippet: {snippet}")
        web_context = "\n".join(lines)

    # --- Chat messages ---
    messages = [
        {"role": "system", "content": sys_inst},
        {"role": "system", "content": STYLE_GUIDE},
        {"role": "system", "content": FEW_SHOT_EXAMPLE},
        {"role": "system", "content": _schema_prompt()},
        {"role": "system", "content": (
            "Every answer MUST: "
            "1. Include a short approval workflow (e.g., Credit → Risk → Shari’ah Supervisory Board → Board → CBB). "
            "2. Include a reporting matrix (roles, items, frequency). "
            "3. Adapt tone/length to question context: if regulatory, be formal and detailed; if general/web, be flexible. "
            "4. Prefer long, ChatGPT-style narrative with headings, bullets, and tables when helpful."
        )},
        {"role": "user", "content": f"Conversation so far (brief):\n{convo_brief}"},
    ]
    if web_context:
        messages.append({"role": "user", "content": web_context})
    messages.append({"role": "user", "content": query})

    chat_model = (model or os.getenv("CHAT_MODEL") or os.getenv("RESPONSES_MODEL") or "gpt-4o-mini").strip()

    resp = client.chat.completions.create(
        model=chat_model,
        temperature=0.35,
        top_p=0.95,
        max_tokens=max_out,
        messages=messages,
    )

    text = ""
    try:
        text = resp.choices[0].message.content or ""
    except Exception:
        text = ""

    data = _parse_json(text or "")
    if not data:
        return DEFAULT_EMPTY
    try:
        return RegulAIteAnswer(**data)
    except ValidationError:
        return DEFAULT_EMPTY
