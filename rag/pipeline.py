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
            "per_source": {"type": "object"},
            "comparison_table_md": {"type": "string"},
            "follow_up_suggestions": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["raw_markdown", "summary", "per_source", "follow_up_suggestions"],
    }

def _schema_prompt() -> str:
    return "Return ONE JSON object that matches this schema only:\n" + json.dumps(_schema_dict())

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
    return {"short": 900, "long": 2600, "research": 3800}.get(mode, 2200)

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

    web_context = ""
    results = ddg_search(query, max_results=k_hint)
    if results:
        lines = ["Web snippets (vector store is primary):"]
        for i, r in enumerate(results, 1):
            snippet = (r.get("snippet") or "").strip()[:400]
            title = r.get("title") or ""
            url = r.get("url") or ""
            lines.append(f"{i}. {title} — {url}\n   Snippet: {snippet}")
        web_context = "\n".join(lines)

    messages = [
        {"role": "system", "content": sys_inst},
        {"role": "system", "content": STYLE_GUIDE},
        {"role": "system", "content": FEW_SHOT_EXAMPLE},
        {"role": "system", "content": _schema_prompt()},
        {"role": "system", "content": (
            "Every answer must be a long, ChatGPT-style narrative with:\n"
            "- Headings & subheadings\n"
            "- Evidence quotes with inline citations\n"
            "- A comparison table (if >1 framework)\n"
            "- A short approval workflow (Credit → Risk → Shari’ah Board → Board → CBB)\n"
            "- A reporting matrix (Owner | Item | Frequency)\n"
            "- 4–6 follow-up questions\n"
            "Do not output plain prose outside JSON."
        )},
        {"role": "user", "content": f"Conversation so far:\n{convo_brief}"},
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

    text = resp.choices[0].message.content or ""
    data = _parse_json(text)
    if not data:
        # Retry once with stronger JSON warning
        retry_messages = messages + [
            {"role": "system", "content": "⚠️ WARNING: You must return ONLY one valid JSON object. No markdown outside JSON."}
        ]
        resp2 = client.chat.completions.create(
            model=chat_model,
            temperature=0.35,
            max_tokens=max_out,
            messages=retry_messages,
        )
        text = resp2.choices[0].message.content or ""
        data = _parse_json(text)

    if not data:
        return DEFAULT_EMPTY
    try:
        return RegulAIteAnswer(**data)
    except ValidationError:
        return DEFAULT_EMPTY
