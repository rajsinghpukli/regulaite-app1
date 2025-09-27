from __future__ import annotations
import os, json, re
from typing import Dict, Any, List, Optional, Union
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
            lines.append(f"User: {content}")
        else:
            lines.append(f"Assistant: {content[:600]}")
    return "\n".join(lines[-(max_pairs * 2):])

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
    return {"short": 900, "long": 2600, "research": 3800}.get(mode, 1600)

def _auto_enable_web(query: str, mode: str) -> bool:
    q = (query or "").lower()
    if mode == "research":
        return True
    for t in ("latest", "recent", "update", "today", "news"):
        if t in q:
            return True
    if "http://" in q or "https://" in q:
        return True
    return False

def ask(
    query: str,
    *,
    user_id: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    mode: str = "long",
    k_hint: int = 12,
    evidence_mode: bool = True,
    web_enabled: Union[bool, str] = True,
    vec_id: Optional[str] = None,
    model: Optional[str] = None,
) -> RegulAIteAnswer:
    mode = normalize_mode(mode)
    convo_brief = _history_to_brief(history or [])
    max_out = _mode_tokens(mode)

    sys_inst = build_system_instruction(
        k_hint=k_hint,
        evidence_mode=evidence_mode,
        mode=mode,
    )

    use_web = True if web_enabled else _auto_enable_web(query, mode)
    web_context = ""
    if use_web:
        results = ddg_search(query, max_results=10)
        if results:
            lines = ["Web snippets (vector store still primary):"]
            for i, r in enumerate(results, 1):
                snippet = (r.get("snippet") or "").strip()[:400]
                url = r.get("url") or ""
                lines.append(f"{i}. {snippet} ({url})")
            web_context = "\n".join(lines)

    messages = [
        {"role": "system", "content": sys_inst},
        {"role": "user", "content": f"Conversation so far:\n{convo_brief}"},
    ]
    if web_context:
        messages.append({"role": "user", "content": web_context})
    messages.append({"role": "user", "content": query})

    chat_model = (model or os.getenv("RESPONSES_MODEL") or "gpt-4.1-mini").strip()

    resp = client.chat.completions.create(
        model=chat_model,
        temperature=0.3,
        max_tokens=max_out,
        messages=messages,
    )

    text = ""
    try:
        text = resp.choices[0].message.content or ""
    except Exception:
        text = ""

    data = _parse_json(text or "")
    if data:
        try:
            return RegulAIteAnswer(**data)
        except ValidationError:
            pass

    # Fallback: return readable ChatGPT-like text
    if text.strip():
        return RegulAIteAnswer(raw_markdown=text.strip())
    return DEFAULT_EMPTY
