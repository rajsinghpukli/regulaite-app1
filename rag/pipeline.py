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
def _history_to_brief(history: List[Dict[str, str]] | None, max_pairs: int = 8) -> str:
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
            # small extract to keep prompt size controlled
            lines.append(f"Assistant: {content[:700]}")
    return "\n".join(lines)

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _parse_json(text: str) -> Dict[str, Any]:
    """Extract JSON object if present; tolerate minor trailing commas & code fences."""
    if not text:
        return {}
    text = _strip_code_fences(text)
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {}
    raw = m.group(0)
    try:
        return json.loads(raw)
    except Exception:
        raw2 = re.sub(r",\s*}", "}", raw)
        raw2 = re.sub(r",\s*]", "]", raw2)
        try:
            return json.loads(raw2)
        except Exception:
            # salvage raw_markdown if possible
            m2 = re.search(r'"raw_markdown"\s*:\s*"(.*)"\s*(,|\})', raw, flags=re.DOTALL)
            if m2:
                val = m2.group(1)
                val = val.replace(r"\\n", "\n").replace(r"\\t", "\t").replace(r"\\\"", "\"")
                return {"raw_markdown": val}
            return {}

def _mode_tokens(mode: str) -> int:
    return {"short": 900, "long": 2600, "research": 3600}.get(mode, 2200)

# ---------- main ----------
def ask(
    query: str,
    *,
    user_id: Optional[str],
    history: Optional[List[Dict[str, str]]],
    k_hint: int = 12,
    evidence_mode: bool = True,
    mode_hint: str | None = "long",
    web_enabled: Union[bool, str] = True,   # kept for compat; we still always do web
    vec_id: Optional[str] = None,
    model: Optional[str] = None,
) -> RegulAIteAnswer:
    """
    Produce a long, ChatGPT-style answer in raw_markdown with headings, tables, workflow,
    reporting matrix, and follow-up suggestions. If JSON parse fails, gracefully fall back
    to the assistant text (never blank).
    """
    mode = normalize_mode(mode_hint)
    convo_brief = _history_to_brief(history)
    max_out = _mode_tokens(mode)

    sys_inst = build_system_instruction(
        k_hint=k_hint,
        evidence_mode=evidence_mode,
        mode=mode,
    )

    # --- Web context (always on) ---
    web_context = ""
    results = ddg_search(query, max_results=max(8, k_hint))
    if results:
        lines = ["Web snippets (use prudently; internal docs take precedence):"]
        for i, r in enumerate(results, 1):
            title = r.get("title") or ""
            url = r.get("url") or r.get("href") or ""
            snippet = (r.get("snippet") or r.get("body") or "").strip()[:400]
            lines.append(f"{i}. {title} — {url}\n   Snippet: {snippet}")
        web_context = "\n".join(lines)

    # --- Schema instruction (we'll still render Markdown even if JSON fails) ---
    schema_msg = (
        "Return ONE JSON object ONLY with keys:\n"
        "raw_markdown (string), summary (string), per_source (object), "
        "comparison_table_md (string, optional), follow_up_suggestions (array of strings).\n"
        "No prose outside JSON."
    )

    messages = [
        {"role": "system", "content": sys_inst},
        {"role": "system", "content": STYLE_GUIDE},
        {"role": "system", "content": FEW_SHOT_EXAMPLE},
        {"role": "system", "content": schema_msg},
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

    # Try parse JSON; fallback to clean Markdown
    data = _parse_json(text)
    if data:
        try:
            ans = RegulAIteAnswer(**data)
        except ValidationError:
            md = data.get("raw_markdown") or ""
            ans = RegulAIteAnswer(raw_markdown=md.strip() if md else "")
    else:
        md = _strip_code_fences(text).strip()
        ans = RegulAIteAnswer(raw_markdown=md if md else "")

    if not (ans.raw_markdown or "").strip():
        return DEFAULT_EMPTY

    # Ensure follow-ups present
    if not ans.follow_up_suggestions:
        topic = (query or "this topic").strip()
        ans.follow_up_suggestions = [
            f"What are approval thresholds and board oversight for {topic}?",
            f"Draft a closure checklist for {topic} with controls and required evidence.",
            f"What reporting pack fields should be in the monthly board pack for {topic}?",
            f"How should breaches/exceptions for {topic} be escalated and documented?",
            f"What stress-test scenarios are relevant for {topic} and how to calibrate them?",
            f"What are the key risks, controls, and KRIs for {topic} (with metrics)?",
        ]

    # Soft-ensure the “Recommendation / Workflow / Reporting matrix” appear
    if "Approval Workflow" not in ans.raw_markdown and "Reporting Matrix" not in ans.raw_markdown:
        ans.raw_markdown += (
            "\n\n### Approval Workflow\n"
            "Credit → Risk → Shari’ah Supervisory Board (if applicable) → Board → CBB notification\n"
            "\n### Reporting Matrix\n"
            "| Owner | Item | Frequency |\n"
            "|---|---|---|\n"
            "| Risk | Large exposure register | Monthly |\n"
            "| Compliance | CBB submissions | Quarterly |\n"
            "| Board | Connected exposure approvals | Ongoing |\n"
        )

    return ans
