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

def _history_to_brief(history: List[Dict[str, str]] | None, max_pairs: int = 8) -> str:
    if not history: return ""
    turns = history[-(max_pairs * 2):]
    lines: List[str] = []
    for h in turns:
        role = h.get("role")
        content = (h.get("content") or "").strip()
        if not content: continue
        lines.append(("User: " if role == "user" else "Assistant: ") + content[:700])
    return "\n".join(lines)

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _parse_json(text: str) -> Dict[str, Any]:
    if not text: return {}
    text = _strip_code_fences(text)
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m: return {}
    raw = m.group(0)
    try:
        return json.loads(raw)
    except Exception:
        raw2 = re.sub(r",\s*}", "}", raw)
        raw2 = re.sub(r",\s*]", "]", raw2)
        try:
            return json.loads(raw2)
        except Exception:
            m2 = re.search(r'"raw_markdown"\s*:\s*"(.*)"\s*(,|\})', raw, flags=re.DOTALL)
            if m2:
                val = m2.group(1)
                val = val.replace(r"\\n", "\n").replace(r"\\t", "\t").replace(r"\\\"", "\"")
                return {"raw_markdown": val}
            return {}

def _mode_tokens(mode: str) -> int:
    return {"short": 900, "long": 2600, "research": 3600}.get(mode, 2200)

def _unescape_field(v: Optional[str]) -> Optional[str]:
    if not isinstance(v, str): return v
    if "\\n" in v and "\n" not in v:
        v = v.replace("\\n", "\n")
    return _strip_code_fences(v).strip()

def ask(
    query: str,
    *,
    user_id: Optional[str],
    history: Optional[List[Dict[str, str]]],
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

    # --- Web context (only if enabled; never raise) ---
    web_context = ""
    if web_enabled:
        try:
            results = ddg_search(query, max_results=max(8, k_hint))
        except Exception:
            results = []
        if results:
            lines = ["Web snippets (use prudently; internal docs take precedence):"]
            for i, r in enumerate(results, 1):
                title = r.get("title") or ""
                url = r.get("url") or ""
                snippet = (r.get("snippet") or "").strip()[:400]
                lines.append(f"{i}. {title} — {url}\n   Snippet: {snippet}")
            web_context = "\n".join(lines)

    schema_msg = (
        "Return ONE JSON object ONLY with keys: "
        "raw_markdown (string), summary (string), per_source (object), "
        "comparison_table_md (string, optional), follow_up_suggestions (array of strings). "
        "IMPORTANT: Use REAL newlines in raw_markdown and comparison_table_md; "
        "do NOT escape them as \\n. No prose outside JSON."
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

    try:
        resp = client.chat.completions.create(
            model=chat_model, temperature=0.35, top_p=0.95, max_tokens=max_out, messages=messages
        )
        text = resp.choices[0].message.content or ""
    except Exception as e:
        return RegulAIteAnswer(raw_markdown=f"### Error\nModel call failed.\n\nDetails: {e}")

    data = _parse_json(text)
    if data:
        if "raw_markdown" in data: data["raw_markdown"] = _unescape_field(data.get("raw_markdown"))
        if "summary" in data: data["summary"] = _unescape_field(data.get("summary"))
        if "comparison_table_md" in data: data["comparison_table_md"] = _unescape_field(data.get("comparison_table_md"))
        try:
            ans = RegulAIteAnswer(**data)
        except ValidationError:
            md = (data.get("raw_markdown") or "") if isinstance(data, dict) else ""
            ans = RegulAIteAnswer(raw_markdown=_unescape_field(md) or "")
    else:
        md = _strip_code_fences(text).strip()
        ans = RegulAIteAnswer(raw_markdown=_unescape_field(md) or "")

    if not (ans.raw_markdown or "").strip() and not (getattr(ans, "summary", "") or "").strip():
        return DEFAULT_EMPTY

    if not getattr(ans, "follow_up_suggestions", None):
        ans.follow_up_suggestions = [
            "Board approval thresholds for large exposures",
            "Monthly reporting checklist for large exposures",
            "Escalation steps for breaches and exceptions",
            "Stress-test scenarios for concentration risk",
            "KRIs and metrics for exposure concentration",
            "Differences CBB vs Basel: connected parties",
        ]

    raw_md = getattr(ans, "raw_markdown", "") or ""
    if "Approval Workflow" not in raw_md and "Reporting Matrix" not in raw_md:
        ans.raw_markdown = (
            f"{raw_md.rstrip()}\n\n"
            "### Approval Workflow\n"
            "Credit → Risk → Shari’ah Supervisory Board (if applicable) → Board → CBB notification\n"
            "\n### Reporting Matrix\n"
            "| Owner | Item | Frequency |\n"
            "|---|---|---|\n"
            "| Risk | Large exposure register | Monthly |\n"
            "| Compliance | CBB submissions | Quarterly |\n"
            "| Board | Connected exposure approvals | Ongoing |\n"
        )

    return ans
