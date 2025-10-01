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
            m2 = re.search(r'"raw_markdown"\s*:\s*"(.*)"\s*(,|\})', raw, flags=re.DOTALL)
            if m2:
                val = m2.group(1)
                val = val.replace(r"\\n", "\n").replace(r"\\t", "\t").replace(r"\\\"", "\"")
                return {"raw_markdown": val}
            return {}

def _mode_tokens(mode: str) -> int:
    return {"short": 900, "long": 2600, "research": 3600}.get(mode, 2200)

# --------- NEW: light intent detection (formatting only) ----------
def _detect_intent(q: str) -> Dict[str, bool]:
    ql = (q or "").lower()

    return_only = any(kw in ql for kw in [
        "return only", "only:", "only the", "just the", "no other", "nothing else",
        "no internal", "no prose", "no explanation",
    ])
    quote_only = any(kw in ql for kw in [
        "quote", "quote verbatim", "verbatim", "exact sentence", "exact line",
        "cite-only", "cite only",
    ])
    list_ids = any(kw in ql for kw in [
        "list only the section ids", "list only the ids", "list only ids",
        "section ids present", "ids present", "cm-5.2.x", "cm-5.3.x"
    ])
    bis_only = ("bis.org" in ql or "bis " in ql or "bcbs " in ql) and return_only
    scenario = any(kw in ql for kw in [
        "scenario", "deliver:", "board-ready", "controls", "kris", "workflow",
        "recommendation", "decision-grade", "exposure calculation",
    ])

    concise = return_only or quote_only or list_ids or bis_only
    return {
        "return_only": return_only,
        "quote_only": quote_only,
        "list_ids": list_ids,
        "bis_only": bis_only,
        "scenario": scenario,
        "concise": concise,
    }

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
    Produce ChatGPT-style answers that adapt to the question.
    - If the user asks for 'URL only', 'quote verbatim', 'IDs only', etc., respond concisely.
    - For scenarios/board memos, keep structured outputs.
    - No more forced 'Approval Workflow / Reporting Matrix' unless the scenario implies it.
    """
    mode = normalize_mode(mode_hint)
    convo_brief = _history_to_brief(history)
    max_out = _mode_tokens(mode)
    intent = _detect_intent(query)

    sys_inst = build_system_instruction(
        k_hint=k_hint,
        evidence_mode=evidence_mode,
        mode=mode,
    )

    # --- Web context (always on) ---
    web_context = ""
    try:
        results = ddg_search(query, max_results=max(8, k_hint))
    except Exception:
        results = []
    if results:
        lines = ["Web snippets (use prudently; internal docs take precedence):"]
        for i, r in enumerate(results, 1):
            title = r.get("title") or ""
            url = r.get("url") or r.get("href") or ""
            snippet = (r.get("snippet") or r.get("body") or "").strip()[:400]
            lines.append(f"{i}. {title} — {url}\n   Snippet: {snippet}")
        web_context = "\n".join(lines)

    # --- Adaptive formatting instruction ---
    if intent["concise"]:
        # User wants minimal output
        style_msg = (
            "Follow the user's output restriction STRICTLY. "
            "If they ask for 'quote verbatim', return only the quoted line(s) and the exact section IDs. "
            "If they ask for 'URL only' or 'return only', give exactly those fields, with no extra prose, no headings. "
            "No preamble, no closing text."
        )
        schema_msg = None  # let the model answer freely; we'll accept plain text
    elif intent["scenario"]:
        style_msg = (
            "This is a board-grade scenario. Use clean headings, tables as needed, and include controls/KRIs/workflow "
            "IF relevant to the user's deliverables. Avoid boilerplate. Keep it concise and decision-focused."
        )
        # keep JSON so downstream renderers can still show tables nicely if provided
        schema_msg = (
            "Return ONE JSON object ONLY with keys: "
            "raw_markdown (string), summary (string, optional), per_source (object, optional), "
            "comparison_table_md (string, optional), follow_up_suggestions (array of strings, optional). "
            "No prose outside JSON."
        )
    else:
        style_msg = (
            "Answer naturally in well-structured Markdown. Use headings and tables if helpful. "
            "Do NOT add 'Approval Workflow' or 'Reporting Matrix' unless the question clearly requires them."
        )
        schema_msg = (
            "Return ONE JSON object ONLY with keys: "
            "raw_markdown (string), summary (string, optional), per_source (object, optional), "
            "comparison_table_md (string, optional), follow_up_suggestions (array of strings, optional). "
            "No prose outside JSON."
        )

    messages = [
        {"role": "system", "content": sys_inst},
        {"role": "system", "content": STYLE_GUIDE},
        {"role": "system", "content": FEW_SHOT_EXAMPLE},
        {"role": "system", "content": style_msg},
        {"role": "user", "content": f"Conversation so far (brief):\n{convo_brief}"},
    ]
    if web_context:
        messages.append({"role": "user", "content": web_context})
    messages.append({"role": "user", "content": query})
    if schema_msg:
        messages.insert(3, {"role": "system", "content": schema_msg})

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

    # For concise intents, accept plain text directly
    if intent["concise"]:
        raw = _strip_code_fences(text).strip()
        if not raw:
            return DEFAULT_EMPTY
        return RegulAIteAnswer(raw_markdown=raw)

    # Otherwise, try parse JSON; fallback to clean Markdown
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

    # IMPORTANT: remove the old forced append of 'Approval Workflow' and 'Reporting Matrix'.
    # If a scenario specifically asked for these, the model will include them; otherwise we won't.

    # Ensure follow-ups present (optional; harmless)
    if not ans.follow_up_suggestions:
        topic = (query or "this topic").strip()
        ans.follow_up_suggestions = [
            f"What approval thresholds and board oversight apply to {topic}?",
            f"Draft a closure checklist for {topic} with controls and required evidence.",
            f"What fields belong in the monthly board pack for {topic}?",
            f"How should breaches/exceptions for {topic} be escalated and documented?",
            f"What stress-test scenarios are relevant for {topic} and how to calibrate them?",
            f"What are the key risks, controls, and KRIs for {topic} (with metrics)?",
        ]

    return ans
