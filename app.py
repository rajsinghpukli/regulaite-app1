# app.py
from __future__ import annotations
import os
import json
import re
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv

from rag.pipeline import ask
from rag.schema import RegulAIteAnswer
from rag.persist import load_chat, save_chat, append_turn, clear_chat

load_dotenv()

APP_NAME = "RegulAIte — Regulatory Assistant (Pilot)"
DEFAULT_MODEL = os.getenv("RESPONSES_MODEL", "gpt-4.1-mini")
VECTOR_STORE_ID = os.getenv("OPENAI_VECTOR_STORE_ID", "").strip()
LLM_KEY_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))

# ---------- Simple preset logins ----------
PRESET_USERS = {
    "guest1": "pass1",
    "guest2": "pass2",
    "guest3": "pass3",
}

st.set_page_config(page_title=APP_NAME, page_icon="🧭", layout="wide")

# ---------- CSS ----------
CSS = """
<style>
.block-container { max-width: 1180px; }
.badge {
  display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;margin-right:6px;
  border:1px solid rgba(0,0,0,0.1)
}
.badge.ok { background:#ecfdf5;color:#065f46;border-color:#10b98133; }
.badge.warn { background:#fff7ed;color:#9a3412;border-color:#f59e0b33; }
.badge.err { background:#fef2f2;color:#991b1b;border-color:#ef444433; }
.regu-msg { border-radius:14px;padding:14px 16px;box-shadow:0 1px 2px rgba(0,0,0,0.06);
  border:1px solid rgba(0,0,0,0.06); margin-bottom:10px; }
.regu-user { background:#f5f7fb; }
.regu-assistant { background:#ffffff; }
.markdown-body h1, .markdown-body h2, .markdown-body h3 { margin-top: 1.2rem; }
.markdown-body p, .markdown-body li { line-height: 1.6; }
.markdown-body table { width:100%; border-collapse:collapse;}
.markdown-body th, .markdown-body td { border:1px solid #e5e7eb; padding:8px; font-size:14px;}
.markdown-body th { background:#f9fafb; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------- Session state ----------
if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False
if "user_id" not in st.session_state:
    st.session_state.user_id = ""
if "history" not in st.session_state:
    st.session_state.history: List[Dict[str, str]] = []
if "last_answer" not in st.session_state:
    st.session_state.last_answer: RegulAIteAnswer | None = None

# ---------- Helpers ----------
def _looks_json_like(s: str) -> bool:
    s = s.strip()
    return (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]"))

def _unescape_backslash_newlines(text: str) -> str:
    # If the string literally contains "\n", map to real newlines
    # but avoid double-unescaping actual newlines
    if "\\n" in text and "\n" not in text:
        text = text.replace("\\n", "\n")
    return text

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _format_per_source(per_source: Dict[str, Any]) -> str:
    if not isinstance(per_source, dict) or not per_source:
        return ""
    lines = ["## Evidence by Framework"]
    for fw, quotes in per_source.items():
        lines.append(f"**{fw}**")
        if isinstance(quotes, list):
            for q in quotes:
                q_str = str(q)
                q_str = _unescape_backslash_newlines(q_str)
                lines.append(f"- {q_str.strip()}")
    return "\n".join(lines)

def _coerce_answer_to_markdown(ans: RegulAIteAnswer) -> str:
    """
    1) Use ans.as_markdown() if it returns clean text.
    2) If it looks JSON-ish or has visible \n, sanitize.
    3) If still empty, compose from fields we know.
    """
    md = ""
    try:
        md = ans.as_markdown() or ""
    except Exception:
        md = ""

    md = _strip_code_fences(md)
    if _looks_json_like(md):
        # Someone returned JSON inside raw_markdown; ignore and rebuild below
        md = ""

    if md:
        return _unescape_backslash_newlines(md).strip()

    # Build from parts if needed
    parts: List[str] = []
    summary = getattr(ans, "summary", "") or ""
    if summary.strip():
        parts.append("## Summary")
        parts.append(_unescape_backslash_newlines(summary.strip()))

    cmp_md = getattr(ans, "comparison_table_md", "") or ""
    if cmp_md.strip():
        parts.append("## Comparison")
        parts.append(_unescape_backslash_newlines(cmp_md.strip()))

    per_source = getattr(ans, "per_source", {}) or {}
    ps = _format_per_source(per_source)
    if ps:
        parts.append(ps)

    raw_md = getattr(ans, "raw_markdown", "") or ""
    if raw_md.strip():
        parts.append(_unescape_backslash_newlines(_strip_code_fences(raw_md.strip())))

    # Fallback message
    if not parts:
        parts.append("_No answer produced._")

    return "\n\n".join(parts).strip()

def render_message(role: str, content_md: str):
    klass = "regu-user" if role == "user" else "regu-assistant"
    st.markdown(f'<div class="regu-msg {klass}">', unsafe_allow_html=True)
    st.markdown(f'<div class="markdown-body">{content_md}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Login ----------
def auth_ui():
    st.markdown("## 🔐 RegulAIte Login")
    with st.form("login_form"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        ok = st.form_submit_button("Login")
        if ok:
            if u in PRESET_USERS and PRESET_USERS[u] == p:
                st.session_state.auth_ok = True
                st.session_state.user_id = u
                # load per-user memory
                st.session_state.history = load_chat(u)
                st.success(f"Welcome {u}!")
                st.rerun()
            else:
                st.error("Invalid username or password.")

if not st.session_state.auth_ok:
    auth_ui()
    st.stop()

USER = st.session_state.user_id

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Session")
    colA, colB = st.columns(2)
    with colA:
        if st.button("Clear chat"):
            clear_chat(USER)
            st.session_state.history = []
            st.session_state.last_answer = None
            st.rerun()
    with colB:
        if st.button("Sign out"):
            st.session_state.auth_ok = False
            st.session_state.user_id = ""
            st.rerun()

    st.markdown("---")
    st.header("Status")
    st.markdown(
        f"""
        <span class="badge {'ok' if VECTOR_STORE_ID else 'warn'}">Vector store: {'connected' if VECTOR_STORE_ID else 'not connected'}</span>
        <span class="badge {'ok' if LLM_KEY_AVAILABLE else 'err'}">LLM API key: {'available' if LLM_KEY_AVAILABLE else 'missing'}</span>
        """,
        unsafe_allow_html=True
    )

# ---------- Main ----------
st.markdown(f"## {APP_NAME}")
query = st.text_input(
    "Ask a question",
    placeholder="e.g., According to the CBB Rulebook, what are disclosure requirements for large exposures (compare with Basel)?",
)

def run_query(q: str):
    if not q.strip():
        return

    # save user turn (memory)
    append_turn(USER, "user", q)
    st.session_state.history.append({"role": "user", "content": q})
    save_chat(USER, st.session_state.history)

    with st.spinner("Thinking…"):
        ans: RegulAIteAnswer = ask(
            query=q,
            user_id=USER,
            history=st.session_state.history,
            k_hint=12,                # high recall
            evidence_mode=True,       # include evidence/quotes when applicable
            mode_hint="long",         # long, detailed narrative
            web_enabled=True,         # always web + vector
            vec_id=VECTOR_STORE_ID or None,
            model=DEFAULT_MODEL,
        )

    # Normalize to clean markdown always
    md = _coerce_answer_to_markdown(ans)
    append_turn(USER, "assistant", md)
    st.session_state.history.append({"role": "assistant", "content": md})
    st.session_state.last_answer = ans
    save_chat(USER, st.session_state.history)

    render_message("assistant", md)

# Ask button
cols = st.columns([1, 6])
with cols[0]:
    if st.button("Ask", type="primary", use_container_width=True) and query:
        run_query(query)

# Render history so far
for turn in st.session_state.history:
    render_message(turn["role"], turn["content"])

# Follow-up chips (always show something, with unique keys)
def render_followups():
    suggs: List[str] = []
    if isinstance(st.session_state.last_answer, RegulAIteAnswer):
        suggs = getattr(st.session_state.last_answer, "follow_up_suggestions", None) or []
    if not suggs:
        topic = (query or "this topic").strip()
        suggs = [
            f"What are approval thresholds and board oversight for {topic}?",
            f"Draft a closure checklist for {topic} with controls and required evidence.",
            f"What reporting pack fields should be in the monthly board pack for {topic}?",
            f"How should breaches/exceptions for {topic} be escalated and documented?",
            f"What stress-test scenarios are relevant for {topic} and how to calibrate them?",
            f"What are the key risks, controls, and KRIs for {topic} (with metrics)?",
        ]
    st.caption("Try a follow-up:")
    chip_cols = st.columns(3)
    for i, s in enumerate(suggs[:6]):
        with chip_cols[i % 3]:
            if st.button(s, key=f"chip_{len(st.session_state.history)}_{i}", use_container_width=True):
                st.session_state["__chip_query"] = s
                st.rerun()

render_followups()

# Handle follow-up chip
chip_q = st.session_state.pop("__chip_query", None)
if chip_q:
    run_query(chip_q)
