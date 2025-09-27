# app.py
from __future__ import annotations
import os, re, json
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

PRESET_USERS = {"guest1": "pass1", "guest2": "pass2", "guest3": "pass3"}

st.set_page_config(page_title=APP_NAME, page_icon="🧭", layout="wide")

CSS = """
<style>
.block-container { max-width: 1180px; }
.badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;margin-right:6px;border:1px solid #0001}
.badge.ok{background:#ecfdf5;color:#065f46;border-color:#10b98133}
.badge.warn{background:#fff7ed;color:#9a3412;border-color:#f59e0b33}
.badge.err{background:#fef2f2;color:#991b1b;border-color:#ef444433}
.regu-msg{border-radius:14px;padding:14px 16px;box-shadow:0 1px 2px #0001;border:1px solid #0001;margin-bottom:10px}
.regu-user{background:#f5f7fb}
.regu-assistant{background:#fff}
.markdown-body { width: 100%; word-break: normal; overflow-wrap: break-word; hyphens: auto; }
.markdown-body h1,.markdown-body h2,.markdown-body h3{margin-top:1.2rem}
.markdown-body p,.markdown-body li{line-height:1.6}
.markdown-body table{width:100%;border-collapse:collapse}
.markdown-body th,.markdown-body td{border:1px solid #e5e7eb;padding:8px;font-size:14px}
.markdown-body th{background:#f9fafb}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---- session ----
if "auth_ok" not in st.session_state: st.session_state.auth_ok = False
if "user_id" not in st.session_state: st.session_state.user_id = ""
if "history" not in st.session_state: st.session_state.history: List[Dict[str,str]] = []
if "last_answer" not in st.session_state: st.session_state.last_answer: RegulAIteAnswer|None = None

# ---- helpers ----
def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _unescape(text: str) -> str:
    # Turn visible "\n" into real newlines only if needed
    return text.replace("\\n", "\n") if "\\n" in text and "\n" not in text else text

def _coerce_answer_to_markdown(ans: RegulAIteAnswer) -> str:
    try:
        md = ans.as_markdown() or ""
    except Exception:
        md = ""
    md = _strip_code_fences(md)
    md = _unescape(md)
    return md.strip() or "_No answer produced._"

def render_message(role: str, md: str):
    kind = "regu-user" if role == "user" else "regu-assistant"
    st.markdown(f'<div class="regu-msg {kind}"><div class="markdown-body">{md}</div></div>', unsafe_allow_html=True)

# ---- login ----
def auth_ui():
    st.markdown("## 🔐 RegulAIte Login")
    with st.form("login_form"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            if PRESET_USERS.get(u) == p:
                st.session_state.auth_ok = True
                st.session_state.user_id = u
                st.session_state.history = load_chat(u)
                st.success(f"Welcome {u}!")
                st.rerun()
            else:
                st.error("Invalid username or password.")

if not st.session_state.auth_ok:
    auth_ui(); st.stop()

USER = st.session_state.user_id

# ---- sidebar ----
with st.sidebar:
    st.header("Session")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Clear chat"): clear_chat(USER); st.session_state.history=[]; st.session_state.last_answer=None; st.rerun()
    with c2:
        if st.button("Sign out"): st.session_state.auth_ok=False; st.session_state.user_id=""; st.rerun()
    st.markdown("---")
    st.header("Status")
    st.markdown(
        f'<span class="badge {"ok" if VECTOR_STORE_ID else "warn"}">Vector store: {"connected" if VECTOR_STORE_ID else "not connected"}</span> '
        f'<span class="badge {"ok" if LLM_KEY_AVAILABLE else "err"}">LLM API key: {"available" if LLM_KEY_AVAILABLE else "missing"}</span>',
        unsafe_allow_html=True
    )

# ---- main ----
st.markdown(f"## {APP_NAME}")

query = st.text_input(
    "Ask a question",
    placeholder="e.g., According to the CBB Rulebook, what are disclosure requirements for large exposures (compare with Basel)?",
)

def run_query(q: str):
    if not q.strip(): return
    # save user turn (no immediate render → avoids duplicates)
    append_turn(USER, "user", q)
    st.session_state.history.append({"role": "user", "content": q})
    save_chat(USER, st.session_state.history)

    with st.spinner("Thinking…"):
        ans: RegulAIteAnswer = ask(
            query=q, user_id=USER, history=st.session_state.history,
            k_hint=12, evidence_mode=True, mode_hint="long",
            web_enabled=True, vec_id=VECTOR_STORE_ID or None, model=DEFAULT_MODEL,
        )

    md = _coerce_answer_to_markdown(ans)
    append_turn(USER, "assistant", md)
    st.session_state.history.append({"role": "assistant", "content": md})
    st.session_state.last_answer = ans
    save_chat(USER, st.session_state.history)
    # NOTE: do NOT render here; the history loop below will render once.

# Ask button
cols = st.columns([1, 6])
with cols[0]:
    if st.button("Ask", type="primary", use_container_width=True) and query:
        run_query(query)

# single-source rendering (no duplicates)
for turn in st.session_state.history:
    render_message(turn["role"], _unescape(_strip_code_fences(turn["content"])))

# Follow-up chips
def render_followups():
    suggs: List[str] = []
    if isinstance(st.session_state.last_answer, RegulAIteAnswer):
        suggs = (st.session_state.last_answer.follow_up_suggestions or [])
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
    cols = st.columns(3)
    for i, s in enumerate(suggs[:6]):
        with cols[i % 3]:
            if st.button(s, key=f"chip_{len(st.session_state.history)}_{i}", use_container_width=True):
                st.session_state["__chip_query"] = s
                st.rerun()

render_followups()

chip_q = st.session_state.pop("__chip_query", None)
if chip_q: run_query(chip_q)
