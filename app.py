from __future__ import annotations
import os
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

# ---------- Session state ----------
if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False
if "user_id" not in st.session_state:
    st.session_state.user_id = ""
if "history" not in st.session_state:
    st.session_state.history: list[dict] = []
if "last_answer" not in st.session_state:
    st.session_state.last_answer: RegulAIteAnswer | None = None

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
    st.success("Vector store: connected" if VECTOR_STORE_ID else "Vector store: not connected")
    st.success("LLM API key: available" if LLM_KEY_AVAILABLE else "LLM API key: missing")

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

    # always render as ChatGPT-style Markdown
    md = (ans.as_markdown() or "").strip()
    if not md:
        md = "_No answer produced._"

    append_turn(USER, "assistant", md)
    st.session_state.history.append({"role": "assistant", "content": md})
    st.session_state.last_answer = ans
    save_chat(USER, st.session_state.history)

    # show the message immediately
    st.chat_message("assistant").write(md)

# Ask button
cols = st.columns([1, 6])
with cols[0]:
    if st.button("Ask", type="primary", use_container_width=True) and query:
        run_query(query)

# Render history so far
for turn in st.session_state.history:
    if turn["role"] == "user":
        st.chat_message("user").write(turn["content"])
    else:
        st.chat_message("assistant").write(turn["content"])

# Follow-up chips (always show something, with unique keys)
def render_followups():
    suggs: list[str] = []
    if isinstance(st.session_state.last_answer, RegulAIteAnswer):
        suggs = st.session_state.last_answer.follow_up_suggestions or []
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
