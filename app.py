from __future__ import annotations
import os
import streamlit as st
from dotenv import load_dotenv
from rag.pipeline import ask
from rag.persist import load_chat, save_chat, append_turn, clear_chat
from rag.persist.users import (
    ensure_bootstrap_admin,
    verify_user,
)
from rag.schema import RegulAIteAnswer

load_dotenv()

APP_NAME = "RegulaiTE — Regulatory Assistant (Pilot)"
DEFAULT_MODEL = os.getenv("RESPONSES_MODEL", "gpt-4.1-mini")
VECTOR_STORE_ID = os.getenv("OPENAI_VECTOR_STORE_ID", "").strip()
LLM_KEY_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))

# Seed an admin if env vars provided
ensure_bootstrap_admin()

st.set_page_config(page_title=APP_NAME, page_icon="🧭", layout="wide")

# -------- Auth state --------
if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False
if "user_id" not in st.session_state:
    st.session_state.user_id = ""

def auth_ui():
    st.markdown("## Sign in")
    with st.form("login_form"):
        u = st.text_input("Username", value=st.session_state.user_id or "")
        p = st.text_input("Password", type="password")
        ok = st.form_submit_button("Sign in")
        if ok:
            if verify_user(u, p):
                st.session_state.auth_ok = True
                st.session_state.user_id = u
                st.success("Signed in")
                st.rerun()
            else:
                st.error("Invalid credentials")

if not st.session_state.auth_ok:
    auth_ui()
    st.stop()

USER = st.session_state.user_id

# -------- State --------
if "history" not in st.session_state:
    st.session_state.history = load_chat(USER)
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None  # RegulAIteAnswer

# -------- Sidebar --------
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
    if VECTOR_STORE_ID:
        st.success("Vector store: connected")
    else:
        st.warning("Vector store: not connected")
    st.success("LLM API key: available" if LLM_KEY_AVAILABLE else "LLM API key: missing")

# -------- Main --------
st.markdown(f"## {APP_NAME}")
query = st.text_input("Ask a question", placeholder="e.g., Connected counterparties completion/closure controls…")

cols = st.columns([1, 6])
with cols[0]:
    ask_clicked = st.button("Ask", type="primary", use_container_width=True)

# Render history
for turn in st.session_state.history:
    if turn["role"] == "user":
        st.chat_message("user").write(turn["content"])
    else:
        st.chat_message("assistant").write(turn["content"])

# Follow-up chips
def render_followups():
    suggs: list[str] = []
    if isinstance(st.session_state.last_answer, RegulAIteAnswer):
        suggs = st.session_state.last_answer.follow_up_suggestions or []
    if not suggs:
        return
    st.caption("Try a follow-up:")
    chip_cols = st.columns(3)
    for i, s in enumerate(suggs[:6]):
        with chip_cols[i % 3]:
            if st.button(s, key=f"chip_{i}", use_container_width=True):
                st.session_state["__chip_query"] = s

render_followups()

def run_query(q: str):
    if not q.strip():
        return
    append_turn(USER, "user", q)
    st.session_state.history.append({"role": "user", "content": q})
    save_chat(USER, st.session_state.history)

    with st.spinner("Thinking…"):
        ans = ask(
            q,
            user_id=USER,
            history=st.session_state.history,
            model=DEFAULT_MODEL,
        )

    md = ans.as_markdown().strip() or "_No answer produced._"

    append_turn(USER, "assistant", md)
    st.session_state.history.append({"role": "assistant", "content": md})
    st.session_state.last_answer = ans
    save_chat(USER, st.session_state.history)

    st.chat_message("assistant").write(md)
    render_followups()

# Handle Ask + chip queries
if ask_clicked and query:
    run_query(query)

chip_q = st.session_state.pop("__chip_query", None)
if chip_q:
    run_query(chip_q)
