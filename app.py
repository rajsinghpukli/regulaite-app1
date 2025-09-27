from __future__ import annotations
import os
import streamlit as st
from dotenv import load_dotenv
from rag.pipeline import ask
from rag.persist import load_chat, save_chat, append_turn, clear_chat
from rag.persist.users import ensure_bootstrap_admin, verify_user, ALLOW_SIGNUP
from rag.schema import RegulAIteAnswer

load_dotenv()

APP_NAME = "RegulAIte — Regulatory Assistant (Pilot)"
VECTOR_STORE_ID = os.getenv("OPENAI_VECTOR_STORE_ID", "").strip()
LLM_KEY_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))
DEFAULT_MODEL = os.getenv("RESPONSES_MODEL", "gpt-4.1-mini")
LOGO_URL = os.getenv("LOGO_URL", "")  # optional public URL
LOCAL_LOGO_PATH = os.getenv("LOCAL_LOGO_PATH", "assets/khaleeji_logo.png")  # optional repo asset

ensure_bootstrap_admin()

st.set_page_config(page_title=APP_NAME, page_icon="📘", layout="wide")

# ---------- Auth ----------
if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False
if "user_id" not in st.session_state:
    st.session_state.user_id = ""

def auth_ui():
    st.markdown("### Sign in")
    with st.form("login_form"):
        u = st.text_input("Username")
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
    # Sign-up deliberately hidden for the pilot (ALLOW_SIGNUP kept in code for future)

if not st.session_state.auth_ok:
    # lightweight header even on login
    cols = st.columns([1, 8])
    with cols[0]:
        try:
            if LOGO_URL:
                st.image(LOGO_URL, use_container_width=True)
            elif os.path.exists(LOCAL_LOGO_PATH):
                st.image(LOCAL_LOGO_PATH, use_container_width=True)
        except Exception:
            pass
    with cols[1]:
        st.title(APP_NAME)
        st.caption("Private preview for Khaleeji Bank stakeholders")
    auth_ui()
    st.stop()

USER = st.session_state.user_id

# ---------- State ----------
if "history" not in st.session_state:
    st.session_state.history = load_chat(USER)
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None  # RegulAIteAnswer

# ---------- Sidebar (minimal, no knobs) ----------
with st.sidebar:
    try:
        if LOGO_URL:
            st.image(LOGO_URL, use_container_width=True)
        elif os.path.exists(LOCAL_LOGO_PATH):
            st.image(LOCAL_LOGO_PATH, use_container_width=True)
    except Exception:
        pass

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
    st.caption("Status")
    if VECTOR_STORE_ID:
        st.success("Vector store: connected")
    else:
        st.info("Vector store: not connected")
    st.success("LLM API key: available" if LLM_KEY_AVAILABLE else "LLM API key: missing")

# ---------- Main ----------
cols = st.columns([1, 8])
with cols[0]:
    try:
        if LOGO_URL:
            st.image(LOGO_URL, use_container_width=True)
        elif os.path.exists(LOCAL_LOGO_PATH):
            st.image(LOCAL_LOGO_PATH, use_container_width=True)
    except Exception:
        pass
with cols[1]:
    st.title(APP_NAME)
    st.caption("Long-form, ChatGPT-style answers with vector + web retrieval (always on).")

query = st.text_input("Ask a question", placeholder="e.g., Approval thresholds and governance for connected counterparties (IFRS/AAOIFI/CBB)…")
go = st.button("Ask", type="primary", use_container_width=True)

# Render history
for turn in st.session_state.history:
    if turn["role"] == "user":
        st.chat_message("user").write(turn["content"])
    else:
        st.chat_message("assistant").write(turn["content"])

def render_followups(default_topic: str | None = None):
    suggs: list[str] = []
    if isinstance(st.session_state.last_answer, RegulAIteAnswer):
        suggs = st.session_state.last_answer.follow_up_suggestions or []
    if not suggs:
        topic = (default_topic or query or "connected counterparties").strip()
        suggs = [
            f"Draft a closure checklist for {topic} with required approvals and documentation.",
            f"What reporting pack fields should go to the board each month for {topic}?",
            f"How should breaches/exceptions for {topic} be escalated and documented?",
            f"What stress-test scenarios are relevant for {topic}, and how to calibrate them?",
            f"Summarize key risks, controls, and KRIs for {topic} (with metrics).",
            f"List typical audit findings and how to avoid them for {topic}.",
        ]
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

    with st.spinner("Generating a comprehensive answer…"):
        ans = ask(
            q,
            user_id=USER,
            history=st.session_state.history,
            # Always-on long/research mode and web (no UI knobs)
            k_hint=12,                 # max Top-K hint internally
            evidence_mode=True,
            mode_hint="research",
            web_enabled=True,
            vec_id=VECTOR_STORE_ID or None,
            model=DEFAULT_MODEL,
        )

    if not ans.follow_up_suggestions:
        topic = (q or "the topic").strip()
        ans.follow_up_suggestions = [
            f"Draft a closure checklist for {topic} with required approvals and documentation.",
            f"What reporting pack fields should go to the board each month for {topic}?",
            f"How should breaches/exceptions for {topic} be escalated and documented?",
            f"What stress-test scenarios are relevant for {topic}, and how to calibrate them?",
            f"Summarize key risks, controls, and KRIs for {topic} (with metrics).",
            f"List typical audit findings and how to avoid them for {topic}.",
        ]

    md = ans.as_markdown().strip() or "_No answer produced._"

    append_turn(USER, "assistant", md)
    st.session_state.history.append({"role": "assistant", "content": md})
    st.session_state.last_answer = ans
    save_chat(USER, st.session_state.history)

    st.chat_message("assistant").write(md)

if go and query:
    run_query(query)

chip_q = st.session_state.pop("__chip_query", None)
if chip_q:
    run_query(chip_q)

with st.expander("Notes", expanded=False):
    st.markdown(
        """
- Pilot mode: **Sign-up hidden**, 3 predefined accounts via `AUTH_USERS` env (`user:pass` pairs).
- Retrieval: **Vector primary** + **web always on** (no toggle).
- Answers: long-form narrative with tables/checklists when helpful; frameworks with no evidence are silently omitted.
- Memory: per-user chat stored under `rag/persist/chats/<user>.json`.
        """
    )
