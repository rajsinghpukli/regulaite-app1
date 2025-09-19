from __future__ import annotations
import os
import streamlit as st
from dotenv import load_dotenv
from rag.pipeline import ask
from rag.persist import load_chat, save_chat, append_turn, clear_chat
from rag.persist.users import (
    ensure_bootstrap_admin,
    verify_user,
    create_user_if_allowed,
    username_exists,
    ALLOW_SIGNUP,
)
from rag.schema import RegulAIteAnswer

load_dotenv()

APP_NAME = "RegulaiTE — RAG Assistant"
DEFAULT_MODEL = os.getenv("RESPONSES_MODEL", "gpt-4.1-mini")
VECTOR_STORE_ID = os.getenv("OPENAI_VECTOR_STORE_ID", "").strip()
LLM_KEY_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))

# Optional: seed an admin from env on first run
ensure_bootstrap_admin()

st.set_page_config(page_title=APP_NAME, page_icon="🧭", layout="wide")

# -------- Auth state --------
if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False
if "user_id" not in st.session_state:
    st.session_state.user_id = ""

def auth_ui():
    st.markdown("## Sign in")

    tabs = st.tabs(["Sign in", "Create account" if ALLOW_SIGNUP else "Create account (disabled)"])

    with tabs[0]:
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

    with tabs[1]:
        st.caption("Create a new account stored securely on this app (hashed passwords).")
        disabled = not ALLOW_SIGNUP
        with st.form("signup_form", clear_on_submit=False):
            nu = st.text_input("New username", disabled=disabled)
            npw = st.text_input("New password", type="password", disabled=disabled)
            npw2 = st.text_input("Confirm password", type="password", disabled=disabled)
            create = st.form_submit_button("Create account", disabled=disabled)
            if create and not disabled:
                if npw != npw2:
                    st.error("Passwords do not match.")
                elif username_exists(nu):
                    st.error("Username already exists.")
                else:
                    ok, msg = create_user_if_allowed(nu, npw)
                    if ok:
                        st.success(msg + " You can now sign in.")
                    else:
                        st.error(msg)

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
    st.header("Settings")
    k_hint = st.slider("Top-K hint", min_value=3, max_value=12, value=5, step=1)
    evidence_mode = st.toggle("Evidence Mode (2–5 quotes/framework)", value=True)
    web_enabled = st.toggle("Web search (beta)", value=False)
    mode_hint = st.selectbox("Mode hint (optional)", ["auto", "short", "long", "research"], index=0)

    st.markdown("---")
    st.header("System status")

    if VECTOR_STORE_ID:
        st.success("Vector store: connected")
        # Mask the id so it isn't exposed publicly
        masked = f"{VECTOR_STORE_ID[:6]}…{VECTOR_STORE_ID[-4:]}" if len(VECTOR_STORE_ID) > 10 else "configured"
        st.caption("id:")
        st.code(masked, language="text")
    else:
        st.info("Vector store: not connected")
        st.caption("id:")
        st.code("(none)", language="text")

    st.success("LLM API key: available" if LLM_KEY_AVAILABLE else "LLM API key: missing")

    st.markdown("---")
    colA, colB = st.columns(2)
    with colA:
        if st.button("Clear conversation"):
            clear_chat(USER)
            st.session_state.history = []
            st.session_state.last_answer = None
            st.rerun()
    with colB:
        if st.button("Sign out"):
            st.session_state.auth_ok = False
            st.session_state.user_id = ""
            st.rerun()

# -------- Main --------
st.markdown(f"## {APP_NAME}")

query = st.text_input("Ask a question", placeholder="e.g., Connected counterparties completion/closure controls…")
cols = st.columns([1, 1, 6])
with cols[0]:
    ask_clicked = st.button("Ask", type="primary", use_container_width=True)
with cols[1]:
    paste_example = st.button("Example", use_container_width=True)
if paste_example:
    query = (
        "Provide IFRS 9, AAOIFI (FAS 30/33), and CBB Rulebook guidance for "
        "completion/closure of exposures to connected counterparties: definitions, approval thresholds, "
        "credit limits/large exposure constraints, reporting/disclosure, and governance controls. "
        "Return 2–5 short evidence quotes per framework with citations, compare differences, and give "
        "a concise recommendation for Khaleeji Bank."
    )

# Render history
for turn in st.session_state.history:
    if turn["role"] == "user":
        st.chat_message("user").write(turn["content"])
    else:
        st.chat_message("assistant").write(turn["content"])

# Follow-up chips (from last answer if present)
def render_followups(default_topic: str | None = None):
    suggs: list[str] = []
    if isinstance(st.session_state.last_answer, RegulAIteAnswer):
        suggs = st.session_state.last_answer.follow_up_suggestions or []
    # Deterministic fallback if model didn't supply any
    if not suggs:
        topic = (default_topic or query or "connected counterparties").strip()
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
            if st.button(s, key=f"chip_{i}", use_container_width=True):
                # Clicking a chip fills the input and fires a query
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
            k_hint=k_hint,
            evidence_mode=evidence_mode,
            mode_hint=mode_hint,
            web_enabled=web_enabled,
            vec_id=VECTOR_STORE_ID or None,
            model=DEFAULT_MODEL,
        )

    # Ensure we always have some follow-ups
    if not ans.follow_up_suggestions:
        topic = (q or "the topic").strip()
        ans.follow_up_suggestions = [
            f"What are approval thresholds and board oversight for {topic}?",
            f"Draft a closure checklist for {topic} with controls and required evidence.",
            f"What reporting pack fields should be in the monthly board pack for {topic}?",
            f"How should breaches/exceptions for {topic} be escalated and documented?",
            f"What stress-test scenarios are relevant for {topic} and how to calibrate them?",
            f"What are the key risks, controls, and KRIs for {topic} (with metrics)?",
        ]

    md = ans.as_markdown().strip() or "_No answer produced._"

    append_turn(USER, "assistant", md)
    st.session_state.history.append({"role": "assistant", "content": md})
    st.session_state.last_answer = ans
    save_chat(USER, st.session_state.history)

    st.chat_message("assistant").write(md)

# Handle button + chip-triggered queries
if ask_clicked and query:
    run_query(query)

chip_q = st.session_state.pop("__chip_query", None)
if chip_q:
    run_query(chip_q)

with st.expander("Notes / Help", expanded=False):
    st.markdown(
        """
- **Create account:** enabled by `ALLOW_SIGNUP=true` (default). Passwords are salted+hashed and saved in `rag/persist/users.json`.
- **Bootstrap admin:** set `BASIC_USER` and `BASIC_PASS` in App Service → Configuration. On first run, that account is created automatically.
- **Vector store:** set `OPENAI_VECTOR_STORE_ID` to `vs_...`. The id is masked in this UI.
- **Models:** primary `RESPONSES_MODEL` (default: `gpt-4.1-mini`); fallback chat `CHAT_MODEL` (default: `gpt-4o-mini`).
- **Memory:** chat history per user at `rag/persist/chats/<user>.json`.
        """
    )
