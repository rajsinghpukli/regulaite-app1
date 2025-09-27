from __future__ import annotations
import os
import streamlit as st
from dotenv import load_dotenv
from rag.pipeline import ask
from rag.schema import RegulAIteAnswer

load_dotenv()

APP_NAME = "RegulAIte — Regulatory Assistant (Pilot)"
DEFAULT_MODEL = os.getenv("RESPONSES_MODEL", "gpt-4.1-mini")
VECTOR_STORE_ID = os.getenv("OPENAI_VECTOR_STORE_ID", "").strip()
LLM_KEY_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))

# --- Simple pre-assigned logins ---
PRESET_USERS = {
    "guest1": "pass1",
    "guest2": "pass2",
    "guest3": "pass3",
}

st.set_page_config(page_title=APP_NAME, page_icon="🧭", layout="wide")

# -------- Session state --------
if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False
if "user_id" not in st.session_state:
    st.session_state.user_id = ""
if "history" not in st.session_state:
    st.session_state.history: list[dict] = []
if "last_answer" not in st.session_state:
    st.session_state.last_answer: RegulAIteAnswer | None = None

# -------- Login UI --------
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
                st.success(f"Welcome {u}!")
                st.rerun()
            else:
                st.error("Invalid username or password.")

if not st.session_state.auth_ok:
    auth_ui()
    st.stop()

USER = st.session_state.user_id

# -------- Sidebar --------
with st.sidebar:
    st.header("Session")
    colA, colB = st.columns(2)
    with colA:
        if st.button("Clear chat"):
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

# -------- Main --------
st.markdown(f"## {APP_NAME}")

query = st.text_input(
    "Ask a question",
    placeholder="e.g., According to the CBB Rulebook, what are disclosure requirements for large exposures (compare with Basel)?",
)

if st.button("Ask", type="primary", use_container_width=True) and query.strip():
    st.session_state.history.append({"role": "user", "content": query})
    with st.spinner("Thinking…"):
        ans: RegulAIteAnswer = ask(
            query=query,
            user_id=USER,
            history=st.session_state.history,
            # hard-tuned for long, high-quality answers:
            k_hint=12,
            evidence_mode=True,
            mode_hint="long",
            web_enabled=True,
            vec_id=VECTOR_STORE_ID or None,
            model=DEFAULT_MODEL,
        )
    st.session_state.last_answer = ans
    md = ans.as_markdown().strip() or "_No answer produced._"
    st.session_state.history.append({"role": "assistant", "content": md})
    st.rerun()

# Render history
for turn in st.session_state.history:
    if turn["role"] == "user":
        st.chat_message("user").write(turn["content"])
    else:
        st.chat_message("assistant").write(turn["content"])

# Follow-ups (always)
def render_followups():
    suggs: list[str] = []
    if isinstance(st.session_state.last_answer, RegulAIteAnswer):
        suggs = st.session_state.last_answer.follow_up_suggestions or []
    if not suggs:
        topic = (query or "the topic").strip()
        suggs = [
            f"What are approval thresholds and board oversight for {topic}?",
            f"Draft a closure checklist for {topic} with controls and required evidence.",
            f"What reporting pack fields should be in the monthly board pack for {topic}?",
            f"How should breaches/exceptions for {topic} be escalated and documented?",
            f"What stress-test scenarios are relevant for {topic} and how to calibrate them?",
            f"What are the key risks, controls, and KRIs for {topic} (with metrics)?",
        ]
    if suggs:
        st.caption("Try a follow-up:")
        chip_cols = st.columns(3)
        for i, s in enumerate(suggs[:6]):
            with chip_cols[i % 3]:
                if st.button(
                    s,
                    key=f"chip_{len(st.session_state.history)}_{i}",  # unique per render
                    use_container_width=True,
                ):
                    st.session_state["__chip_query"] = s
                    st.rerun()

render_followups()

# Handle follow-up chip
chip_q = st.session_state.pop("__chip_query", None)
if chip_q:
    st.session_state.history.append({"role": "user", "content": chip_q})
    with st.spinner("Thinking…"):
        ans: RegulAIteAnswer = ask(
            query=chip_q,
            user_id=USER,
            history=st.session_state.history,
            k_hint=12,
            evidence_mode=True,
            mode_hint="long",
            web_enabled=True,
            vec_id=VECTOR_STORE_ID or None,
            model=DEFAULT_MODEL,
        )
    st.session_state.last_answer = ans
    md = ans.as_markdown().strip() or "_No answer produced._"
    st.session_state.history.append({"role": "assistant", "content": md})
    st.rerun()
