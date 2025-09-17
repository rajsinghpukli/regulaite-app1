from __future__ import annotations
import os
import streamlit as st
from dotenv import load_dotenv
from rag.pipeline import ask
from rag.persist import load_chat, save_chat, append_turn, clear_chat
from rag.schema import RegulAIteAnswer

load_dotenv()

APP_NAME = "RegulaiTE — RAG Assistant"
DEFAULT_MODEL = os.getenv("RESPONSES_MODEL", "gpt-4.1-mini")
VECTOR_STORE_ID = os.getenv("OPENAI_VECTOR_STORE_ID", "").strip()
BASIC_USER = os.getenv("BASIC_USER", "raj")
BASIC_PASS = os.getenv("BASIC_PASS", "pass")
LLM_KEY_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title=APP_NAME, page_icon="🧭", layout="wide")

# -------- Auth --------
if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False
if "user_id" not in st.session_state:
    st.session_state.user_id = ""

def login_ui():
    with st.form("login"):
        st.markdown("### Sign in")
        u = st.text_input("Username", value=st.session_state.user_id or "")
        p = st.text_input("Password", type="password")
        ok = st.form_submit_button("Sign in")
        if ok:
            if (u == BASIC_USER) and (p == BASIC_PASS):
                st.session_state.auth_ok = True
                st.session_state.user_id = u
                st.success("Signed in")
                st.rerun()
            else:
                st.error("Invalid credentials")

if not st.session_state.auth_ok:
    login_ui()
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
    st.success("Vector store: connected" if VECTOR_STORE_ID else "Vector store: not connected")
    st.caption("id:")
    st.code(VECTOR_STORE_ID or "(none)", language="text")
    st.success("LLM API key: available" if LLM_KEY_AVAILABLE else "LLM API key: missing")

    st.markdown("---")
    if st.button("Clear conversation"):
        clear_chat(USER)
        st.session_state.history = []
        st.session_state.last_answer = None
        st.rerun()

# -------- Main --------
st.markdown(f"## {APP_NAME}")

query = st.text_input("Ask a question", placeholder="e.g., Connected counterparties completion/closure controls…")
cols = st.columns([1,1,6])
with cols[0]:
    ask_clicked = st.button("Ask", type="primary", use_container_width=True)
with cols[1]:
    paste_example = st.button("Example", use_container_width=True)
if paste_example:
    query = "Provide IFRS vs AAOIFI vs CBB guidance on connected counterparty exposures: risk limits, approvals, reporting, and completion/closure controls. Include evidence."

for turn in st.session_state.history:
    if turn["role"] == "user":
        st.chat_message("user").write(turn["content"])
    else:
        st.chat_message("assistant").write(turn["content"])

if isinstance(st.session_state.last_answer, RegulAIteAnswer):
    suggs = st.session_state.last_answer.follow_up_suggestions or []
    if suggs:
        st.caption("Try a follow-up:")
        chip_cols = st.columns(3)
        for i, s in enumerate(suggs[:6]):
            with chip_cols[i % 3]:
                if st.button(s, key=f"chip_{i}", use_container_width=True):
                    query = s
                    ask_clicked = True

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
    md = ans.as_markdown().strip() or "_No answer produced._"

    append_turn(USER, "assistant", md)
    st.session_state.history.append({"role": "assistant", "content": md})
    st.session_state.last_answer = ans
    save_chat(USER, st.session_state.history)

    st.chat_message("assistant").write(md)

if ask_clicked:
    run_query(query)

with st.expander("Notes / Help", expanded=False):
    st.markdown(
        """
- Set `BASIC_USER` / `BASIC_PASS` in App Service → Configuration.
- Set `OPENAI_VECTOR_STORE_ID` to your `vs_...` id.
- Change `RESPONSES_MODEL` if needed (default `gpt-4.1-mini`).
- Chat memory is saved per user in `rag/persist/chats/<user>.json`.
        """
    )
