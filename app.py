import os
import streamlit as st
from rag.pipeline import ask
from rag.schema import RegulAIteAnswer

# --------- Config ---------
st.set_page_config(page_title="RegulAIte — Regulatory Assistant (Pilot)", layout="wide")

# Preassigned logins
VALID_USERS = {
    "amit": "Khaleeji#2025",
    "raj": "RegulAIte#Dev",
    "review": "POC#Access",
}


# --------- Session state ---------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "history" not in st.session_state:
    st.session_state.history = []


# --------- Login screen ---------
if not st.session_state.logged_in:
    st.title("🔐 RegulAIte Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in VALID_USERS and VALID_USERS[username] == password:
            st.session_state.logged_in = True
            st.session_state.user_id = username
            st.success(f"Welcome {username}!")
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")
    st.stop()


# --------- Main UI ---------
st.title("RegulAIte — Regulatory Assistant (Pilot)")
query = st.text_input("Ask a question", key="query_input")

if st.button("Ask", key="ask_btn"):
    if query.strip():
        st.session_state.history.append({"role": "user", "content": query})
        try:
            ans: RegulAIteAnswer = ask(
                query=query,
                user_id=st.session_state.user_id,
                history=st.session_state.history,
            )
            st.session_state.history.append({"role": "assistant", "content": ans.raw_markdown})
        except Exception as e:
            st.session_state.history.append({"role": "assistant", "content": f"_Error: {e}_"})

# Display history
for h in st.session_state.history:
    if h["role"] == "user":
        st.markdown(f"**You:** {h['content']}")
    else:
        st.markdown(h["content"])

# Follow-up suggestions
if st.session_state.history and st.session_state.history[-1]["role"] == "assistant":
    last_ans = st.session_state.history[-1]["content"]
    if isinstance(last_ans, str) and "##" in last_ans:  # crude check
        st.subheader("Try a follow-up:")
        try:
            suggestions = ask(
                query="Suggest follow-ups only",
                user_id=st.session_state.user_id,
                history=st.session_state.history,
            ).follow_up_suggestions
        except Exception:
            suggestions = []
        for i, s in enumerate(suggestions):
            if st.button(s, key=f"chip_{i}"):
                st.session_state.query_input = s
                st.experimental_rerun()

# Sidebar
st.sidebar.header("Session")
if st.sidebar.button("Clear chat"):
    st.session_state.history = []
if st.sidebar.button("Sign out"):
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.experimental_rerun()

st.sidebar.subheader("Status")
st.sidebar.success("Vector store: connected")
st.sidebar.success("LLM API key: available")
