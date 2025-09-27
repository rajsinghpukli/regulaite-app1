import streamlit as st
from rag.pipeline import ask
from rag.schema import RegulAIteAnswer

# ---------- Preassigned logins ----------
USERS = {
    "guest1": "pass1",
    "guest2": "pass2",
    "guest3": "pass3"
}

# ---------- Login ----------
if "auth" not in st.session_state:
    st.session_state.auth = False
    st.session_state.user = None

if not st.session_state.auth:
    st.title("🔐 RegulAIte Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if u in USERS and USERS[u] == p:
            st.session_state.auth = True
            st.session_state.user = u
            st.success(f"Welcome {u}!")
            st.rerun()
        else:
            st.error("Invalid credentials")
    st.stop()

# ---------- Main UI ----------
st.title("RegulAIte — Regulatory Assistant (Pilot)")

query = st.text_input("Ask a question")
if st.button("Ask") and query:
    with st.spinner("Thinking..."):
        ans: RegulAIteAnswer = ask(
            query=query,
            user_id=st.session_state.user,
            history=st.session_state.get("history", []),
            k_hint=12,
            evidence_mode=True,
            mode_hint="long",
            web_enabled=True,
            vec_id=None,
            model="gpt-4o-mini"
        )

    # save history
    st.session_state.history = st.session_state.get("history", [])
    st.session_state.history.append({"role": "user", "content": query})
    st.session_state.history.append({"role": "assistant", "content": ans.summary})

    # render answer in sections
    if ans.summary:
        st.subheader("Summary")
        st.markdown(ans.summary)

    if ans.comparative_analysis:
        st.subheader("Comparative Analysis")
        st.markdown(ans.comparative_analysis)

    if ans.recommendation:
        st.subheader("Recommendation")
        st.markdown(ans.recommendation)

    if ans.general_knowledge:
        st.subheader("General Knowledge")
        st.markdown(ans.general_knowledge)

    if ans.gaps_or_next_steps:
        st.subheader("Gaps / Next Steps")
        st.markdown(ans.gaps_or_next_steps)

    if ans.citations:
        st.subheader("Citations")
        for c in ans.citations:
            st.markdown(f"- {c}")

    if ans.follow_up_suggestions:
        st.subheader("Try a follow-up:")
        for i, f in enumerate(ans.follow_up_suggestions):
            if st.button(f, key=f"chip_{i}", use_container_width=True):
                st.session_state.query = f
                st.rerun()
