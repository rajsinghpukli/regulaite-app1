import os
import streamlit as st

from rag.pipeline import ask, PERSIST_DIR


# ------------------------------
# AUTH / LOGIN
# ------------------------------
def _login_required() -> bool:
    """
    Returns True if the app should enforce login.
    Toggle with LOGIN_REQUIRED env var (default: True).
    """
    val = os.getenv("LOGIN_REQUIRED", "true").strip().lower()
    return val in ("1", "true", "yes", "on")


def _check_login_gate() -> None:
    """
    A very simple password gate:
    - Set env APP_PASSWORD (or STREAMLIT_PASSWORD) to require login.
    - Use LOGIN_REQUIRED=false to disable this gate.
    """
    if not _login_required():
        st.session_state["authed"] = True
        return

    required_password = os.getenv("APP_PASSWORD") or os.getenv("STREAMLIT_PASSWORD")
    if not required_password:
        # If no password is set, let the user in with a warning
        st.session_state["authed"] = True
        st.sidebar.info("APP_PASSWORD not set — login gate is disabled.")
        return

    if st.session_state.get("authed"):
        # already logged in
        return

    st.markdown(
        "<h2 style='text-align:center;margin-top:3rem;'>RegulaiTE — Login</h2>",
        unsafe_allow_html=True,
    )
    with st.form("login"):
        st.write("Please enter your password to continue.")
        pwd = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            if pwd == required_password:
                st.session_state["authed"] = True
                st.experimental_rerun()
            else:
                st.error("Incorrect password.")


# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="RegulaiTE — RAG Assistant",
    page_icon="🤖",
    layout="wide",
)

# GATE
_check_login_gate()
if not st.session_state.get("authed"):
    st.stop()


# ------------------------------
# HEADER
# ------------------------------
st.markdown(
    """
    <h1 style="margin-top:0.2rem;margin-bottom:0.4rem;">RegulaiTE — RAG Assistant</h1>
    """,
    unsafe_allow_html=True,
)


# ------------------------------
# SIDEBAR SETTINGS
# ------------------------------
with st.sidebar:
    st.header("Settings")

    top_k = st.slider("Top-K hint", min_value=1, max_value=10, value=5)
    evidence_mode = st.toggle("Evidence Mode (2–5 quotes/framework)", value=False)

    st.divider()
    st.subheader("System status")

    vs_id = os.getenv("OPENAI_VECTOR_STORE", "").strip()
    if vs_id:
        st.success(f"Vector store: connected\n\n**id:** `{vs_id}`")
    else:
        st.error("Vector store: not set")
        st.caption(f"Local fallback label: `{PERSIST_DIR.as_posix()}`")

    if os.getenv("OPENAI_API_KEY"):
        st.success("LLM API key: available")
    else:
        st.error("LLM API key: missing (set OPENAI_API_KEY)")

    if _login_required():
        if st.button("Logout"):
            st.session_state.clear()
            st.experimental_rerun()


# ------------------------------
# MAIN INPUT
# ------------------------------
query = st.text_input(
    "Ask a question",
    placeholder="“Main features of capital instruments” — identify the disclosure template; give 2–3 short quotes with file name and page.",
)

col1, col2 = st.columns([1, 3])
with col1:
    include_web = st.toggle("Web search (beta)", value=False)
with col2:
    mode_hint = st.selectbox(
        "Mode hint (optional)",
        ["auto", "policy", "accounting", "finance", "other"],
        index=0,
    )

# Call
if st.button("Ask", type="primary"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking…"):
            # Call pipeline.ask; now supports k and evidence_mode
            result = ask(
                query.strip(),
                include_web=include_web,
                mode_hint=(None if mode_hint == "auto" else mode_hint),
                k=top_k,
                evidence_mode=evidence_mode,
            )

        # Normalize output
        st.markdown("### Answer")
        if isinstance(result, str):
            st.write(result)
        else:
            st.write(str(result))
