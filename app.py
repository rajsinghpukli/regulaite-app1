import os
import streamlit as st

from rag.pipeline import ask, PERSIST_DIR

# ---------- UI CONFIG ----------
st.set_page_config(
    page_title="RegulaiTE — RAG Assistant",
    page_icon="🤖",
    layout="wide",
)

st.title("RegulaiTE — RAG Assistant")

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
        st.caption(f"Fallback local label: `{PERSIST_DIR.as_posix()}`")

    api_key_present = bool(os.getenv("OPENAI_API_KEY"))
    if api_key_present:
        st.success("LLM API key: available")
    else:
        st.error("LLM API key: missing (set OPENAI_API_KEY)")

# ---------- MAIN INPUT ----------
query = st.text_input(
    "Ask a question",
    placeholder="How do we classify and measure sukuk held to maturity under IFRS 9 vs AAOIFI?",
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

if st.button("Ask", type="primary") and query.strip():
    with st.spinner("Thinking…"):
        try:
            answer = ask(
                query,
                include_web=include_web,
                mode_hint=None if mode_hint == "auto" else mode_hint,
                k=top_k,  # IMPORTANT: UI passes k (now supported)
                evidence_mode=evidence_mode,
            )
        except TypeError as e:
            # If you ever see this, your local deploy is out-of-sync. But we guard anyway.
            answer = f"Error: {e}"

    if isinstance(answer, str):
        st.write(answer)
    else:
        st.write(str(answer))
