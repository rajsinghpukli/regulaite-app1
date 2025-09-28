from __future__ import annotations
import os, re, time, json
from typing import Any, Dict, List
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

PRESET_USERS = {"guest1": "pass1", "guest2": "pass2", "guest3": "pass3"}

st.set_page_config(page_title=APP_NAME, page_icon="🧭", layout="wide")

CSS = """
<style>
.block-container { max-width: 1180px; }
.badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;margin-right:6px;border:1px solid #0001}
.badge.ok{background:#ecfdf5;color:#065f46;border-color:#10b98133}
.badge.warn{background:#fff7ed;color:#9a3412;border-color:#f59e0b33}
.badge.err{background:#fef2f2;color:#991b1b;border-color:#ef444433}
.regu-msg{border-radius:14px;padding:14px 16px;box-shadow:0 1px 2px #0001;border:1px solid #0001;margin-bottom:10px}
.regu-user{background:#f5f7fb}
.regu-assistant{background:#fff}
.meta { font-size:12px;color:#6b7280;margin-top:6px }
.markdown-body { width:100%; word-break: normal; overflow-wrap: break-word; hyphens: auto; }
.markdown-body h1,.markdown-body h2,.markdown-body h3{margin-top:1.2rem}
.markdown-body p,.markdown-body li{line-height:1.6}
.markdown-body table{width:100%;border-collapse:collapse}
.markdown-body th,.markdown-body td{border:1px solid #e5e7eb;padding:8px;font-size:14px}
.markdown-body th{background:#f9fafb}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---- session
if "auth_ok" not in st.session_state: st.session_state.auth_ok = False
if "user_id" not in st.session_state: st.session_state.user_id = ""
if "history" not in st.session_state: st.session_state.history: List[Dict[str,str]] = []
if "last_answer" not in st.session_state: st.session_state.last_answer: RegulAIteAnswer|None = None

# ---- rendering helpers (display-only; safe)
def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _unescape_newlines(text: str) -> str:
    # Convert visible \n to actual newlines if needed
    return text.replace("\\n", "\n") if "\\n" in text and "\n" not in text else text

def _find_json_blob(s: str) -> Dict[str, Any] | None:
    """Find { ... } in a string and parse to dict if possible."""
    s = _strip_code_fences(s)
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return None
    raw = m.group(0)
    try:
        return json.loads(raw)
    except Exception:
        # Tolerate trailing commas
        raw2 = re.sub(r",\s*}", "}", raw)
        raw2 = re.sub(r",\s*]", "]", raw2)
        try:
            return json.loads(raw2)
        except Exception:
            # Try to salvage raw_markdown specifically
            m2 = re.search(r'"raw_markdown"\s*:\s*"(.*)"\s*(,|\})', raw, flags=re.DOTALL)
            if m2:
                val = m2.group(1)
                val = val.replace(r"\\n", "\n").replace(r"\\t", "\t").replace(r"\\\"", "\"")
                return {"raw_markdown": val}
            return None

def _format_per_source(per_source: Dict[str, Any]) -> str:
    if not isinstance(per_source, dict) or not per_source:
        return ""
    lines = ["## Evidence by Framework"]
    for fw, quotes in per_source.items():
        lines.append(f"**{fw}**")
        if isinstance(quotes, list):
            for q in quotes:
                q = str(q)
                lines.append(f"- {_unescape_newlines(q).strip()}")
    return "\n".join(lines)

def _normalize_to_markdown(text: str) -> str:
    """
    Accept:
      - clean markdown string
      - markdown with visible \n
      - a JSON-ish blob containing raw_markdown / summary / comparison_table_md / per_source
    Return: clean markdown for display.
    """
    text = text or ""
    # 1) If we can parse JSON, format from keys
    blob = _find_json_blob(text)
    if isinstance(blob, dict) and blob:
        parts: List[str] = []
        # Prefer raw_markdown if present
        raw_md = blob.get("raw_markdown")
        if isinstance(raw_md, str) and raw_md.strip():
            parts.append(_unescape_newlines(_strip_code_fences(raw_md.strip())))
        else:
            # Compose from pieces
            summary = blob.get("summary")
            if isinstance(summary, str) and summary.strip():
                parts.append("## Summary")
                parts.append(_unescape_newlines(summary.strip()))
            cmp_md = blob.get("comparison_table_md")
            if isinstance(cmp_md, str) and cmp_md.strip():
                parts.append("## Comparison")
                parts.append(_unescape_newlines(_strip_code_fences(cmp_md.strip())))
            ps = blob.get("per_source") or {}
            ps_md = _format_per_source(ps) if isinstance(ps, dict) else ""
            if ps_md:
                parts.append(ps_md)
        if parts:
            return "\n\n".join(parts).strip()

    # 2) Not JSON → fix visible \n then show
    return _unescape_newlines(_strip_code_fences(text)).strip()

def _coerce_answer_to_markdown(ans: RegulAIteAnswer) -> str:
    """Use your model’s as_markdown(), but normalize if it’s messy."""
    try:
        md = ans.as_markdown() or ""
    except Exception:
        md = ""
    md = _normalize_to_markdown(md)
    return md if md else "_No answer produced._"

def render_message(role: str, md: str, meta: str = ""):
    kind = "regu-user" if role == "user" else "regu-assistant"
    st.markdown(
        f'<div class="regu-msg {kind}"><div class="markdown-body">{md}</div>'
        f'{f"<div class=meta>{meta}</div>" if meta else ""}</div>',
        unsafe_allow_html=True,
    )

def _ts() -> str:
    return time.strftime("%H:%M")

# ---- login (unchanged)
def auth_ui():
    st.markdown("## 🔐 RegulAIte Login")
    with st.form("login_form"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            if PRESET_USERS.get(u) == p:
                st.session_state.auth_ok = True
                st.session_state.user_id = u
                st.session_state.history = load_chat(u)
                st.success(f"Welcome {u}!")
                st.rerun()
            else:
                st.error("Invalid username or password.")

if not st.session_state.auth_ok:
    auth_ui(); st.stop()

USER = st.session_state.user_id

# ---- sidebar (unchanged)
with st.sidebar:
    st.header("Session")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Clear chat"):
            clear_chat(USER)
            st.session_state.history=[]
            st.session_state.last_answer=None
            st.rerun()
    with c2:
        if st.button("Sign out"):
            st.session_state.auth_ok=False
            st.session_state.user_id=""
            st.rerun()
    st.markdown("---")
    st.header("Status")
    st.markdown(
        f'<span class="badge {"ok" if VECTOR_STORE_ID else "warn"}">Vector store: {"connected" if VECTOR_STORE_ID else "not connected"}</span> '
        f'<span class="badge {"ok" if LLM_KEY_AVAILABLE else "err"}">LLM API key: {"available" if LLM_KEY_AVAILABLE else "missing"}</span>',
        unsafe_allow_html=True
    )

# ---- main header (unchanged baseline)
st.markdown(f"## {APP_NAME}")

first_q = st.text_input(
    "Ask a question",
    placeholder="e.g., CBB disclosure requirements for large exposures (compare with Basel)?",
)

def run_query(q: str):
    if not q.strip(): return
    # Avoid accidental double-submit of same question
    if st.session_state.history and st.session_state.history[-1]["role"] == "user" \
       and st.session_state.history[-1]["content"].strip() == q.strip():
        return

    append_turn(USER, "user", q)
    st.session_state.history.append({"role": "user", "content": q, "meta": _ts()})
    save_chat(USER, st.session_state.history)

    with st.spinner("Thinking…"):
        try:
            ans: RegulAIteAnswer = ask(
                query=q,
                user_id=USER,
                history=st.session_state.history,
                k_hint=12,
                evidence_mode=True,
                mode_hint="long",
                web_enabled=True,
                vec_id=VECTOR_STORE_ID or None,
                model=DEFAULT_MODEL,
            )
        except Exception as e:
            ans = RegulAIteAnswer(raw_markdown=f"### Error\nCould not complete the request.\n\nDetails: {e}")

    md = _coerce_answer_to_markdown(ans)
    append_turn(USER, "assistant", md)
    st.session_state.history.append({"role": "assistant", "content": md, "meta": " "})
    st.session_state.last_answer = ans
    save_chat(USER, st.session_state.history)

# First Ask button (unchanged)
cbtn, _ = st.columns([1,6])
with cbtn:
    if st.button("Ask", type="primary", use_container_width=True) and first_q:
        run_query(first_q)

# Render chat (now normalized for both old/new content)
for turn in st.session_state.history:
    clean = _normalize_to_markdown(turn["content"])
    render_message(turn["role"], clean, turn.get("meta",""))

# Sticky bottom composer (unchanged addition)
follow_q = st.chat_input("Type a follow-up…")
if follow_q:
    run_query(follow_q)

# Simple follow-up chips (unchanged)
def render_followups():
    suggs = [
        "Board approval thresholds for large exposures",
        "Monthly reporting checklist for large exposures",
        "Escalation steps for breaches/exceptions",
        "Stress-test scenarios for concentration risk",
        "KRIs and metrics for exposure concentration",
        "Differences CBB vs Basel: connected parties",
    ]
    st.caption("Try a follow-up:")
    cols = st.columns(3)
    for i, s in enumerate(suggs):
        with cols[i % 3]:
            if st.button(s, key=f"chip_{len(st.session_state.history)}_{i}", use_container_width=True):
                run_query(s)

render_followups()
