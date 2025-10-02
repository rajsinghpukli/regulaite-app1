from __future__ import annotations
import os, re, time, json
from typing import Any, Dict, List
from datetime import datetime
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

# ---- Updated preset users ----
PRESET_USERS = {
    "user1@khaleeji": "abcd@1234",
    "user2@khaleeji": "abcd@1234",
    "user3@khaleeji": "abcd@1234",
}

# ---- Branding / assets ----
BRAND_BG = "#EAF3FF"       # light blue
BRAND_PRIMARY = "#0C5ECD"  # accent
BRAND_DARK = "#2A2F36"     # headings

# If your image is in rag/assets/khaleeji_logo.png, change this to that exact path.
LOGO_PATH = "rag/khaleeji_logo.png"

# Logo sizes
LOGIN_LOGO_WIDTH = 180   # bigger on login
HEADER_LOGO_WIDTH = 120  # bigger on page header

st.set_page_config(page_title=APP_NAME, page_icon="🧭", layout="wide")

# -------------------- Styles (display-only) --------------------
CSS = f"""
<style>
.stApp {{ background:{BRAND_BG}; }}
.block-container {{ max-width: 1180px; }}

.kh-header {{ display:flex; align-items:center; gap:14px; padding:6px 0 12px 0; }}
.kh-title {{ font-size:28px; font-weight:800; color:{BRAND_DARK}; letter-spacing:.2px; }}
.kh-subtitle {{ color:#5a6473; font-size:13px; margin-top:-4px; }}

.badge{{display:inline-block;padding:4px 8px;border-radius:999px;font-size:12px;margin-right:6px;border:1px solid #C9E0FF;background:#E7F2FF;color:#164C96}}

.regu-msg{{border-radius:14px;padding:14px 16px;box-shadow:0 1px 2px #0001;border:1px solid #0001;margin-bottom:12px;background:#ffffffcc}}
.regu-user{{background:#FFFFFF}}
.regu-assistant{{background:#F7FAFF}}
.hdr{{font-size:12px;font-weight:700;margin-bottom:6px;opacity:.8}}
.hdr .u{{color:#1f2937}}
.hdr .a{{color:#0f766e}}

.markdown-body {{ width:100%; word-break: normal; overflow-wrap: break-word; hyphens: auto; }}
.markdown-body h1,.markdown-body h2,.markdown-body h3{{margin-top:1.1rem}}
.markdown-body p,.markdown-body li{{line-height:1.6}}
.markdown-body table{{width:100%;border-collapse:collapse}}
.markdown-body th,.markdown-body td{{border:1px solid #e5e7eb;padding:8px;font-size:14px}}
.markdown-body th{{background:#f9fafb}}

.stButton > button[kind="primary"] {{ background:{BRAND_PRIMARY}; border-color:{BRAND_PRIMARY}; }}
.stButton > button {{ border-radius:14px !important; }}

section.main > div.block-container {{ padding-top: 0.8rem; padding-bottom: 5rem; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# -------------------- Session --------------------
if "auth_ok" not in st.session_state: st.session_state.auth_ok = False
if "user_id" not in st.session_state: st.session_state.user_id = ""
if "history" not in st.session_state: st.session_state.history: List[Dict[str,str]] = []
if "last_answer" not in st.session_state: st.session_state.last_answer: RegulAIteAnswer|None = None

# -------------------- Helpers (display-only) --------------------
def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _unescape_newlines(text: str) -> str:
    return text.replace("\\n", "\n") if "\\n" in text and "\n" not in text else text

def _find_json_blob(s: str) -> Dict[str, Any] | None:
    s = _strip_code_fences(s)
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m: return None
    raw = m.group(0)
    try:
        return json.loads(raw)
    except Exception:
        raw2 = re.sub(r",\s*}", "}", raw)
        raw2 = re.sub(r",\s*]", "]", raw2)
        try:
            return json.loads(raw2)
        except Exception:
            m2 = re.search(r'"raw_markdown"\s*:\s*"(.*)"\s*(,|\})', raw, flags=re.DOTALL)
            if m2:
                val = m2.group(1).replace(r"\\n","\n").replace(r"\\t","\t").replace(r"\\\"","\"")
                return {"raw_markdown": val}
            return None

def _format_per_source(per_source: Dict[str, Any]) -> str:
    if not isinstance(per_source, dict) or not per_source: return ""
    lines = ["## Evidence by Framework"]
    for fw, quotes in per_source.items():
        lines.append(f"**{fw}**")
        if isinstance(quotes, list):
            for q in quotes:
                lines.append(f"- {_unescape_newlines(str(q)).strip()}")
    return "\n".join(lines)

def _normalize_to_markdown(text: str) -> str:
    text = text or ""
    blob = _find_json_blob(text)
    if isinstance(blob, dict) and blob:
        parts: List[str] = []
        raw_md = blob.get("raw_markdown")
        if isinstance(raw_md, str) and raw_md.strip():
            parts.append(_unescape_newlines(_strip_code_fences(raw_md.strip())))
        else:
            summary = blob.get("summary")
            if isinstance(summary, str) and summary.strip():
                parts += ["## Summary", _unescape_newlines(summary.strip())]
            cmp_md = blob.get("comparison_table_md")
            if isinstance(cmp_md, str) and cmp_md.strip():
                parts += ["## Comparison", _unescape_newlines(_strip_code_fences(cmp_md.strip()))]
            ps = blob.get("per_source")
            if isinstance(ps, dict):
                ps_md = _format_per_source(ps)
                if ps_md: parts.append(ps_md)
        if parts: return "\n\n".join(parts).strip()
    return _unescape_newlines(_strip_code_fences(text)).strip()

def _coerce_answer_to_markdown(ans: RegulAIteAnswer) -> str:
    try:
        md = ans.as_markdown() or ""
    except Exception:
        md = ""
    md = _normalize_to_markdown(md)
    return md if md else "_No answer produced._"

def render_message(role: str, md: str, meta: str = ""):
    kind = "regu-user" if role == "user" else "regu-assistant"
    who  = '<span class="u">You</span>' if role == "user" else '<span class="a">Assistant</span>'
    st.markdown(
        f'<div class="regu-msg {kind}">'
        f'  <div class="hdr">{who}</div>'
        f'  <div class="markdown-body">{md}</div>'
        f'  {f"<div class=meta>{meta}</div>" if meta else ""}'
        f'</div>',
        unsafe_allow_html=True,
    )

def _ts() -> str:
    return time.strftime("%H:%M")

# ---- NEW: export helpers ----
def _latest_assistant_md() -> str:
    """Return the most recent assistant message as clean markdown."""
    for t in reversed(st.session_state.history):
        if t.get("role") == "assistant":
            return _normalize_to_markdown(t.get("content", ""))
    return ""

def _chat_history_as_markdown() -> str:
    """Readable Markdown export of the whole conversation."""
    parts = []
    for t in st.session_state.history:
        role = t.get("role", "assistant").title()
        content = _normalize_to_markdown(t.get("content", ""))
        timestamp = t.get("meta", "")
        header = f"### {role} {f'({timestamp})' if timestamp else ''}"
        parts.append(f"{header}\n\n{content}\n")
    return "\n\n".join(parts).strip()

def _last_answer_as_html() -> str:
    """Wrap the last assistant answer in a basic HTML page for easy Print-to-PDF."""
    md = _latest_assistant_md()
    safe = md.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"""<!doctype html>
<html><head>
<meta charset="utf-8">
<title>RegulAIte Answer</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; padding: 24px; }}
pre, code {{ background:#f6f8fa; padding:2px 4px; border-radius:4px; }}
table {{ border-collapse: collapse; width:100%; }}
th, td {{ border:1px solid #e5e7eb; padding:8px; }}
h1, h2, h3 {{ margin-top:1.2em; }}
</style>
</head><body>
<pre style="white-space:pre-wrap;word-wrap:break-word;">{safe}</pre>
</body></html>"""

# -------------------- Login (bigger logo; logic unchanged) --------------------
def auth_ui():
    with st.container():
        cols = st.columns([1, 5, 1])
        with cols[1]:
            st.markdown('<div class="kh-header" style="justify-content:center;">', unsafe_allow_html=True)
            if os.path.exists(LOGO_PATH) or LOGO_PATH.startswith("http"):
                st.image(LOGO_PATH, width=LOGIN_LOGO_WIDTH)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='kh-title' style='text-align:center;margin-top:8px;'>Khaleeji • RegulAIte</div>"
                f"<div class='kh-subtitle' style='text-align:center;'>Secure login</div>",
                unsafe_allow_html=True,
            )
            with st.form("login_form"):
                u = st.text_input("Username")
                p = st.text_input("Password", type="password")
                if st.form_submit_button("Sign in"):
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

# -------------------- Header (bigger logo) --------------------
with st.container():
    hcol1, hcol2 = st.columns([1, 10])
    with hcol1:
        if os.path.exists(LOGO_PATH) or LOGO_PATH.startswith("http"):
            st.image(LOGO_PATH, width=HEADER_LOGO_WIDTH)
    with hcol2:
        st.markdown(
            f"""
            <div class="kh-header">
              <div>
                <div class="kh-title">Khaleeji • RegulAIte</div>
                <div class="kh-subtitle">Regulatory assistant (pilot)</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# -------------------- Sidebar (session + exports; no status) --------------------
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
    st.header("Export")

    # Latest answer downloads
    last_md = _latest_assistant_md()
    if last_md:
        fname_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        st.download_button(
            "⬇️ Download last answer (Markdown)",
            data=last_md,
            file_name=f"regulaite_answer_{fname_stamp}.md",
            mime="text/markdown",
            use_container_width=True,
        )

        last_html = _last_answer_as_html()
        st.download_button(
            "⬇️ Download last answer (HTML)",
            data=last_html,
            file_name=f"regulaite_answer_{fname_stamp}.html",
            mime="text/html",
            use_container_width=True,
        )

    # Full chat history download (JSON)
    hist_json = json.dumps(st.session_state.history, ensure_ascii=False, indent=2)
    st.download_button(
        "⬇️ Download chat history (JSON)",
        data=hist_json,
        file_name=f"regulaite_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True,
    )

    # Optional: Markdown export of full chat
    chat_md = _chat_history_as_markdown()
    if chat_md:
        st.download_button(
            "⬇️ Download chat (Markdown)",
            data=chat_md,
            file_name=f"regulaite_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True,
        )

# -------------------- Query execution (unchanged) --------------------
def run_query(q: str):
    if not q.strip(): return
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
    st.session_state.history.append({"role": "assistant", "content": md, "meta": ""})
    st.session_state.last_answer = ans
    save_chat(USER, st.session_state.history)

# -------------------- Render existing chat --------------------
for turn in st.session_state.history:
    clean = _normalize_to_markdown(turn["content"])
    render_message(turn["role"], clean, turn.get("meta",""))

# -------------------- Follow-up chips (kept) --------------------
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
                st.rerun()

render_followups()

# -------------------- Single sticky input --------------------
prompt = st.chat_input("Type your question…")
if prompt:
    run_query(prompt)
    st.rerun()
