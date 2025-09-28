from __future__ import annotations
import os, json, re
from typing import Dict, Any, List, Optional, Union
from openai import OpenAI
from pydantic import ValidationError

from .schema import RegulAIteAnswer, DEFAULT_EMPTY
from .agents import build_system_instruction
from .router import normalize_mode
from .websearch import ddg_search
from .prompts import STYLE_GUIDE, FEW_SHOT_EXAMPLE

client = OpenAI()

# ---------- helpers ----------
def _history_to_brief(history: List[Dict[str, str]] | None, max_pairs: int = 8) -> str:
    if not history: return ""
    turns = history[-(max_pairs * 2):]
    lines: List[str] = []
    for h in turns:
        role = h.get("role"); content = (h.get("content") or "").strip()
        if not content: continue
        lines.append(("User: " if role == "user" else "Assistant: ") + content[:700])
    return "\n".join(lines)

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _parse_json(text: str) -> Dict[str, Any]:
    if not text: return {}
    text = _strip_code_fences(text)
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m: return {}
    raw = m.group(0)
    try:
        return json.loads(raw)
    except Exception:
        raw2 = re.sub(r",\s*}", "}", raw); raw2 = re.sub(r",\s*]", "]", raw2)
        try: return json.loads(raw2)
        except Exception:
            m2 = re.search(r'"raw_markdown"\s*:\s*"(.*)"\s*(,|\})', raw, flags=re.DOTALL)
            if m2:
                val = m2.group(1).replace(r"\\n","\n").replace(r"\\t","\t").replace(r"\\\"","\"")
                return {"raw_markdown": val}
            return {}

def _mode_tokens(mode: str) -> int:
    return {"short": 900, "long": 2600, "research": 3600}.get(mode, 2200)

def _unescape_field(v: Optional[str]) -> Optional[str]:
    if not isinstance(v, str): return v
    if "\\n" in v and "\n" not in v:
        v = v.replace("\\n", "\n")
    return _strip_code_fences(v).strip()

def _weak(ans: RegulAIteAnswer, query: str) -> bool:
    md = (ans.raw_markdown or "").lower()
    too_short = len(md) < 400
    says_not_found = "not found" in md and any(k in query.lower() for k in ["cbb", "rulebook", "cm-5"])
    no_evidence = not (ans.per_source or {})
    return too_short or says_not_found or no_evidence

def _doc_only_from_query(q: str) -> bool:
    ql = q.lower()
    return ("cite only" in ql) or ("cbb rulebook" in ql and "bis" not in ql and "basel" not in ql and "web" not in ql)

def _needs_web_bias(q: str) -> bool:
    ql = (q or "").lower()
    # Force a web pass for BIS/bis.org style questions (metadata/URL)
    return ("bis.org" in ql) or (" bcbs" in ql) or ("bis " in ql and "url" in ql)

def _call_llm(messages: List[Dict[str,str]], model: str, max_out: int) -> RegulAIteAnswer:
    resp = client.chat.completions.create(
        model=model, temperature=0.35, top_p=0.95, max_tokens=max_out, messages=messages
    )
    text = resp.choices[0].message.content or ""
    data = _parse_json(text)
    if data:
        if "raw_markdown" in data: data["raw_markdown"] = _unescape_field(data.get("raw_markdown"))
        if "summary" in data: data["summary"] = _unescape_field(data.get("summary"))
        if "comparison_table_md" in data: data["comparison_table_md"] = _unescape_field(data.get("comparison_table_md"))
        try:
            return RegulAIteAnswer(**data)
        except ValidationError:
            md = (data.get("raw_markdown") or "") if isinstance(data, dict) else ""
            return RegulAIteAnswer(raw_markdown=_unescape_field(md) or "")
    else:
        md = _strip_code_fences(text).strip()
        return RegulAIteAnswer(raw_markdown=_unescape_field(md) or "")

# ---------- STRICT CITE-ONLY MODE ----------
_STRICT_KEYWORDS = [
    "cite only", "exact sentence", "verbatim", "quote verbatim", "return only",
    "cm-5.", "ifrs 7", "ifrs 9", "fas 30", "fas 33", "§"
]
def _is_strict_citation(q: str) -> bool:
    ql = (q or "").lower()
    return any(k in ql for k in _STRICT_KEYWORDS)

# Regex to harvest literal quotes and references
_QUOTE_RE = re.compile(r'"[^"\n]{5,}"')  # some quoted text
_REF_RE   = re.compile(
    r'\b('
    r'CM-\d+\.\d+(?:\.\d+)?'             # CBB CM-5.2.1 etc.
    r'|IFRS\s*\d+(?:\.\d+)?(?:\.\d+)?'    # IFRS 9.5.5.1 etc.
    r'|IFRS\s*7\.\d+\w*'                  # IFRS 7.xx
    r'|FAS\s*\d+\s*§\s*[\w\.-]+'          # FAS 33 §4.2
    r')\b',
    re.IGNORECASE
)

def _reduce_to_quotes_only(ans: RegulAIteAnswer) -> RegulAIteAnswer:
    """
    Keep only literal quotes + references in raw_markdown.
    If no quotes are found, return 'not found' (never emit refs-only junk).
    If quotes exist but no clear refs, return just the quotes.
    """
    body = (ans.raw_markdown or "").strip()
    quotes = _QUOTE_RE.findall(body)
    refs   = _REF_RE.findall(body)
    if not quotes:
        return RegulAIteAnswer(raw_markdown="not found")

    lines: List[str] = []
    if refs:
        # naive pairing: interleave up to min length
        n = min(len(quotes), len(refs))
        for i in range(n):
            lines.append(f'{quotes[i].strip()}  \n— {refs[i].strip()}')
        # if more quotes than refs, keep the remaining quotes
        for q in quotes[n:]:
            lines.append(q.strip())
    else:
        for q in quotes:
            lines.append(q.strip())

    return RegulAIteAnswer(raw_markdown="\n\n".join(lines))

# ---------- main ----------
def ask(
    query: str,
    *,
    user_id: Optional[str],
    history: Optional[List[Dict[str, str]]],
    k_hint: int = 12,
    evidence_mode: bool = True,
    mode_hint: str | None = "long",
    web_enabled: Union[bool, str] = True,
    vec_id: Optional[str] = None,
    model: Optional[str] = None,
) -> RegulAIteAnswer:

    mode = normalize_mode(mode_hint)
    convo_brief = _history_to_brief(history)
    max_out = _mode_tokens(mode)
    chat_model = (model or os.getenv("CHAT_MODEL") or os.getenv("RESPONSES_MODEL") or "gpt-4o-mini").strip()

    sys_inst = build_system_instruction(k_hint=k_hint, evidence_mode=evidence_mode, mode=mode)

    strict = _is_strict_citation(query)
    force_web = _needs_web_bias(query)

    if strict:
        schema_msg = (
            "Return ONE JSON object ONLY with key raw_markdown (string). "
            "In raw_markdown return ONLY literal quoted sentence(s) and precise reference(s) "
            "(e.g., \"…\" — CM-5.2.x / IFRS 7.35F / FAS 33 §4.2). "
            "If exact quote in the requested chapter/standard is not found, return exactly: not found. "
            "No summaries, no explanations, no extra text outside JSON."
        )
    else:
        schema_msg = (
            "Return ONE JSON object ONLY with keys: "
            "raw_markdown (string), summary (string), per_source (object), "
            "comparison_table_md (string, optional), follow_up_suggestions (array of strings). "
            "IMPORTANT: Use REAL newlines in raw_markdown and comparison_table_md; "
            "do NOT escape them as \\n. No prose outside JSON."
        )

    # ----- PASS 1: Vector-first -----
    messages = [
        {"role": "system", "content": sys_inst},
        {"role": "system", "content": STYLE_GUIDE},
        {"role": "system", "content": FEW_SHOT_EXAMPLE},
        {"role": "system", "content": schema_msg},
        {"role": "user", "content": f"Conversation so far (brief):\n{convo_brief}"},
        {"role": "user", "content": query},
    ]
    try:
        ans1 = _call_llm(messages, chat_model, max_out)
    except Exception as e:
        return RegulAIteAnswer(raw_markdown=f"### Error\nModel call failed.\n\nDetails: {e}")

    ans = ans1

    # ----- PASS 2: Web-fallback (or forced for BIS) -----
    allow_web = bool(web_enabled) and not _doc_only_from_query(query)
    should_try_web = (allow_web and not strict and (_weak(ans1, query) or force_web))
    if should_try_web:
        results = ddg_search(query, max_results=max(8, k_hint))
        if results:
            lines = ["Web snippets (use prudently; internal docs take precedence):"]
            for i, r in enumerate(results, 1):
                lines.append(f"{i}. {r.get('title','')} — {r.get('url','')}\n   Snippet: {(r.get('snippet') or '')[:400]}")
            web_context = "\n".join(lines)

            messages2 = [
                {"role": "system", "content": sys_inst},
                {"role": "system", "content": STYLE_GUIDE},
                {"role": "system", "content": FEW_SHOT_EXAMPLE},
                {"role": "system", "content": schema_msg},
                {"role": "user", "content": f"Conversation so far (brief):\n{convo_brief}"},
                {"role": "user", "content": "Use this web context ONLY if internal sources were insufficient:\n" + web_context},
                {"role": "user", "content": query},
            ]
            ans2 = _call_llm(messages2, chat_model, max_out)
            if (ans2.raw_markdown or "").strip():
                ans = ans2

    # ----- STRICT POST-PROCESSING -----
    if strict:
        ans = _reduce_to_quotes_only(ans)

    # ----- Safety net -----
    if not (ans.raw_markdown or "").strip() and not (getattr(ans, "summary", "") or "").strip():
        return DEFAULT_EMPTY

    # ----- Helper sections: only for normal, longer narratives -----
    q_l = (query or "").lower()
    is_citation_like = any(k in q_l for k in [
        "cite only", "exact sentence", "verbatim", "quote", "return only",
        "cm-5.", "ifrs 7", "ifrs 9", "fas 30", "fas 33", "§"
    ])
    raw_md = getattr(ans, "raw_markdown", "") or ""
    is_long_narrative = len(raw_md) >= 500

    if (not strict) and (not is_citation_like) and is_long_narrative:
        if "Approval Workflow" not in raw_md and "Reporting Matrix" not in raw_md:
            ans.raw_markdown = (
                f"{raw_md.rstrip()}\n\n"
                "### Approval Workflow\n"
                "Credit → Risk → Shari’ah Supervisory Board (if applicable) → Board → CBB notification\n"
                "\n### Reporting Matrix\n"
                "| Owner | Item | Frequency |\n"
                "|---|---|---|\n"
                "| Risk | Large exposure register | Monthly |\n"
                "| Compliance | CBB submissions | Quarterly |\n"
                "| Board | Connected exposure approvals | Ongoing |\n"
            )

    return ans
