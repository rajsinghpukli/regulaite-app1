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

# ---------------- helpers ----------------
def _history_to_brief(history: List[Dict[str, str]] | None, max_pairs: int = 8) -> str:
    if not history:
        return ""
    turns = history[-(max_pairs * 2):]
    out = []
    for h in turns:
        role = h.get("role", "")
        content = (h.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            out.append(f"User: {content}")
        else:
            out.append(f"Assistant: {content[:700]}")
    return "\n".join(out)

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _parse_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    text = _strip_code_fences(text)
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {}
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
                val = m2.group(1)
                val = val.replace(r"\\n", "\n").replace(r"\\t", "\t").replace(r"\\\"", "\"")
                return {"raw_markdown": val}
            return {}

def _mode_tokens(mode: str) -> int:
    return {"short": 900, "long": 2600, "research": 3600}.get(mode, 2200)

def _unescape_field(v: Optional[str]) -> Optional[str]:
    if not isinstance(v, str):
        return v
    if "\\n" in v and "\n" not in v:
        v = v.replace("\\n", "\n")
    return _strip_code_fences(v).strip()

def _weak(ans: RegulAIteAnswer, query: str) -> bool:
    md = (ans.raw_markdown or "").lower()
    too_short = len(md) < 400
    says_not_found = "not found" in md and any(k in query.lower() for k in ["cbb", "rulebook", "cm-5"])
    no_evidence = not (ans.per_source or {})
    return too_short or says_not_found or no_evidence

# ---- tiny intent (formatting only; does not change retrieval) ----
def _detect_intent(q: str) -> Dict[str, bool]:
    ql = (q or "").lower()
    return_only = any(k in ql for k in ["return only", "only:", "only the", "just the", "no other", "nothing else", "no prose", "no explanation"])
    quote_only  = any(k in ql for k in ["quote", "quote verbatim", "verbatim", "exact sentence", "exact line", "cite-only", "cite only"])
    list_ids    = any(k in ql for k in ["list only the section ids", "list only the ids", "section ids present", "ids present"])
    bis_only    = ("bis.org" in ql or "bcbs " in ql) and return_only
    scenario    = any(k in ql for k in ["scenario", "deliver:", "board-ready", "controls", "kris", "workflow", "recommendation", "decision-grade", "exposure calculation"])
    concise = return_only or quote_only or list_ids or bis_only
    return {"return_only":return_only, "quote_only":quote_only, "list_ids":list_ids, "bis_only":bis_only, "scenario":scenario, "concise":concise}

# ---------------- Responses API (File Search) helpers ----------------
def _responses_build_messages(sys_inst: str, style_msg: str, convo_brief: str, query: str) -> List[Dict[str, Any]]:
    # We pass system & style as separate system messages, then a compact conversation brief, then the query.
    return [
        {"role": "system", "content": sys_inst},
        {"role": "system", "content": STYLE_GUIDE},
        {"role": "system", "content": FEW_SHOT_EXAMPLE},
        {"role": "system", "content": style_msg},
        {"role": "user", "content": f"Conversation so far (brief):\n{convo_brief}"},
        {"role": "user", "content": query},
    ]

def _responses_try_file_search(query: str, sys_inst: str, style_msg: str, convo_brief: str, model: str, vector_store_id: Optional[str]) -> Optional[RegulAIteAnswer]:
    """
    Use OpenAI Responses + file_search if vector_store_id is available.
    Returns RegulAIteAnswer or None if the call fails.
    """
    if not vector_store_id:
        return None

    messages = _responses_build_messages(sys_inst, style_msg, convo_brief, query)

    try:
        # Attach the vector store to the *last user message*
        # SDK supports "attachments" on messages; tools include file_search
        resp = client.responses.create(
            model=model,
            messages=[
                # first 5 messages unchanged
                {"role": m["role"], "content": m["content"]} for m in messages[:-1]
            ] + [{
                "role": "user",
                "content": messages[-1]["content"],
                "attachments": [{"vector_store_id": vector_store_id}],
            }],
            tools=[{"type": "file_search"}],
            temperature=1,
            top_p=1,
            max_output_tokens= _mode_tokens("long"),
        )
    except Exception:
        return None

    # Fast path: if SDK exposes output_text, use it
    raw_md = ""
    try:
        raw_md = (resp.output_text or "").strip()
    except Exception:
        raw_md = ""

    # If no flat text, attempt to reconstruct from the content blocks
    if not raw_md:
        try:
            # resp.output is a list of items; find text content
            chunks = []
            for item in getattr(resp, "output", []) or []:
                if item.get("type") in ("message", "output_text"):
                    txt = (item.get("content") or "").strip()
                    if txt:
                        chunks.append(txt)
            raw_md = "\n\n".join(chunks).strip()
        except Exception:
            raw_md = ""

    if not raw_md:
        return None

    # The model may have returned our JSON schema or plain markdown.
    data = _parse_json(raw_md)
    if data:
        try:
            return RegulAIteAnswer(**data)
        except ValidationError:
            md = data.get("raw_markdown") or ""
            return RegulAIteAnswer(raw_markdown=_unescape_field(md) or "")
    else:
        return RegulAIteAnswer(raw_markdown=_unescape_field(raw_md) or "")

# ---------------- main ----------------
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
    vector_store_id = (vec_id or os.getenv("OPENAI_VECTOR_STORE_ID") or "").strip() or None

    intent = _detect_intent(query)

    sys_inst = build_system_instruction(k_hint=k_hint, evidence_mode=evidence_mode, mode=mode)

    # Style/Schema (formatting only)
    if intent["concise"]:
        style_msg = (
            "Follow any output restriction STRICTLY (e.g., 'quote verbatim', 'return only URL/date/code', 'IDs only'). "
            "No extra prose, no headings, no boilerplate."
        )
        schema_msg = None  # allow plain text
    elif intent["scenario"]:
        style_msg = (
            "Board-grade scenario. Use clean headings and tables as needed. Include controls/KRIs/workflow only if requested. "
            "Be concise and decision-focused."
        )
        schema_msg = (
            "Return ONE JSON object ONLY with keys: "
            "raw_markdown (string), summary (string, optional), per_source (object, optional), "
            "comparison_table_md (string, optional), follow_up_suggestions (array of strings, optional). "
            "No prose outside JSON."
        )
    else:
        style_msg = (
            "Answer naturally in well-structured Markdown. Use headings/tables if helpful. "
            "Do NOT add generic workflow/matrix unless the question requires them."
        )
        schema_msg = (
            "Return ONE JSON object ONLY with keys: "
            "raw_markdown (string), summary (string, optional), per_source (object, optional), "
            "comparison_table_md (string, optional), follow_up_suggestions (array of strings, optional). "
            "No prose outside JSON."
        )

    # ---------- 1) Try OpenAI File Search (if vector store available) ----------
    ans_fs: Optional[RegulAIteAnswer] = _responses_try_file_search(
        query=query,
        sys_inst=sys_inst,
        style_msg=style_msg,
        convo_brief=convo_brief,
        model=chat_model,
        vector_store_id=vector_store_id,
    )

    if ans_fs and (ans_fs.raw_markdown or "").strip():
        # Optionally ensure follow-ups
        if not ans_fs.follow_up_suggestions:
            topic = (query or "this topic").strip()
            ans_fs.follow_up_suggestions = [
                f"What approval thresholds and board oversight apply to {topic}?",
                f"Draft a closure checklist for {topic} with controls and required evidence.",
                f"What fields belong in the monthly board pack for {topic}?",
                f"How should breaches/exceptions for {topic} be escalated and documented?",
                f"What stress-test scenarios are relevant for {topic} and how to calibrate them?",
                f"What are the key risks, controls, and KRIs for {topic} (with metrics)?",
            ]
        return ans_fs

    # ---------- 2) Fallback: your existing chat completions path ----------
    # (We keep this so normal answers still work if File Search is down or not configured)
    # Optional web context for non-concise asks
    web_context = ""
    if bool(web_enabled) and not intent["concise"]:
        try:
            results = ddg_search(query, max_results=max(8, k_hint))
        except Exception:
            results = []
        if results:
            lines = ["Web snippets (use prudently; internal docs take precedence):"]
            for i, r in enumerate(results, 1):
                title = r.get("title") or ""
                url = r.get("url") or r.get("href") or ""
                snippet = (r.get("snippet") or r.get("body") or "").strip()[:400]
                lines.append(f"{i}. {title} — {url}\n   Snippet: {snippet}")
            web_context = "\n".join(lines)

     #{"role": "system", "content": FEW_SHOT_EXAMPLE},
    messages = [
        {"role": "system", "content": sys_inst},
        {"role": "system", "content": STYLE_GUIDE},
       
        {"role": "system", "content": style_msg},
        {"role": "user", "content": f"Conversation so far (brief):\n{convo_brief}"},
    ]
    if web_context:
        messages.append({"role": "user", "content": web_context})
    messages.append({"role": "user", "content": query})
    if schema_msg:
        messages.insert(3, {"role": "system", "content": schema_msg})

    try:
        resp = client.chat.completions.create(
            model=chat_model,
            temperature=1,
            top_p=1,
            max_tokens=max_out,
            messages=messages,
        )
        text = resp.choices[0].message.content or ""
    except Exception as e:
        return RegulAIteAnswer(raw_markdown=f"### Error\nModel call failed.\n\nDetails: {e}")

    if intent["concise"]:
        raw = _strip_code_fences(text).strip()
        return RegulAIteAnswer(raw_markdown=raw if raw else "not found")

    data = _parse_json(text)
    if data:
        try:
            ans = RegulAIteAnswer(**data)
        except ValidationError:
            md = data.get("raw_markdown") or ""
            ans = RegulAIteAnswer(raw_markdown=_unescape_field(md) or "")
    else:
        md = _strip_code_fences(text).strip()
        ans = RegulAIteAnswer(raw_markdown=_unescape_field(md) or "")

    if not (ans.raw_markdown or "").strip():
        return DEFAULT_EMPTY

    if not ans.follow_up_suggestions:
        topic = (query or "this topic").strip()
        ans.follow_up_suggestions = [
            f"What approval thresholds and board oversight apply to {topic}?",
            f"Draft a closure checklist for {topic} with controls and required evidence.",
            f"What fields belong in the monthly board pack for {topic}?",
            f"How should breaches/exceptions for {topic} be escalated and documented?",
            f"What stress-test scenarios are relevant for {topic} and how to calibrate them?",
            f"What are the key risks, controls, and KRIs for {topic} (with metrics)?",
        ]

    return ans

