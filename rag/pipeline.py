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
def _history_to_brief(history: List[Dict[str, str]], max_pairs: int = 8) -> str:
    """Convert last few turns of history into a compact text summary."""
    if not history:
        return ""
    turns = history[-(max_pairs * 2):]
    lines: List[str] = []
    for h in turns:
        role = h.get("role")
        content = (h.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            lines.append(f"User asked: {content}")
        else:
            lines.append(f"Assistant replied (extract): {content[:600]}")
    return "\n".join(lines[-(max_pairs * 2):])

def _schema_dict() -> Dict[str, Any]:
    """Schema we ask the model to return."""
    return {
        "type": "object",
        "properties": {
            "raw_markdown": {"type": "string"},
            "summary": {"type": "string"},
            "per_source": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "notes": {"type": "string"},
                        "quotes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "framework": {"type": "string", "enum": ["IFRS","AAOIFI","CBB","InternalPolicy"]},
                                    "snippet": {"type": "string"},
                                    "citation": {"type": "string"},
                                },
                                "required": ["framework", "snippet"],
                            },
                        },
                    },
                },
            },
            "comparison_table_md": {"type": "string"},
            "follow_up_suggestions": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["raw_markdown", "summary", "per_source", "follow_up_suggestions"],
    }

def _schema_prompt() -> str:
    """Prompt asking model to respect memo style and schema."""
    style_bar = (
        "STYLE DIRECTIVES:\n"
        "- Write as a regulatory memo (sections, table, recommendation).\n"
        "- Always include either a workflow or reporting matrix in recommendations.\n"
        "- No 'Meaning:' lines. Integrate interpretation into prose.\n"
        "- Use compact inline citations like [IFRS 7 §35].\n"
        "- Omit frameworks if no evidence, never write 'N/A'.\n"
    )
    return style_bar + "\nReturn ONE JSON object only:\n" + json.dumps(_schema_dict(), ensure_ascii=False)

def _parse_json(text: str) -> Dict[str, Any]:
    """Extract and parse JSON from model output."""
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {}
    raw = m.group(0)
    try:
        return json.loads(raw)
    except Exception:
        # clean trailing commas
        raw = re.sub(r",\s*}", "}", raw)
        raw = re.sub(r",\s*]", "]", raw)
        try:
            return json.loads(raw)
        except Exception:
            return {}

def _mode_tokens(mode: str) -> int:
    return {"short": 900, "long": 2200, "research": 3200}.get(mode, 2600)

# ---------- main ----------
def ask(
    query: str,
    *,
    user_id: str,
    history: List[Dict[str, str]],
    k_hint: int = 12,
    evidence_mode: bool = True,
    mode_hint: str | None = "research",
    web_enabled: Union[bool, str] = True,
    vec_id: Optional[str] = None,
    model: Optional[str] = None,
) -> RegulAIteAnswer:
    """Main entry: answer a user query as a structured regulatory memo."""
    mode = normalize_mode(mode_hint or "research")
    convo_brief = _history_to_brief(history)
    max_out = _mode_tokens(mode)

    sys_inst = build_system_instruction(k_hint=k_hint, evidence_mode=evidence_mode, mode=mode)

    # --- Web enrichment (always on) ---
    results = ddg_search(query, max_results=max(8, k_hint))
    web_context = ""
    if results:
        lines = ["Web snippets (use prudently; internal docs take precedence):"]
        for i, r in enumerate(results, 1):
            snippet = (r.get("snippet") or "").strip()[:400]
            title = r.get("title") or ""
            url = r.get("url") or ""
            lines.append(f"{i}. {title} — {url}\n   Snippet: {snippet}")
        web_context = "\n".join(lines)

    messages = [
        {"role": "system", "content": sys_inst},
        {"role": "system", "content": STYLE_GUIDE},
        {"role": "system", "content": FEW_SHOT_EXAMPLE},
        {"role": "system", "content": _schema_prompt()},
        {"role": "user", "content": f"Conversation so far (brief):\n{convo_brief}"},
    ]
    if web_context:
        messages.append({"role": "user", "content": web_context})
    messages.append({"role": "user", "content": query})

    chat_model = (model or os.getenv("CHAT_MODEL") or os.getenv("RESPONSES_MODEL") or "gpt-4o-mini").strip()
    resp = client.chat.completions.create(
        model=chat_model,
        temperature=0.35,
        top_p=0.95,
        max_tokens=max_out,
        messages=messages,
    )

    text = ""
    try:
        text = resp.choices[0].message.content or ""
    except Exception:
        text = ""

    data = _parse_json(text or "")
    if not data:
        return DEFAULT_EMPTY

    # --- Guarantee follow-ups ---
    if not data.get("follow_up_suggestions"):
        topic = query.strip() or "this topic"
        data["follow_up_suggestions"] = [
            f"Draft a policy workflow for approvals of {topic}.",
            f"Design a board/committee reporting matrix for {topic}.",
            f"List audit pitfalls and lessons for {topic}.",
            f"Suggest KRIs and thresholds for {topic}.",
            f"Outline escalation and breach handling steps for {topic}.",
        ]

    # --- Guarantee recommendation section in raw_markdown ---
    raw_md = (data.get("raw_markdown") or "").strip()
    if "Recommendation for Khaleeji Bank" not in raw_md:
        rec_text = (
            "\n\n### Recommendation for Khaleeji Bank\n"
            "- Establish board-approved policy thresholds (10% reporting, 25% max exposure).\n"
            "- Require connected exposures to go through Credit → Risk → Board → CBB notification.\n"
            "- Maintain a reporting matrix showing which exposures go to which committee and how often.\n"
        )
        data["raw_markdown"] = raw_md + rec_text

    try:
        return RegulAIteAnswer(**data)
    except ValidationError:
        return DEFAULT_EMPTY
