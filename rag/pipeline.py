from __future__ import annotations
import os, json, re
from typing import Dict, Any, List, Optional
from openai import OpenAI
from pydantic import ValidationError
from .schema import RegulAIteAnswer, DEFAULT_EMPTY
from .agents import build_system_instruction
from .router import normalize_mode
from .websearch import ddg_search

client = OpenAI()

# ---------- helpers ----------
def _history_to_brief(history: List[Dict[str, str]], max_pairs: int = 8) -> str:
    if not history:
        return ""
    turns = history[-(max_pairs * 2):]
    lines: List[str] = []
    for h in turns:
        role = h["role"]
        content = h["content"].strip().replace("\n", " ")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


# ---------- main ask ----------
def ask(
    query: str,
    user_id: str,
    history: Optional[List[Dict[str, str]]] = None,
    k_hint: int = 10,
    evidence_mode: bool = True,
    mode_hint: str = "long",
    web_enabled: bool = True,
    vec_id: Optional[str] = None,
    model: str = "gpt-4o-mini"
) -> RegulAIteAnswer:

    mode = normalize_mode(mode_hint)
    sys_inst = build_system_instruction(
        k_hint=k_hint,
        evidence_mode=evidence_mode,
        mode=mode
    )

    # build messages
    messages = [
        {"role": "system", "content": sys_inst},
        {"role": "user", "content": query},
    ]

    if history:
        brief = _history_to_brief(history)
        if brief:
            messages.insert(1, {"role": "system", "content": f"Conversation so far:\n{brief}"})

    # call OpenAI
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
        )
        raw_text = resp.choices[0].message.content

        try:
            parsed = json.loads(raw_text)
            return RegulAIteAnswer(**parsed)
        except Exception:
            # graceful fallback
            return RegulAIteAnswer(
                summary=raw_text or "No answer generated.",
                per_source={},
                comparative_analysis="",
                recommendation="",
                general_knowledge="",
                gaps_or_next_steps="",
                citations=[],
                follow_up_suggestions=[]
            )

    except Exception as e:
        return RegulAIteAnswer(
            summary=f"Error while generating answer: {str(e)}",
            per_source={},
            comparative_analysis="",
            recommendation="",
            general_knowledge="",
            gaps_or_next_steps="",
            citations=[],
            follow_up_suggestions=[]
        )
