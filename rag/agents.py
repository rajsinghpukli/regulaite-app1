from __future__ import annotations
import os
from typing import List, Dict, Any
from .router import length_directive

DEFAULT_PLAN = """You are RegulAIte, a regulatory RAG assistant for Khaleeji Bank.
Answer ONLY from (a) the attached OpenAI Vector Store (IFRS, AAOIFI, CBB, Internal policies),
and (b) web search when explicitly enabled. If a framework cannot be evidenced, mark it
`not_found`. NEVER invent citations.

Output MUST follow the provided JSON schema exactly (structured output), with these rules:
- Fill the canonical 9 sections (Summary, per_source, Comparative analysis, Recommendation,
  General knowledge, Gaps/Next steps, Citations, AI opinion, Follow-up suggestions).
- Evidence mode: Provide 2–5 short verbatim quotes **per framework** that you mark as `addressed`,
  each with a source citation (filename:page or URL). If <2 quotes for a framework, set status
  to `not_found` and provide a note why.
- Use short quotes (1–3 sentences). No ultra-long blocks.
- Comparative analysis should highlight differences across IFRS vs AAOIFI vs CBB clearly.
- Follow-up suggestions: 4–6 actionable, clickable questions that a Khaleeji CRO might ask next.
"""

def build_system_instruction(
    k_hint: int,
    evidence_mode: bool,
    mode: str,
    org_tone: str = "professional, direct, bank-grade clarity"
) -> str:
    ev = "Evidence mode is ON (2–5 quotes per framework)" if evidence_mode else \
         "Evidence mode is OFF; still cite sources when used."
    size = length_directive(mode)
    return f"""{DEFAULT_PLAN}

House rules:
- Tone: {org_tone}.
- Top-K hint for file search: {k_hint}.
- {ev}
- {size}
- If query is ambiguous, briefly state assumptions and proceed.

Safety:
- If a requested topic concerns confidential policies not present in the vector store, explain that you cannot confirm and suggest providing the official document."""
    
def history_to_brief(history: List[Dict[str, str]], max_pairs: int = 8) -> str:
    """
    Compress last turns into a lightweight running brief.
    """
    turns = history[-(max_pairs*2):]
    lines = []
    for h in turns:
        role = h.get("role")
        content = h.get("content", "").strip()
        if not content:
            continue
        if role == "user":
            lines.append(f"User asked: {content}")
        else:
            lines.append(f"Assistant replied (summary/extract): {content[:600]}")
    return "\n".join(lines[-(max_pairs*2):])
