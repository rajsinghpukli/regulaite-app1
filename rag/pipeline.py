from __future__ import annotations
import os, json, re
from typing import Dict, Any, List, Optional
from openai import OpenAI
from pydantic import ValidationError
from .schema import RegulAIteAnswer, DEFAULT_EMPTY
from .agents import build_system_instruction, history_to_brief
from .router import normalize_mode

client = OpenAI()

def _json_schema() -> Dict[str, Any]:
    # Minimal schema mirroring RegulAIteAnswer (for structured outputs)
    return {
        "name": "RegulAIteAnswer",
        "schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "per_source": {
                    "type": "object",
                    "properties": {
                        "IFRS": _per_source_schema(),
                        "AAOIFI": _per_source_schema(),
                        "CBB": _per_source_schema(),
                        "InternalPolicy": _per_source_schema(),
                    },
                    "additionalProperties": False,
                },
                "comparative_analysis": {"type": "string"},
                "recommendation": {"type": "string"},
                "general_knowledge": {"type": "string"},
                "gaps_or_next_steps": {"type": "string"},
                "citations": {"type": "array", "items": {"type": "string"}},
                "ai_opinion": {"type": "string"},
                "follow_up_suggestions": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["summary", "per_source"],
            "additionalProperties": False,
        },
        "strict": True,
    }

def _per_source_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["addressed", "not_found"]},
            "notes": {"type": "string"},
            "quotes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "framework": {"type": "string"},
                        "snippet": {"type": "string"},
                        "citation": {"type": "string"},
                    },
                    "required": ["framework", "snippet"],
                    "additionalProperties": False,
                },
            },
        },
        "additionalProperties": False,
    }

def _attachments(vec_id: Optional[str], k_hint: int) -> Optional[List[Dict[str, Any]]]:
    if not vec_id:
        return None
    return [{
        "vector_store_id": vec_id,
        "file_search": {"max_num_results": int(k_hint)}  # hint to the Responses API
    }]

def _parse_json_loose(text: str) -> Dict[str, Any]:
    # Grab the first top-level JSON object from text (robust to prose around it)
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {}
    raw = m.group(0)
    try:
        return json.loads(raw)
    except Exception:
        # attempt to fix trailing commas, etc.
        raw = re.sub(r",\s*}", "}", raw)
        raw = re.sub(r",\s*]", "]", raw)
        try:
            return json.loads(raw)
        except Exception:
            return {}

def ask(
    query: str,
    *,
    user_id: str,
    history: List[Dict[str, str]],
    k_hint: int = 5,
    evidence_mode: bool = True,
    mode_hint: str | None = "auto",
    web_enabled: bool = False,
    vec_id: Optional[str] = None,
    model: Optional[str] = None,
) -> RegulAIteAnswer:
    """
    Orchestrates a single turn to the OpenAI Responses API using vector store and optional web search.
    Returns a validated RegulAIteAnswer object.
    """
    model = model or os.getenv("RESPONSES_MODEL", "gpt-4.1-mini")
    mode = normalize_mode(mode_hint)
    sys_inst = build_system_instruction(k_hint=k_hint, evidence_mode=evidence_mode, mode=mode)
    convo_brief = history_to_brief(history)

    tools = []
    if web_enabled:
        tools.append({"type": "web_search"})

    attachments = _attachments(vec_id, k_hint)

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": sys_inst},
            {"role": "user", "content": f"Conversation so far (brief):\n{convo_brief}"},
            {"role": "user", "content": query},
        ],
        response_format={"type": "json_schema", "json_schema": _json_schema()},
        tools=tools if tools else None,
        attachments=attachments if attachments else None,
        metadata={"app": "RegulAIte", "user": user_id},
    )

    # Try to extract structured JSON
    text = getattr(response, "output_text", None)
    if not text:
        # Some SDK builds use .output with content parts
        try:
            parts = []
            for block in getattr(response, "output", []):
                for c in getattr(block, "content", []):
                    if getattr(c, "type", None) == "output_text":
                        parts.append(getattr(c, "text", {}).get("value", ""))
            text = "\n".join(parts)
        except Exception:
            text = ""

    data = _parse_json_loose(text or "")
    if not data:
        return DEFAULT_EMPTY

    try:
        return RegulAIteAnswer(**data)
    except ValidationError:
        return DEFAULT_EMPTY
