import os
from typing import Optional

try:
    # OpenAI >= 1.x
    from openai import OpenAI
except Exception:
    OpenAI = None

DEFAULT_RESPONSES_MODEL = os.getenv("RESPONSES_MODEL", "gpt-4o-mini")


def _make_client() -> Optional["OpenAI"]:
    if not OpenAI:
        return None
    try:
        return OpenAI()
    except Exception:
        return None


def _to_text_from_responses(resp) -> str:
    """
    Best effort extraction for the Responses API across versions.
    """
    # Newer SDKs expose output_text
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text

    # Some versions embed 'output' items
    if hasattr(resp, "output") and resp.output:
        parts = []
        for item in resp.output:
            piece = getattr(item, "content", None) or getattr(item, "text", None)
            if isinstance(piece, str):
                parts.append(piece)
        if parts:
            return "\n".join(parts)

    # Generic fallback
    return "Sorry, I couldn't produce a response."


def ask_agent(
    q: str,
    include_web: bool,
    mode: Optional[str],
    k: int = 5,
    vector_store_id: Optional[str] = None,
    evidence_mode: bool = False,
) -> str:
    """
    Core agent logic:
    - Prefer the Responses API with file_search to use your OpenAI vector store.
    - Falls back to Chat Completions if Responses is not available.
    """
    client = _make_client()
    if not client:
        return "OpenAI SDK not available. Please ensure the `openai` package is installed."

    # Compose system + user instructions
    sys = (
        "You are a precise financial policy assistant. "
        "Answer concisely and cite from retrieved sources when available. "
        "If the user enabled evidence mode, include 2–5 short quotes with source hints."
    )

    mode_note = f"(mode: {mode})" if mode else "(mode: auto)"
    web_note = "with web search allowed" if include_web else "without web search"
    user = f"{mode_note} {web_note}.\n\nUser question:\n{q}\n"

    if evidence_mode:
        user += "\nReturn 2–5 short, relevant quotes or references (file name/page) if possible."

    # Try the Responses API with file_search tool
    tools = []
    tool_resources = None
    if vector_store_id:
        tools = [{"type": "file_search"}]
        # Acceptable across API variants; ignored if unsupported
        tool_resources = {
            "file_search": {
                "vector_store_ids": [vector_store_id],
                "max_num_results": int(k),
                "ranking_options": {"max_results": int(k)},
            }
        }

    try:
        resp = client.responses.create(
            model=DEFAULT_RESPONSES_MODEL,
            input=user,
            tools=tools or None,
            tool_resources=tool_resources or None,
            # Some SDKs support 'system' in responses; harmless if ignored by others
            system=sys,
        )
        return _to_text_from_responses(resp)
    except Exception:
        # Graceful fallback to Chat Completions (no file_search)
        try:
            chat = client.chat.completions.create(
                model=DEFAULT_RESPONSES_MODEL,
                messages=[
                    {"role": "system", "content": sys},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
            )
            return chat.choices[0].message.content
        except Exception as e2:
            return f"Model error: {e2}"
