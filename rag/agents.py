import os
from typing import Optional

try:
    # openai>=1.0
    from openai import OpenAI
except Exception:
    OpenAI = None

DEFAULT_RESPONSES_MODEL = os.getenv("RESPONSES_MODEL", "gpt-4o-mini")


def _make_client() -> Optional["OpenAI"]:
    if not OpenAI:
        return None
    return OpenAI()


def _responses_create(client, *, model, prompt, vector_store_id, k, evidence_mode):
    """
    Best-effort call to the Responses API with vector store (file_search tool).
    Falls back to chat.completions if Responses isn't available.
    """
    tools = []
    tool_resources = None

    if vector_store_id:
        tools = [{"type": "file_search"}]
        # Different versions have different shapes. We pass both safely.
        tool_resources = {
            "file_search": {
                "vector_store_ids": [vector_store_id],
                # Some releases accept ranking options; harmless if ignored.
                "max_num_results": int(k),
                "ranking_options": {"max_results": int(k)},
            }
        }

    # Compose system guide
    system = (
        "You are a cautious financial policy assistant. "
        "Answer precisely and concisely. "
        "If file_search results are available, use them. "
        "If the user enabled evidence mode, provide 2–5 short quotes or references."
    )
    if evidence_mode:
        prompt = f"{prompt}\n\nReturn 2–5 short, relevant quotes or references where appropriate."

    try:
        resp = client.responses.create(
            model=model,
            input=prompt,
            tools=tools or None,
            tool_resources=tool_resources or None,
            system=system,  # Some SDK versions support this; ignored if unsupported
        )
        # Extract text from the Responses API result (new SDKs)
        if hasattr(resp, "output_text"):
            return resp.output_text
        # Fallback: try to stitch text pieces
        if hasattr(resp, "output") and resp.output:
            # join any text parts
            parts = []
            for item in resp.output:
                maybe_text = getattr(item, "content", None) or getattr(item, "text", None)
                if isinstance(maybe_text, str):
                    parts.append(maybe_text)
            if parts:
                return "\n".join(parts)
        return "Sorry, I couldn't produce a response."
    except Exception:
        # Fall back to chat.completions (no vector search)
        try:
            chat = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            return chat.choices[0].message.content
        except Exception as e:
            return f"Upstream model error: {e}"


def ask_agent(
    q: str,
    include_web: bool,
    mode: Optional[str],
    k: int = 5,
    vector_store_id: Optional[str] = None,
    evidence_mode: bool = False,
) -> str:
    """
    Core agent logic: uses OpenAI Responses if available (with file_search on your vector store).
    """
    client = _make_client()
    if not client:
        return "OpenAI SDK not found. Please ensure the `openai` package is installed."

    # Simple instructions that blend mode & behavior
    mode_note = f"(mode: {mode})" if mode else "(mode: auto)"
    web_note = "with web search allowed" if include_web else "without web search"
    prompt = (
        f"{mode_note} {web_note}.\n\n"
        f"User question:\n{q}\n"
    )

    model = DEFAULT_RESPONSES_MODEL
    return _responses_create(
        client,
        model=model,
        prompt=prompt,
        vector_store_id=vector_store_id,
        k=max(1, int(k)),
        evidence_mode=evidence_mode,
    )
