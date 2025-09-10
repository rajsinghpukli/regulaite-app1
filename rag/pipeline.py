from pathlib import Path
import os

from .agents import ask_agent
from .router import classify_intent_and_scope

# Keep this for compatibility with older UI / logs
PERSIST_DIR = Path(__file__).parent / "persist"
PERSIST_DIR.mkdir(parents=True, exist_ok=True)


def ask(
    q: str,
    include_web: bool = False,
    mode_hint: str | None = None,
    k: int = 5,
    evidence_mode: bool = False,
) -> str:
    """
    Main entrypoint the UI calls.
    - q: question text
    - include_web: if True, allow web search (not required if you only use vector store)
    - mode_hint: optional override; otherwise we classify
    - k: top-k retrieval size (fixed bug: now supported)
    - evidence_mode: if True, ask for quotes/framework in style
    """
    route = classify_intent_and_scope(q, mode_hint)

    vector_store_id = os.getenv("OPENAI_VECTOR_STORE", "").strip() or None

    return ask_agent(
        q=q,
        include_web=include_web,
        mode=route.get("intent"),
        k=k,
        vector_store_id=vector_store_id,
        evidence_mode=evidence_mode,
    )
