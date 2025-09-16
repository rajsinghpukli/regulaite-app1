from pathlib import Path
import os

from .agents import ask_agent
from .router import classify_intent_and_scope

# Keep this for compatibility with your UI references
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
    Main pipeline entrypoint used by app.py.

    - Supports `k` (Top-K) and `evidence_mode`.
    - Will use OPENAI_VECTOR_STORE if provided.
    """
    route = classify_intent_and_scope(q, mode_hint)
    vector_store_id = os.getenv("OPENAI_VECTOR_STORE", "").strip() or None

    # IMPORTANT: pass positional args to avoid mismatches on older servers
    return ask_agent(
        q,
        include_web,
        route.get("intent"),
        k,
        vector_store_id,
        evidence_mode,
    )
