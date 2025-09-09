# rag/pipeline.py
# minimal pipeline shim for the app
# exposes constants expected by app.py:
#   - PERSIST_DIR (string path)
#   - TOP_K_DEFAULT (int)
# and keeps the existing ask(...) bridge function.

from pathlib import Path
from typing import Any, Dict, Optional

# relative imports used by your repo
from .agents import ask_agent
from .router import classify_intent_and_scope

# ---------------------------------------------------------------------------
# constants expected by app.py (exported names)
# PERSIST_DIR: location where the app expects to persist vector DB / caches
# TOP_K_DEFAULT: default "top k" for retrievals
PERSIST_DIR = str(Path(__file__).parent / "persist")  # string path for compatibility
TOP_K_DEFAULT = 5
# ---------------------------------------------------------------------------


def ask(q: str, include_web: bool = False, mode_hint: Optional[str] = None) -> Dict[str, Any]:
    """
    Bridge function used by the web app.

    Parameters
    ----------
    q : str
        The user question text.
    include_web : bool
        Whether to include web search.
    mode_hint : Optional[str]
        Optional hint about which agent/mode to use.

    Returns
    -------
    dict
        Whatever ask_agent returns (keeps original behavior).
    """
    route = classify_intent_and_scope(q, mode_hint)
    resp = ask_agent(q, include_web=include_web, mode=route.get("intent"))
    return resp
