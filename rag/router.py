def classify_intent_and_scope(q: str, mode_hint: str | None = None) -> dict:
    """
    Minimal router: use mode_hint if given, else default to 'policy'.
    Replace later with an LLM/heuristic classifier if needed.
    """
    if mode_hint:
        return {"intent": mode_hint, "scope": None}
    return {"intent": "policy", "scope": None}
