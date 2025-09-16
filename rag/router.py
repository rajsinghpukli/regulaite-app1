def classify_intent_and_scope(q: str, mode_hint: str | None = None) -> dict:
    """
    Minimal router.
    - If mode_hint is provided, use it.
    - Else, default to 'policy'.
    Extend later with an LLM router if you like.
    """
    if mode_hint:
        return {"intent": mode_hint, "scope": None}
    return {"intent": "policy", "scope": None}
