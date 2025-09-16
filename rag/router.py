from __future__ import annotations

def normalize_mode(mode_hint: str | None) -> str:
    mode = (mode_hint or "auto").strip().lower()
    if mode in {"short", "concise"}:
        return "short"
    if mode in {"long", "detailed"}:
        return "long"
    if mode in {"research", "deep"}:
        return "research"
    return "auto"

def length_directive(mode: str) -> str:
    if mode == "short":
        return "Respond in ~5–8 bullet points with tight phrasing."
    if mode == "long":
        return "Respond as a detailed brief (8–15 bullets + short paragraphs)."
    if mode == "research":
        return "Respond as a structured memo with sections, bullets, and short paragraphs; include context, caveats, and alternatives."
    return "Pick an appropriate level of detail automatically."
