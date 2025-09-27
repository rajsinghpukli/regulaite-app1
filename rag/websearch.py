from __future__ import annotations
from typing import List, Dict

# We keep duckduckgo-search for text. Images can be added later if needed.
try:
    from duckduckgo_search import DDGS
except Exception:
    DDGS = None

def ddg_search(query: str, max_results: int = 8) -> List[Dict[str, str]]:
    """Return a list of dicts: {title, url, snippet}. Empty list if DDG not available."""
    if DDGS is None:
        return []
    rows: List[Dict[str, str]] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            title = r.get("title") or ""
            url = r.get("href") or r.get("url") or ""
            body = r.get("body") or ""
            if title and url:
                rows.append({"title": title, "url": url, "snippet": body})
    return rows
