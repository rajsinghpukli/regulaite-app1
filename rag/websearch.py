from __future__ import annotations
from typing import List, Tuple
try:
    from duckduckgo_search import DDGS
except Exception:
    DDGS = None

def ddg_search(query: str, max_results: int = 5) -> List[Tuple[str, str]]:
    """
    Returns list of (title, url). If ddg unavailable, returns [].
    """
    if DDGS is None:
        return []
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            title = r.get("title") or r.get("body") or ""
            url = r.get("href") or r.get("url") or ""
            if title and url:
                results.append((title, url))
    return results
