from __future__ import annotations
from typing import List, Tuple, Dict, Any

try:
    from duckduckgo_search import DDGS
except Exception:
    DDGS = None

def ddg_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Return a list of dicts: {title, url, snippet}.
    If DDG is unavailable, return [].
    """
    if DDGS is None:
        return []
    rows = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            title = r.get("title") or ""
            url = r.get("href") or r.get("url") or ""
            body = r.get("body") or ""
            if title and url:
                rows.append({"title": title, "url": url, "snippet": body})
    return rows
