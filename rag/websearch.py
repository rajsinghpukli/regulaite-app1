# rag/websearch.py
from __future__ import annotations
from typing import List, Dict, Any

def ddg_search(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Safe wrapper around duckduckgo_search. Returns [] on any error so the app never crashes.
    Tries twice: generic; then, if query hints 'bis.org', force a site:bis.org search.
    """
    def _search(q: str, n: int) -> List[Dict[str, Any]]:
        try:
            from duckduckgo_search import DDGS  # lazy import
            out: List[Dict[str, Any]] = []
            with DDGS() as ddgs:
                for r in ddgs.text(q, max_results=n, safesearch="moderate", region="wt-wt"):
                    out.append({
                        "title": r.get("title") or "",
                        "url": r.get("href") or r.get("url") or "",
                        "snippet": (r.get("body") or r.get("snippet") or "")[:400],
                    })
            return out
        except Exception:
            return []

    # 1st attempt: as-is
    res = _search(query, max_results)
    if res:
        return res

    # 2nd attempt: if BIS is implied, constrain to site:bis.org
    ql = (query or "").lower()
    if "bis.org" in ql or "bis " in ql or "bcbs" in ql:
        site_q = f"site:bis.org {query}"
        res = _search(site_q, max_results + 5)
        if res:
            return res

    return []
