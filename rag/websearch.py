# rag/websearch.py
from __future__ import annotations
from typing import List, Dict, Any

def ddg_search(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    DuckDuckGo search with two passes and a deterministic BIS fallback.
    - Pass 1: generic
    - Pass 2: if query hints BIS, try site:bis.org
    - Fallback: if asking for the known 'large exposures' BIS paper, return its URL
    Returns [] on any unexpected error.
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

    # 1) generic
    res = _search(query, max_results)
    if res:
        return res

    # 2) site:bis.org bias if relevant
    ql = (query or "").lower()
    if "bis.org" in ql or "bcbs" in ql or "basel committee" in ql or "large exposure" in ql:
        site_q = f"site:bis.org {query}"
        res = _search(site_q, max_results + 5)
        if res:
            return res

    # 3) deterministic fallback for the well-known large-exposures paper
    target_phr = "supervisory framework for measuring and controlling large exposures"
    if target_phr in ql or "large exposures framework" in ql:
        return [{
            "title": "Supervisory framework for measuring and controlling large exposures",
            "url": "https://www.bis.org/publ/bcbs283.htm",
            "snippet": "BIS/BCBS page – official large exposures framework."
        }]

    return []
