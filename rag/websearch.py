# rag/websearch.py
from __future__ import annotations
from typing import List, Dict, Any

def ddg_search(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Safe wrapper around duckduckgo_search. Returns [] on any error so the app never crashes.
    """
    try:
        from duckduckgo_search import DDGS  # lazy import
        out: List[Dict[str, Any]] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results, safesearch="moderate"):
                out.append({
                    "title": r.get("title") or "",
                    "url": r.get("href") or r.get("url") or "",
                    "snippet": (r.get("body") or r.get("snippet") or "")[:400],
                })
        return out
    except Exception:
        return []  # never raise
