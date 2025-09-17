from __future__ import annotations
import json, os, time
from typing import List, Dict, Any

# Where chat histories are stored (one JSON per user)
BASE_DIR = os.path.join(os.path.dirname(__file__), "chats")
os.makedirs(BASE_DIR, exist_ok=True)

def _path(user: str) -> str:
    safe = "".join(c for c in user if c.isalnum() or c in ("_", "-"))
    return os.path.join(BASE_DIR, f"{safe or 'default'}.json")

def load_chat(user: str) -> List[Dict[str, Any]]:
    p = _path(user)
    if not os.path.exists(p):
        return []
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_chat(user: str, history: List[Dict[str, Any]]) -> None:
    p = _path(user)
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception:
        # If write fails (e.g., read-only), just ignore
        pass

def append_turn(user: str, role: str, content: str) -> None:
    hist = load_chat(user)
    hist.append({"ts": time.time(), "role": role, "content": content})
    save_chat(user, hist)

def clear_chat(user: str) -> None:
    p = _path(user)
    try:
        if os.path.exists(p):
            os.remove(p)
    except Exception:
        pass
