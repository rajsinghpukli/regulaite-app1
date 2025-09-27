from __future__ import annotations
import json, os, time, secrets, hashlib
from typing import Dict, Any, Tuple

BASE_DIR = os.path.dirname(__file__)
USERS_PATH = os.path.join(BASE_DIR, "users.json")

# 👇 Pilot behaviour: SIGN-UP HIDDEN/DISABLED in UI, but code path kept for future
ALLOW_SIGNUP = False  # override later via env/UI when you open access

PEPPER = os.getenv("PASSWORD_PEPPER", "")  # optional extra secret

# Predefined pilot logins (override via AUTH_USERS env: "user1:pass1,user2:pass2,...")
DEFAULT_FIXED = {
    "amit": "Khaleeji#2025",
    "raj": "RegulAIte#Dev",
    "review": "POC#Access",
}

def _env_fixed() -> Dict[str, str]:
    env = os.getenv("AUTH_USERS", "").strip()
    if not env:
        return DEFAULT_FIXED
    out: Dict[str, str] = {}
    for pair in env.split(","):
        if ":" in pair:
            u, p = pair.split(":", 1)
            out[u.strip()] = p.strip()
    return out or DEFAULT_FIXED

def _load() -> Dict[str, Any]:
    if not os.path.exists(USERS_PATH):
        return {"users": {}}
    try:
        with open(USERS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"users": {}}

def _save(data: Dict[str, Any]) -> None:
    try:
        with open(USERS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _hash(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password + PEPPER).encode("utf-8")).hexdigest()

def ensure_bootstrap_admin() -> None:
    # still supported for later; does nothing in pilot unless BASIC_* set
    data = _load()
    users = data.get("users", {})
    admin_user = os.getenv("BASIC_USER")
    admin_pass = os.getenv("BASIC_PASS")
    if admin_user and admin_pass and admin_user not in users:
        salt = secrets.token_hex(12)
        users[admin_user] = {
            "salt": salt,
            "hash": _hash(admin_pass, salt),
            "created_at": time.time(),
            "role": "admin",
        }
        data["users"] = users
        _save(data)

def username_exists(username: str) -> bool:
    # kept for future; checks dynamic store only
    return username in _load().get("users", {})

def verify_user(username: str, password: str) -> bool:
    # Pilot: fixed accounts take precedence
    fixed = _env_fixed()
    if username in fixed and password == fixed[username]:
        return True
    # Fallback to dynamic store (kept for future)
    users = _load().get("users", {})
    u = users.get(username)
    if not u:
        return False
    return _hash(password, u["salt"]) == u["hash"]

def create_user_if_allowed(username: str, password: str) -> Tuple[bool, str]:
    if not ALLOW_SIGNUP:
        return False, "Sign-up is disabled for the pilot."
    if not username or not password:
        return False, "Username and password required."
    data = _load()
    users = data.get("users", {})
    if username in users:
        return False, "Username already exists."
    salt = secrets.token_hex(12)
    users[username] = {
        "salt": salt,
        "hash": _hash(password, salt),
        "created_at": time.time(),
        "role": "user",
    }
    data["users"] = users
    _save(data)
    return True, "Account created."
