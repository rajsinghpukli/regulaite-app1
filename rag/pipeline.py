# placeholder pipeline.py
from .agents import ask_agent
from .router import classify_intent_and_scope

def ask(q, include_web=False, mode_hint=None):
    route = classify_intent_and_scope(q, mode_hint)
    resp = ask_agent(q, include_web=include_web, mode=route.get("intent"))
    return resp
