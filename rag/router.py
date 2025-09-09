# rag/router.py
from typing import Dict, Any, Optional

def classify_intent_and_scope(query: str, mode_hint: Optional[str] = None) -> Dict[str, Any]:
    q = (query or "").lower()
    if mode_hint in {"regulatory","research","quick","mixed"}:
        intent = mode_hint
    else:
        if any(k in q for k in ["ifrs","aaoifi","cbb","rulebook","standard","fas "]):
            intent = "regulatory"
        elif len(q) < 80:
            intent = "quick"
        elif any(k in q for k in ["compare","vs","difference","differences"]):
            intent = "mixed"
        else:
            intent = "research"
    return {"intent": intent, "needs_web": intent in {"research","mixed"}, "frameworks": []}
