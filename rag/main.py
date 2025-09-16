# Optional tiny CLI runner for quick sanity tests (local only)
from __future__ import annotations
import os, json
from rag.pipeline import ask

if __name__ == "__main__":
    q = input("Ask: ").strip()
    ans = ask(
        q,
        user_id="dev",
        history=[],
        k_hint=6,
        evidence_mode=True,
        mode_hint="research",
        web_enabled=False,
        vec_id=os.getenv("OPENAI_VECTOR_STORE_ID"),
        model=os.getenv("RESPONSES_MODEL", "gpt-4.1-mini"),
    )
    print(json.dumps(ans.model_dump(), indent=2, ensure_ascii=False))
