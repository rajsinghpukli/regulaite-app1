# Optional dev/CLI helper; not used by the Streamlit app.
from .pipeline import ask

def cli():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("question", help="Your question")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--mode", default=None)
    p.add_argument("--web", action="store_true")
    p.add_argument("--evidence", action="store_true")
    args = p.parse_args()

    print(
        ask(
            args.question,
            include_web=args.web,
            mode_hint=args.mode,
            k=args.k,
            evidence_mode=args.evidence,
        )
    )

if __name__ == "__main__":
    cli()
