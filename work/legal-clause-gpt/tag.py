"""Lesson 6: tag a clause from the command line.

    python tag.py "This Agreement shall be governed by the laws of the State of Delaware."
    python tag.py --file clauses.txt          # one clause per line
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import clausegpt  # noqa: F401  (must come before transformers)
from clausegpt.tagger import ClauseTagger

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clause", nargs="*")
    ap.add_argument("--file")
    ap.add_argument("--model", default=str(ROOT / "outputs" / "clause-gpt2"))
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")       # Windows consoles can't print every character
    clauses = [" ".join(args.clause)] if args.clause else []
    if args.file:
        clauses += [l.strip() for l in Path(args.file).read_text(encoding="utf-8").splitlines() if l.strip()]
    if not clauses:
        ap.error("give a clause or --file")
    tagger = ClauseTagger.load(args.model)
    for c, p in zip(clauses, tagger.predict(clauses)):
        alts = ", ".join(f"{t} {pr:.0%}" for t, pr in p.top)
        note = "" if p.valid else f"  (free generation said {p.generated!r})"
        print(f"<{p.label}>  [{alts}]{note}\n  {c[:120]}{'...' if len(c) > 120 else ''}")


if __name__ == "__main__":
    main()
