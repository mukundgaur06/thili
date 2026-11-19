"""Build the fantasy story corpus from Project Gutenberg plain-text books.

    python scripts/build_fantasy_corpus.py            # download (if needed) and build
    python scripts/build_fantasy_corpus.py --offline  # rebuild from corpus/fantasy/raw only

Writes ``corpus/fantasy/fantasy.txt``: one paragraph per line, Gutenberg licence text,
tables of contents, chapter headings and illustration markers removed, curly quotes
normalised. All books are public domain in the United States.
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "corpus" / "fantasy" / "raw"
OUT_FILE = ROOT / "corpus" / "fantasy" / "fantasy.txt"
MIRROR = "https://gutenberg.pglaf.org"

BOOKS = {
    55: "The Wonderful Wizard of Oz (L. Frank Baum)",
    54: "The Marvelous Land of Oz (L. Frank Baum)",
    11: "Alice's Adventures in Wonderland (Lewis Carroll)",
    12: "Through the Looking-Glass (Lewis Carroll)",
    16: "Peter Pan (J. M. Barrie)",
    2591: "Grimms' Fairy Tales (Jacob and Wilhelm Grimm)",
    503: "The Blue Fairy Book (Andrew Lang)",
    34339: "The Princess and the Goblin (George MacDonald)",
}

_START_RE = re.compile(r"^\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*$", re.M)
_END_RE = re.compile(r"^\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*$", re.M)
_HEADING_RE = re.compile(r"^(chapter|book|part|contents|illustrations|preface|introduction)\b", re.I)
_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "—": " - ", "–": " - "})


def mirror_url(book_id: int, suffix: str) -> str:
    digits = str(book_id)
    folders = "/".join(digits[:-1]) or "0"
    return f"{MIRROR}/{folders}/{book_id}/{book_id}{suffix}"


def download(book_id: int) -> Path:
    path = RAW_DIR / f"{book_id}.txt"
    if path.exists() and path.stat().st_size > 10_000:
        return path
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("-0.txt", ".txt"):
        try:
            with urllib.request.urlopen(mirror_url(book_id, suffix), timeout=60) as response:
                data = response.read()
        except OSError:
            continue
        if b"START OF" in data:
            path.write_bytes(data)
            return path
    raise SystemExit(f"could not download book {book_id}")


def clean_book(text: str) -> list[str]:
    """Return the story paragraphs of one Gutenberg text, one string per paragraph."""
    text = text.replace("\r\n", "\n").lstrip("﻿")
    start, end = _START_RE.search(text), _END_RE.search(text)
    body = text[start.end() if start else 0 : end.start() if end else len(text)]
    body = re.sub(r"\[[^\]]*\]", " ", body)               # [Illustration], [Footnote ...]
    body = body.translate(_QUOTES)
    paragraphs = []
    for block in re.split(r"\n\s*\n", body):
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        lines = [ln for ln in lines if not _HEADING_RE.match(ln)
                 and not (ln.isupper() and len(ln) < 60)]
        paragraph = " ".join(lines)
        # Prose has sentences; titles, tables of contents and credits do not.
        if len(paragraph) >= 60 and re.search(r"[.!?][\"')]*$", paragraph):
            paragraphs.append(re.sub(r"\s+", " ", paragraph))
    return paragraphs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--offline", action="store_true", help="use files already in corpus/fantasy/raw")
    args = parser.parse_args()

    all_paragraphs: list[str] = []
    for book_id, title in BOOKS.items():
        path = RAW_DIR / f"{book_id}.txt" if args.offline else download(book_id)
        if not path.exists():
            raise SystemExit(f"missing {path}; run without --offline first")
        paragraphs = clean_book(path.read_text(encoding="utf-8", errors="replace"))
        words = sum(len(p.split()) for p in paragraphs)
        print(f"{book_id:>6}  {len(paragraphs):5d} paragraphs  {words:7d} words  {title}")
        all_paragraphs.extend(paragraphs)

    OUT_FILE.write_text("\n".join(all_paragraphs) + "\n", encoding="utf-8")
    print(f"wrote {OUT_FILE} ({OUT_FILE.stat().st_size // 1024} KB, {len(all_paragraphs)} paragraphs)")


if __name__ == "__main__":
    sys.exit(main())
