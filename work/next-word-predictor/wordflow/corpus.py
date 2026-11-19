"""Corpus loading, validation, cleaning and the deterministic train/test split."""

from __future__ import annotations

import random
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

from .preprocessing import LINE_MODES, split_sentences, tokenize

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MIN_SENTENCES = 2
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f�]")


class CorpusError(ValueError):
    """Raised when a corpus cannot be used; the message is safe to show to a user."""


@dataclass(frozen=True)
class Corpus:
    """A cleaned corpus, already split into raw sentence strings."""

    name: str
    source: str                    # "sample", "upload" or "file"
    sentences: tuple[str, ...]
    line_mode: str


def decode_bytes(data: bytes) -> str:
    """Decode an uploaded file, rejecting empty and binary content."""
    if not data or not data.strip():
        raise CorpusError("The file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise CorpusError(f"The file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    if b"\x00" in data:
        raise CorpusError("The file looks like binary data, not plain text.")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def clean_text(text: str) -> str:
    """Normalise newlines and strip control / undecodable characters."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("﻿")
    return _CONTROL_CHARS_RE.sub(" ", text)


def detect_line_mode(text: str) -> str:
    """Guess whether newlines separate sentences or merely wrap paragraphs.

    Hard-wrapped prose has long lines of similar length that mostly do *not* end in
    sentence punctuation; a sentence-per-line corpus has the opposite profile.
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) < 4:
        return "line"
    ending = sum(1 for line in lines if re.search(r"[.!?][\"')\]]*$", line)) / len(lines)
    median_length = statistics.median(len(line) for line in lines)
    return "paragraph" if ending < 0.5 and median_length >= 45 else "line"


def build_corpus(name: str, text: str, source: str, line_mode: str = "auto") -> Corpus:
    """Clean ``text`` and split it into sentences, validating that it is usable."""
    text = clean_text(text)
    if not text.strip():
        raise CorpusError("The corpus is empty.")
    if line_mode == "auto":
        line_mode = detect_line_mode(text)
    if line_mode not in LINE_MODES:
        raise CorpusError(f"line_mode must be 'auto' or one of {LINE_MODES}.")
    sentences = tuple(s for s in split_sentences(text, line_mode) if tokenize(s))
    if len(sentences) < MIN_SENTENCES:
        raise CorpusError(
            f"The corpus needs at least {MIN_SENTENCES} sentences with words; "
            f"found {len(sentences)}."
        )
    return Corpus(name=name, source=source, sentences=sentences, line_mode=line_mode)


def corpus_from_upload(filename: str, data: bytes, line_mode: str = "auto") -> Corpus:
    """Validate an uploaded ``.txt`` file and turn it into a :class:`Corpus`."""
    name = Path(filename or "").name
    if not name.lower().endswith(".txt"):
        raise CorpusError("Only plain-text .txt files are supported.")
    return build_corpus(name, decode_bytes(data), "upload", line_mode)


def corpus_from_file(path: str | Path, source: str = "file", line_mode: str = "auto") -> Corpus:
    """Load a corpus from a text file on disk."""
    path = Path(path)
    if not path.is_file():
        raise CorpusError(f"Corpus file not found: {path}")
    return build_corpus(path.name, decode_bytes(path.read_bytes()), source, line_mode)


def split_train_test(
    sentences: list[str] | tuple[str, ...], test_fraction: float = 0.2, seed: int = 42
) -> tuple[list[str], list[str]]:
    """Deterministic sentence-level split with no sentence on both sides.

    Sentences are grouped by their token sequence first, so duplicates (including
    ones that differ only in case or punctuation) always land on the same side --
    a test sentence can never also be a training sentence. The groups are shuffled
    with ``random.Random(seed)``, so the same inputs always give the same split.
    Original corpus order is preserved inside each side.
    """
    if isinstance(test_fraction, bool) or not isinstance(test_fraction, (int, float)):
        raise ValueError("test_fraction must be a number")
    if not 0 <= test_fraction <= 0.9:
        raise ValueError(f"test_fraction must be between 0 and 0.9, got {test_fraction!r}")

    keys = [tuple(tokenize(sentence)) for sentence in sentences]
    unique = list(dict.fromkeys(key for key in keys if key))
    if not unique:
        raise ValueError("cannot split an empty corpus")
    random.Random(seed).shuffle(unique)

    n_test = round(len(unique) * test_fraction)
    if test_fraction > 0 and len(unique) >= 2:
        n_test = max(1, n_test)
    n_test = min(n_test, len(unique) - 1)          # always keep something to train on
    test_keys = set(unique[:n_test])

    train = [s for s, key in zip(sentences, keys) if key and key not in test_keys]
    test = [s for s, key in zip(sentences, keys) if key in test_keys]
    return train, test
