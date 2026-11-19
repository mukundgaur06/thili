"""Shared deterministic toy data for the test-suite."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wordflow import NGramModel, tokenize_sentences  # noqa: E402

TOY_TEXT = "I love Python\nI love machine learning\nI love AI"


def toy_sentences() -> list[list[str]]:
    return tokenize_sentences(TOY_TEXT)


def toy_model(order: int = 3) -> NGramModel:
    return NGramModel(order=order).fit(toy_sentences())
