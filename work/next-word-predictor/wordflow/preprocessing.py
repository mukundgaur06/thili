"""Text preprocessing: normalisation, sentence splitting, tokenisation, vocabulary.

Everything downstream (counting, probabilities, evaluation) consumes the output of this
module, so the rules here are deliberately small and explicit:

* text is lower-cased and typographic apostrophes are normalised;
* sentences are split on ``. ! ?`` (with a short abbreviation guard), on blank lines, and
  optionally on every newline;
* punctuation is *not* a token -- it only marks sentence boundaries;
* each sentence is padded independently with ``<s>`` / ``</s>`` so that no n-gram ever
  spans two sentences;
* words outside the vocabulary are mapped to ``<unk>``.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Sequence

BOS = "<s>"      # start-of-sentence padding, only ever appears in a context
EOS = "</s>"     # end-of-sentence, a real event the model predicts
UNK = "<unk>"    # stands in for every out-of-vocabulary word
SPECIAL_TOKENS = (BOS, EOS, UNK)

LINE_MODES = ("line", "paragraph")

# A word is letters with optional internal apostrophes (don't, o'clock); a number may
# carry internal separators (3.14, 1,000). Neither can contain "<" or ">", so a corpus
# word can never collide with a special token.
_TOKEN_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*|\d+(?:[.,]\d+)*")
# A sentence ends at . ! or ? (possibly repeated, possibly followed by closing quotes or
# brackets) when whitespace follows.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+")
_ABBREVIATIONS = frozenset(
    {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e"}
)
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "`": "'"})


def normalize(text: str) -> str:
    """Lower-case ``text`` and normalise typographic apostrophes to ``'``."""
    return text.translate(_APOSTROPHES).lower()


def tokenize(text: str) -> list[str]:
    """Split one sentence into lower-case word tokens, dropping punctuation.

    >>> tokenize("Don't panic -- it's 3.14, OK?")
    ["don't", 'panic', "it's", '3.14', 'ok']
    """
    return _TOKEN_RE.findall(normalize(text))


def _ends_with_abbreviation(fragment: str) -> bool:
    words = fragment.rstrip().split()
    if not words:
        return False
    last = words[-1].lower().strip("\"'()[]")
    return last.endswith(".") and last[:-1] in _ABBREVIATIONS


def split_sentences(text: str, line_mode: str = "line") -> list[str]:
    """Split raw text into sentence strings.

    ``line_mode="line"`` treats every newline as a sentence boundary (one sentence or
    more per line -- the usual layout for a clean corpus). ``line_mode="paragraph"``
    joins the lines of a paragraph first and treats only blank lines as boundaries,
    which is what hard-wrapped prose (e.g. Project Gutenberg files) needs.
    """
    if line_mode not in LINE_MODES:
        raise ValueError(f"line_mode must be one of {LINE_MODES}, got {line_mode!r}")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if line_mode == "line":
        blocks = text.split("\n")
    else:
        blocks = [" ".join(block.split("\n")) for block in re.split(r"\n\s*\n", text)]

    sentences: list[str] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        pending = ""
        for piece in _SENTENCE_END_RE.split(block):
            candidate = f"{pending} {piece}".strip() if pending else piece.strip()
            if _ends_with_abbreviation(candidate):
                pending = candidate          # "Dr." is not the end of a sentence
                continue
            pending = ""
            if candidate:
                sentences.append(candidate)
        if pending:
            sentences.append(pending)
    return sentences


def tokenize_sentences(text: str, line_mode: str = "line") -> list[list[str]]:
    """Split ``text`` into sentences and tokenise each; empty sentences are dropped."""
    tokenized = (tokenize(sentence) for sentence in split_sentences(text, line_mode))
    return [tokens for tokens in tokenized if tokens]


def ends_sentence(text: str) -> bool:
    """True when ``text`` ends with sentence-final punctuation (so a new sentence starts)."""
    return bool(re.search(r"[.!?][\"')\]]*\s*$", text))


def context_tokens(text: str) -> list[str]:
    """Tokens of the sentence the user is currently typing.

    Only the last, unfinished sentence matters for predicting the next word. If the
    text is empty or already ends with ``. ! ?`` the next word starts a new sentence,
    so the context is empty (it will be padded with ``<s>``).
    """
    if not text.strip() or ends_sentence(text):
        return []
    sentences = tokenize_sentences(text, line_mode="line")
    return sentences[-1] if sentences else []


def pad_sentence(tokens: Sequence[str], order: int) -> list[str]:
    """Pad a sentence for an ``order``-gram model: ``order-1`` ``<s>`` and one ``</s>``.

    A unigram model (order 1) gets no start padding but still predicts ``</s>``.
    """
    if order < 1:
        raise ValueError("order must be >= 1")
    return [BOS] * (order - 1) + list(tokens) + [EOS]


class Vocabulary:
    """The closed set of words the model knows, built from *training* tokens only.

    ``min_count`` drops rare words (they become ``<unk>``) and ``max_size`` keeps only
    the most frequent words. ``size`` is the number of outcomes the model can predict:
    every known word plus ``<unk>`` and ``</s>``. ``<s>`` is excluded because it is
    never predicted -- this is the ``V`` used by add-k smoothing.
    """

    def __init__(self, words: Iterable[str]) -> None:
        self.words: tuple[str, ...] = tuple(sorted(set(words) - set(SPECIAL_TOKENS)))
        self._known = frozenset(self.words)
        #: every token that can follow a context, in a fixed deterministic order
        self.prediction_tokens: tuple[str, ...] = self.words + (UNK, EOS)

    @classmethod
    def build(
        cls,
        sentences: Iterable[Sequence[str]],
        min_count: int = 1,
        max_size: int | None = None,
    ) -> "Vocabulary":
        if min_count < 1:
            raise ValueError("min_count must be >= 1")
        if max_size is not None and max_size < 1:
            raise ValueError("max_size must be >= 1 when given")
        counts: Counter[str] = Counter()
        for sentence in sentences:
            counts.update(sentence)
        kept = [word for word, count in counts.items() if count >= min_count]
        if max_size is not None:
            kept.sort(key=lambda word: (-counts[word], word))
            kept = kept[:max_size]
        return cls(kept)

    @property
    def size(self) -> int:
        """V: number of predictable outcomes (known words + ``<unk>`` + ``</s>``)."""
        return len(self.prediction_tokens)

    def __contains__(self, token: object) -> bool:
        return token in self._known

    def __len__(self) -> int:
        return len(self.words)

    def lookup(self, token: str) -> str:
        """Return ``token`` if it is known (or special), otherwise ``<unk>``."""
        if token in self._known or token in SPECIAL_TOKENS:
            return token
        return UNK

    def map_tokens(self, tokens: Iterable[str]) -> list[str]:
        return [self.lookup(token) for token in tokens]
