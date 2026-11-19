"""The n-gram language model: counting and probability lookup.

One ``NGramModel`` of order ``n`` stores the count tables for *every* order from 1 to
``n``. Each table is built exactly as a standalone model of that order would build it
(its own ``<s>`` padding), so the bigram table inside a trigram model is identical to
the table of a pure bigram model. That gives the application a single trained object
from which unigram, bigram and trigram behaviour can all be read.

Data structure, for each order ``m``::

    tables[m - 1] : dict[context tuple of length m-1] -> Counter[next word]
    totals[m - 1] : dict[context tuple of length m-1] -> int   (sum of that Counter)

``totals`` is defined as the sum of the follower counts, so for every observed context
``sum_w C(context, w) == C(context)`` holds by construction and MLE distributions
normalise exactly.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

from .preprocessing import BOS, EOS, UNK, Vocabulary, pad_sentence
from .smoothing import probability as estimate_probability
from .smoothing import validate_k

Context = tuple[str, ...]
MODEL_FORMAT = "wordflow-ngram-v1"
_EMPTY: Counter[str] = Counter()


class NGramModel:
    """Count-based n-gram language model supporting orders ``1..order``."""

    def __init__(
        self, order: int = 3, min_count: int = 1, max_vocab_size: int | None = None
    ) -> None:
        if isinstance(order, bool) or not isinstance(order, int) or order < 1:
            raise ValueError(f"order must be a positive integer, got {order!r}")
        self.order = order
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        self.vocab: Vocabulary = Vocabulary(())
        self.num_sentences = 0
        self.num_words = 0          # word tokens seen in training (no <s> / </s>)
        self._tables: list[dict[Context, Counter[str]]] = [{} for _ in range(order)]
        self._totals: list[dict[Context, int]] = [{} for _ in range(order)]
        self._fitted = False

    # ------------------------------------------------------------------ training

    def fit(self, sentences: Iterable[Sequence[str]]) -> "NGramModel":
        """Count n-grams of every order from tokenised training sentences.

        ``sentences`` is an iterable of token lists (one per sentence, no padding).
        The vocabulary is built from these sentences only; anything rarer than
        ``min_count`` is replaced by ``<unk>`` before counting.
        """
        corpus = [list(sentence) for sentence in sentences if sentence]
        if not corpus:
            raise ValueError("cannot train on an empty corpus: no sentences with tokens")

        self.vocab = Vocabulary.build(corpus, self.min_count, self.max_vocab_size)
        tables: list[defaultdict[Context, Counter[str]]] = [
            defaultdict(Counter) for _ in range(self.order)
        ]
        for sentence in corpus:
            mapped = self.vocab.map_tokens(sentence)
            for m in range(1, self.order + 1):
                for context, word in self.sentence_events(mapped, m):
                    tables[m - 1][context][word] += 1

        self._tables = [dict(table) for table in tables]
        self._totals = [
            {context: sum(followers.values()) for context, followers in table.items()}
            for table in self._tables
        ]
        self.num_sentences = len(corpus)
        self.num_words = sum(len(sentence) for sentence in corpus)
        self._fitted = True
        return self

    @staticmethod
    def sentence_events(tokens: Sequence[str], order: int) -> Iterator[tuple[Context, str]]:
        """Yield every ``(context, next_word)`` event of one sentence for ``order``.

        The sentence is padded on its own, so the first word is conditioned on
        ``<s>`` padding and the last event predicts ``</s>``. Training and
        evaluation both use this generator, which keeps them consistent.
        """
        padded = pad_sentence(tokens, order)
        for i in range(order - 1, len(padded)):
            yield tuple(padded[i - order + 1 : i]), padded[i]

    # ------------------------------------------------------------------- lookups

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def vocab_size(self) -> int:
        """V used by smoothing: known words + ``<unk>`` + ``</s>``."""
        return self.vocab.size

    def _check_context(self, context: Sequence[str]) -> Context:
        context = tuple(context)
        if len(context) >= self.order:
            raise ValueError(
                f"context of length {len(context)} needs an order-{len(context) + 1} "
                f"model, but this model was trained with order {self.order}"
            )
        return context

    def followers(self, context: Sequence[str] = ()) -> Mapping[str, int]:
        """Counter of the words observed after ``context`` (empty if unseen)."""
        context = self._check_context(context)
        return self._tables[len(context)].get(context, _EMPTY)

    def count(self, context: Sequence[str], word: str) -> int:
        """C(context, word): how often ``word`` followed ``context`` in training."""
        return self.followers(context).get(word, 0)

    def context_count(self, context: Sequence[str] = ()) -> int:
        """C(context): how often ``context`` was followed by anything in training."""
        context = self._check_context(context)
        return self._totals[len(context)].get(context, 0)

    def ngram_counts(self, order: int) -> Counter[Context]:
        """All observed n-grams of ``order`` as ``(w1, ..., wn) -> count``."""
        self._check_order(order)
        counts: Counter[Context] = Counter()
        for context, followers in self._tables[order - 1].items():
            for word, count in followers.items():
                counts[context + (word,)] = count
        return counts

    def num_ngrams(self, order: int) -> int:
        """Number of distinct n-grams of ``order``."""
        self._check_order(order)
        return sum(len(followers) for followers in self._tables[order - 1].values())

    def total_ngrams(self, order: int) -> int:
        """Number of n-gram tokens (events) of ``order`` seen in training."""
        self._check_order(order)
        return sum(self._totals[order - 1].values())

    def context_counts(self, order: int) -> dict[Context, int]:
        """``context -> C(context)`` for every observed context of the given order."""
        self._check_order(order)
        return dict(self._totals[order - 1])

    def _check_order(self, order: int) -> None:
        if isinstance(order, bool) or not isinstance(order, int) or not 1 <= order <= self.order:
            raise ValueError(f"order must be between 1 and {self.order}, got {order!r}")

    # -------------------------------------------------------------- probabilities

    def probability(self, word: str, context: Sequence[str] = (), k: float = 0.0) -> float:
        """P(word | context) with MLE (``k = 0``) or add-k smoothing (``k > 0``).

        The order is implied by the context length: an empty context is the unigram
        model, one word of context the bigram model, and so on. ``<s>`` is never a
        predictable outcome, so its probability is always 0.
        """
        if word == BOS:
            return 0.0
        word = self.vocab.lookup(word)
        return estimate_probability(
            self.count(context, word), self.context_count(context), k, self.vocab_size
        )

    def distribution(self, context: Sequence[str] = (), k: float = 0.0) -> dict[str, float]:
        """Full next-word distribution over the ``V`` predictable tokens.

        With ``k = 0`` only seen continuations get mass, and an unseen context
        yields an empty dict (MLE is undefined there). With ``k > 0`` all ``V``
        tokens are returned and the values sum to 1.
        """
        k = validate_k(k)
        followers = self.followers(context)
        total = self.context_count(context)
        if k == 0:
            return {
                word: estimate_probability(count, total, 0.0, self.vocab_size)
                for word, count in followers.items()
            }
        return {
            word: estimate_probability(followers.get(word, 0), total, k, self.vocab_size)
            for word in self.vocab.prediction_tokens
        }

    # ------------------------------------------------------------------ contexts

    def resolve_context(self, history: Sequence[str], order: int) -> Context:
        """Turn a token history into the ``order - 1`` word context the model needs.

        Unknown words become ``<unk>`` and a history that is too short is padded on
        the left with ``<s>``, exactly as during training.
        """
        self._check_order(order)
        if order == 1:
            return ()
        mapped = [BOS] * (order - 1) + self.vocab.map_tokens(history)
        return tuple(mapped[-(order - 1):])

    def backoff_context(self, context: Sequence[str]) -> Context:
        """Longest suffix of ``context`` that was observed in training.

        Drops the oldest word until the remaining context has a non-zero count.
        The empty (unigram) context is always observed in a fitted model, so this
        always terminates with a usable context. Because the choice depends only
        on the history -- never on the word being predicted -- each history still
        maps to exactly one normalised distribution.
        """
        context = self._check_context(context)
        while context and self.context_count(context) == 0:
            context = context[1:]
        return context

    # --------------------------------------------------------------- persistence

    def to_dict(self) -> dict:
        """JSON-serialisable snapshot of the trained model."""
        return {
            "format": MODEL_FORMAT,
            "order": self.order,
            "min_count": self.min_count,
            "max_vocab_size": self.max_vocab_size,
            "num_sentences": self.num_sentences,
            "num_words": self.num_words,
            "vocabulary": list(self.vocab.words),
            "tables": [
                {" ".join(context): dict(followers) for context, followers in table.items()}
                for table in self._tables
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "NGramModel":
        if data.get("format") != MODEL_FORMAT:
            raise ValueError(f"not a WordFlow model file (format={data.get('format')!r})")
        model = cls(data["order"], data.get("min_count", 1), data.get("max_vocab_size"))
        if len(data["tables"]) != model.order:
            raise ValueError("model file is corrupt: table count does not match order")
        model.vocab = Vocabulary(data["vocabulary"])
        model._tables = [
            {
                tuple(key.split(" ")) if key else (): Counter(followers)
                for key, followers in table.items()
            }
            for table in data["tables"]
        ]
        model._totals = [
            {context: sum(followers.values()) for context, followers in table.items()}
            for table in model._tables
        ]
        model.num_sentences = data.get("num_sentences", 0)
        model.num_words = data.get("num_words", 0)
        model._fitted = True
        return model

    def save(self, path: str | Path) -> Path:
        """Write the model to ``path`` as JSON and return the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "NGramModel":
        """Load a model previously written by :meth:`save`."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"no trained model at {path}; run `python main.py train` first")
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def __repr__(self) -> str:
        return (
            f"NGramModel(order={self.order}, vocab_size={self.vocab_size}, "
            f"sentences={self.num_sentences}, words={self.num_words})"
        )


__all__ = ["NGramModel", "Context", "BOS", "EOS", "UNK"]
