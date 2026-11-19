"""Application layer: owns the one trained model and answers every dashboard query.

Training (:func:`train_state`) produces an immutable :class:`TrainedState`. The
:class:`Workspace` holds exactly one such state at a time -- the single source of
truth -- and swaps it atomically when the user retrains. Every query method reads
from that state, so the interface can never show numbers from a stale model.
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .corpus import Corpus, build_corpus, corpus_from_file, corpus_from_upload, split_train_test
from .evaluation import compare_models, evaluate, score_sentence
from .generation import TextGenerator
from .ngram_model import NGramModel
from .predictor import NextWordPredictor, format_context, order_name
from .preprocessing import BOS, EOS, SPECIAL_TOKENS, UNK, context_tokens, pad_sentence, tokenize
from .smoothing import add_k_probability, mle_probability, validate_k

PROJECT_DIR = Path(__file__).resolve().parent.parent
SAMPLE_CORPUS = PROJECT_DIR / "corpus" / "corpus.txt"
MODEL_PATH = PROJECT_DIR / "models" / "wordflow_model.json"
META_PATH = PROJECT_DIR / "models" / "wordflow_meta.json"

SWEEP_KS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
TOY_CORPUS = "I love Python\nI love machine learning\nI love AI"
TOY_TEST_SENTENCE = "I love machine learning"
MAX_ORDER_LIMIT = 5


@dataclass(frozen=True)
class TrainingConfig:
    """Everything that determines a trained model besides the corpus text."""

    max_order: int = 3
    min_count: int = 1
    test_fraction: float = 0.2
    seed: int = 42
    max_vocab_size: int | None = None

    def validate(self) -> "TrainingConfig":
        if isinstance(self.max_order, bool) or not isinstance(self.max_order, int):
            raise ValueError("max_order must be an integer")
        if not 1 <= self.max_order <= MAX_ORDER_LIMIT:
            raise ValueError(f"max_order must be between 1 and {MAX_ORDER_LIMIT}")
        if isinstance(self.min_count, bool) or not isinstance(self.min_count, int) or self.min_count < 1:
            raise ValueError("min_count must be an integer >= 1")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if not isinstance(self.test_fraction, (int, float)) or not 0 <= self.test_fraction <= 0.9:
            raise ValueError("test_fraction must be between 0 and 0.9")
        return self


@dataclass(frozen=True)
class TrainedState:
    """An immutable snapshot: corpus, split, configuration and the fitted model."""

    version: int
    corpus: Corpus
    config: TrainingConfig
    train_sentences: tuple[str, ...]
    test_sentences: tuple[str, ...]
    train_tokens: tuple[tuple[str, ...], ...]
    test_tokens: tuple[tuple[str, ...], ...]
    model: NGramModel
    trained_at: str


def train_state(corpus: Corpus, config: TrainingConfig, version: int = 1) -> TrainedState:
    """Split the corpus and fit a model on the training side only."""
    config.validate()
    train, test = split_train_test(corpus.sentences, config.test_fraction, config.seed)
    train_tokens = tuple(tuple(tokenize(sentence)) for sentence in train)
    test_tokens = tuple(tuple(tokenize(sentence)) for sentence in test)
    model = NGramModel(config.max_order, config.min_count, config.max_vocab_size)
    model.fit(train_tokens)                      # the test sentences are never counted
    return TrainedState(
        version=version,
        corpus=corpus,
        config=config,
        train_sentences=tuple(train),
        test_sentences=tuple(test),
        train_tokens=train_tokens,
        test_tokens=test_tokens,
        model=model,
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def parse_context(text: str) -> list[str]:
    """Parse a context typed by the user, keeping literal ``<s>`` / ``<unk>`` tokens."""
    tokens: list[str] = []
    for chunk in text.split():
        if chunk.lower() in (BOS, UNK):
            tokens.append(chunk.lower())
        else:
            tokens.extend(tokenize(chunk))
    return tokens


class Workspace:
    """Holds the current trained state and serves JSON-ready answers from it."""

    def __init__(self, model_path: Path | None = MODEL_PATH, meta_path: Path | None = META_PATH) -> None:
        self._lock = threading.Lock()
        self._state: TrainedState | None = None
        self._version = 0
        self._eval_cache: dict[tuple[int, float], dict] = {}
        self._model_path = model_path
        self._meta_path = meta_path

    # ------------------------------------------------------------------ training

    @property
    def state(self) -> TrainedState:
        state = self._state
        if state is None:
            raise RuntimeError("no model has been trained yet")
        return state

    def train(self, corpus: Corpus, config: TrainingConfig | None = None) -> TrainedState:
        """Train on ``corpus`` and make the result the active model."""
        config = (config or TrainingConfig()).validate()
        with self._lock:
            state = train_state(corpus, config, self._version + 1)
            self._version = state.version
            self._state = state
            self._eval_cache.clear()
            self._persist(state)
        return state

    def train_sample(self, config: TrainingConfig | None = None) -> TrainedState:
        return self.train(corpus_from_file(SAMPLE_CORPUS, source="sample"), config)

    def train_upload(
        self, filename: str, data: bytes, config: TrainingConfig | None = None, line_mode: str = "auto"
    ) -> TrainedState:
        return self.train(corpus_from_upload(filename, data, line_mode), config)

    def train_text(
        self, name: str, text: str, config: TrainingConfig | None = None, line_mode: str = "auto"
    ) -> TrainedState:
        return self.train(build_corpus(name, text, "text", line_mode), config)

    def _persist(self, state: TrainedState) -> None:
        """Save the model artefact so it can be reused for inference without retraining."""
        if self._model_path is None:
            return
        try:
            state.model.save(self._model_path)
            if self._meta_path is not None:
                meta = {
                    "corpus": state.corpus.name,
                    "source": state.corpus.source,
                    "config": asdict(state.config),
                    "trained_at": state.trained_at,
                    "train_sentences": len(state.train_sentences),
                    "test_sentences": len(state.test_sentences),
                }
                self._meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except OSError:
            pass          # a read-only folder must not stop the in-memory model from working

    # ------------------------------------------------------------------ overview

    def overview(self) -> dict:
        state = self.state
        model = state.model
        unigrams = model.followers(())
        top_words = sorted(
            ((w, c) for w, c in unigrams.items() if w not in SPECIAL_TOKENS),
            key=lambda item: (-item[1], item[0]),
        )[:12]
        return {
            "version": state.version,
            "trained_at": state.trained_at,
            "corpus": {
                "name": state.corpus.name,
                "source": state.corpus.source,
                "line_mode": state.corpus.line_mode,
                "sentences": len(state.corpus.sentences),
                "train_sentences": len(state.train_sentences),
                "test_sentences": len(state.test_sentences),
                "train_words": sum(len(s) for s in state.train_tokens),
                "test_words": sum(len(s) for s in state.test_tokens),
                "preview": list(state.train_sentences[:6]),
            },
            "config": asdict(state.config),
            "model": {
                "max_order": model.order,
                "vocab_words": len(model.vocab),
                "vocab_size": model.vocab_size,
                "ngrams": [
                    {
                        "order": order,
                        "name": order_name(order),
                        "distinct": model.num_ngrams(order),
                        "total": model.total_ngrams(order),
                    }
                    for order in range(1, model.order + 1)
                ],
                "top_words": [{"word": w, "count": c} for w, c in top_words],
            },
            "special_tokens": {"bos": BOS, "eos": EOS, "unk": UNK},
        }

    # ----------------------------------------------------------------- inference

    def predict(self, text: str, order: int, top_k: int, smoothing: bool, k: float) -> dict:
        state = self.state
        result = NextWordPredictor(state.model).suggest_next(
            text, top_k=top_k, smoothing=smoothing, k=k, order=order
        )
        return {"version": state.version, **result.to_dict()}

    def generate(
        self,
        seed_text: str,
        order: int,
        max_words: int,
        temperature: float,
        seed: int | None,
        k: float = 0.0,
        stop_at_eos: bool = True,
    ) -> dict:
        state = self.state
        result = TextGenerator(state.model).generate_text(
            seed_text,
            max_words=max_words,
            temperature=temperature,
            order=order,
            seed=seed,
            k=k,
            stop_at_eos=stop_at_eos,
        )
        return {"version": state.version, **result.to_dict()}

    def score(self, text: str, order: int, k: float = 0.0, backoff: bool = False) -> dict:
        """Per-token probabilities of a typed sentence, for the 'surprise' heatmap.

        Scores the sentence being typed (the last unfinished one) word by word. The
        final ``</s>`` event is left out because the user has not finished the
        sentence yet. Summary metrics use the same formulas as the evaluator.
        """
        state = self.state
        model = state.model
        k = validate_k(k)
        tokens = context_tokens(text)
        if not tokens:
            return {"version": state.version, "order": order, "k": k, "tokens": [],
                    "cross_entropy": None, "perplexity": None, "zero_prob_tokens": 0}
        rows = score_sentence(model, tokens, order, k, backoff)[:-1]       # drop </s>
        zeros = sum(1 for *_rest, p in rows if p <= 0)
        nll = sum(-math.log(p) for *_rest, p in rows if p > 0)
        h = None if zeros else nll / len(rows)
        return {
            "version": state.version,
            "order": order,
            "k": k,
            "tokens": [
                {
                    "word": original,
                    "mapped": word,
                    "context": list(context),
                    "probability": p,
                    "surprisal_bits": -math.log2(p) if p > 0 else None,
                    "count": model.count(context, word) if len(context) < model.order else 0,
                }
                for original, (context, word, p) in zip(tokens, rows)
            ],
            "cross_entropy": h,
            "perplexity": math.exp(h) if h is not None else None,
            "zero_prob_tokens": zeros,
        }

    # ------------------------------------------------------------------ explorer

    def ngrams(
        self,
        order: int,
        query: str = "",
        limit: int = 50,
        min_count: int = 1,
        include_special: bool = True,
    ) -> dict:
        """Observed n-grams of ``order``, filtered by a substring query and a count floor."""
        model = self.state.model
        counts = model.ngram_counts(order)
        needle = " ".join(query.lower().split())
        rows = []
        for ngram, count in counts.items():
            if count < min_count:
                continue
            if not include_special and any(token in SPECIAL_TOKENS for token in ngram):
                continue
            if needle and needle not in " ".join(ngram):
                continue
            rows.append((ngram, count))
        rows.sort(key=lambda item: (-item[1], item[0]))
        limit = max(1, min(limit, 500))
        return {
            "version": self.state.version,
            "order": order,
            "distinct": model.num_ngrams(order),
            "total": model.total_ngrams(order),
            "matches": len(rows),
            "rows": [
                {
                    "ngram": list(ngram),
                    "count": count,
                    "context_count": model.context_count(ngram[:-1]),
                    "mle": mle_probability(count, model.context_count(ngram[:-1])),
                }
                for ngram, count in rows[:limit]
            ],
        }

    def distribution(self, context_text: str, order: int, k: float = 1.0, limit: int = 15) -> dict:
        """MLE and add-k next-word distributions for one context, side by side."""
        model = self.state.model
        k = validate_k(k)
        if k == 0:
            raise ValueError("k must be > 0 to compare against a smoothed distribution")
        typed = parse_context(context_text)
        unknown = sorted({t for t in typed if t not in model.vocab and t not in SPECIAL_TOKENS})
        context = model.resolve_context(typed, order)
        followers = model.followers(context)
        total = model.context_count(context)
        vocab_size = model.vocab_size
        ranked = sorted(followers.items(), key=lambda item: (-item[1], item[0]))
        unseen_types = vocab_size - len(followers)
        unseen_each = add_k_probability(0, total, k, vocab_size)
        smoothed = model.distribution(context, k)
        popular = sorted(
            model.context_counts(order).items(),
            key=lambda item: (-item[1], item[0]),
        )[:14]
        limit = max(1, min(limit, 200))
        return {
            "version": self.state.version,
            "order": order,
            "k": k,
            "typed_tokens": typed,
            "unknown_words": unknown,
            "context": list(context),
            "context_count": total,
            "seen": total > 0,
            "distinct_followers": len(followers),
            "vocab_size": vocab_size,
            "rows": [
                {
                    "word": word,
                    "count": count,
                    "mle": mle_probability(count, total),
                    "smoothed": add_k_probability(count, total, k, vocab_size),
                }
                for word, count in ranked[:limit]
            ],
            "rows_truncated": max(0, len(ranked) - limit),
            "unseen": {
                "types": unseen_types,
                "probability_each": unseen_each,
                "total_mass": unseen_types * unseen_each,
            },
            "checks": {
                "mle_sum": sum(mle_probability(c, total) for c in followers.values()),
                "smoothed_sum": sum(smoothed.values()),
            },
            "popular_contexts": [
                {"context": list(c), "count": n} for c, n in popular if order > 1
            ],
        }

    # ---------------------------------------------------------------- evaluation

    def evaluation(self, k: float = 1.0) -> dict:
        """Held-out and training-set metrics for the standard model line-up."""
        state = self.state
        k = validate_k(k)
        if k == 0:
            raise ValueError("k must be > 0 for the smoothed models")
        key = (state.version, k)
        cached = self._eval_cache.get(key)
        if cached is not None:
            return cached
        model = state.model
        has_test = bool(state.test_tokens)
        held_out = [r.to_dict() for r in compare_models(model, state.test_tokens, k)] if has_test else []
        training = [r.to_dict() for r in compare_models(model, state.train_tokens, k)]
        sweep = []
        if has_test:
            for order in range(1, model.order + 1):
                sweep.append(
                    {
                        "order": order,
                        "name": order_name(order),
                        "points": [
                            {"k": kk, "perplexity": evaluate(model, state.test_tokens, order, kk).perplexity}
                            for kk in SWEEP_KS
                        ],
                    }
                )
        result = {
            "version": state.version,
            "has_test": has_test,
            "configuration": {
                "k": k,
                "corpus": state.corpus.name,
                "test_fraction": state.config.test_fraction,
                "seed": state.config.seed,
                "min_count": state.config.min_count,
                "train_sentences": len(state.train_sentences),
                "test_sentences": len(state.test_sentences),
                "train_words": sum(len(s) for s in state.train_tokens),
                "test_words": sum(len(s) for s in state.test_tokens),
                "vocab_size": model.vocab_size,
                "log_base": "e (nats); bits = nats / ln 2",
                "predicted_tokens": "every word plus one </s> per sentence; <s> is never predicted",
                "vocabulary": "built from the training split only; unknown test words score as <unk>",
            },
            "held_out": held_out,
            "training": training,
            "sweep": sweep,
        }
        self._eval_cache[key] = result
        return result

    # --------------------------------------------------------------------- learn

    def learn(
        self,
        corpus_text: str = TOY_CORPUS,
        test_sentence: str = TOY_TEST_SENTENCE,
        context_word: str = "love",
        target_word: str = "machine",
        k: float = 1.0,
    ) -> dict:
        """Worked numerical examples computed by the real engine on a toy corpus.

        A throw-away model is trained on ``corpus_text``; the active model is not
        touched. Every number in the response comes from the same functions the
        rest of the application uses.
        """
        k = validate_k(k)
        if k == 0:
            raise ValueError("k must be > 0")
        if len(corpus_text) > 20_000:
            raise ValueError("the toy corpus is limited to 20,000 characters")
        corpus = build_corpus("toy", corpus_text, "toy", "line")
        sentences = [tokenize(s) for s in corpus.sentences]
        model = NGramModel(order=3).fit(sentences)
        vocab_size = model.vocab_size

        def top(order: int, limit: int = 12) -> list[dict]:
            counts = sorted(model.ngram_counts(order).items(), key=lambda i: (-i[1], i[0]))
            return [{"ngram": list(g), "count": c} for g, c in counts[:limit]]

        ctx = tokenize(context_word)[-1:] or ["love"]
        tgt = (tokenize(target_word) or ["machine"])[0]
        bigram_context = model.resolve_context(ctx, 2)
        c_pair = model.count(bigram_context, model.vocab.lookup(tgt))
        c_ctx = model.context_count(bigram_context)
        followers = sorted(model.followers(bigram_context).items(), key=lambda i: (-i[1], i[0]))

        test_tokens = tokenize(test_sentence)
        scored = {}
        if test_tokens:
            for label, order, kk in (("bigram_mle", 2, 0.0), ("bigram_smoothed", 2, k)):
                rows = score_sentence(model, test_tokens, order, kk)
                result = evaluate(model, [test_tokens], order, kk).to_dict()
                scored[label] = {
                    "rows": [
                        {
                            "context": list(c),
                            "word": w,
                            "probability": p,
                            "neg_log": -math.log(p) if p > 0 else None,
                        }
                        for c, w, p in rows
                    ],
                    "result": result,
                }

        unigram_total = model.context_count(())
        return {
            "k": k,
            "sentences": [" ".join(s) for s in sentences],
            "padded": {
                "bigram": [pad_sentence(s, 2) for s in sentences[:4]],
                "trigram": [pad_sentence(s, 3) for s in sentences[:4]],
            },
            "vocab_words": list(model.vocab.words),
            "vocab_size": vocab_size,
            "unigram_total": unigram_total,
            "unigrams": top(1),
            "bigrams": top(2),
            "trigrams": top(3),
            "focus": {
                "context": list(bigram_context),
                "context_label": format_context(bigram_context),
                "word": tgt,
                "known": tgt in model.vocab,
                "pair_count": c_pair,
                "context_count": c_ctx,
                "mle": mle_probability(c_pair, c_ctx),
                "smoothed": add_k_probability(c_pair, c_ctx, k, vocab_size),
                "followers": [
                    {
                        "word": w,
                        "count": c,
                        "mle": mle_probability(c, c_ctx),
                        "smoothed": add_k_probability(c, c_ctx, k, vocab_size),
                    }
                    for w, c in followers
                ],
                "unseen_types": vocab_size - len(followers),
                "unseen_each": add_k_probability(0, c_ctx, k, vocab_size),
                "unigram_count": model.count((), model.vocab.lookup(tgt)),
                "unigram_probability": model.probability(tgt, (), 0.0),
            },
            "test_sentence": " ".join(test_tokens),
            "scored": scored,
        }
