"""Intrinsic evaluation: negative log-likelihood, cross-entropy, perplexity, OOV rate.

For a set of sentences the model predicts every word *and* the end-of-sentence token
``</s>`` (but never ``<s>``). With ``N`` such predicted tokens::

    NLL          = - sum_i ln P(w_i | context_i)
    H            = NLL / N                  (cross-entropy, nats per token)
    perplexity   = exp(H)

If any token receives probability 0 the log-likelihood is ``-inf``; cross-entropy and
perplexity are then reported as infinite rather than being patched with a made-up
floor. That is exactly what smoothing exists to fix.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

from .ngram_model import NGramModel
from .predictor import order_name
from .smoothing import validate_k


@dataclass(frozen=True)
class EvaluationResult:
    label: str
    order: int
    k: float
    backoff: bool
    num_sentences: int
    num_tokens: int              # N: predicted tokens = words + one </s> per sentence
    num_words: int               # word tokens only
    oov_tokens: int              # words not in the training vocabulary
    oov_rate: float              # oov_tokens / num_words
    zero_prob_tokens: int        # predicted tokens that received probability 0
    total_nll: float             # sum of -ln P, math.inf when zero_prob_tokens > 0
    cross_entropy: float         # nats per token
    cross_entropy_bits: float    # bits per token (= nats / ln 2)
    perplexity: float

    @property
    def is_finite(self) -> bool:
        return self.zero_prob_tokens == 0

    def to_dict(self) -> dict:
        """JSON-safe dict: infinities become ``None`` with ``finite = False``."""
        data = asdict(self)
        data["finite"] = self.is_finite
        for key in ("total_nll", "cross_entropy", "cross_entropy_bits", "perplexity"):
            if not math.isfinite(data[key]):
                data[key] = None
        return data


def model_label(order: int, k: float, backoff: bool = False) -> str:
    name = order_name(order).capitalize()
    name += f" + add-{k:g}" if k > 0 else " (MLE)"
    return name + " + backoff" if backoff else name


def token_probability(
    model: NGramModel, context: Sequence[str], word: str, k: float, backoff: bool
) -> float:
    """P(word | context) as used for scoring; optionally via the seen-context fallback."""
    if backoff:
        context = model.backoff_context(context)
    return model.probability(word, context, k)


def score_sentence(
    model: NGramModel,
    sentence: Sequence[str],
    order: int,
    k: float = 0.0,
    backoff: bool = False,
) -> list[tuple[tuple[str, ...], str, float]]:
    """Per-token breakdown of one sentence: ``(context, word, P(word | context))``.

    Out-of-vocabulary words are scored as ``<unk>``; the final entry is ``</s>``.
    """
    mapped = model.vocab.map_tokens(sentence)
    return [
        (context, word, token_probability(model, context, word, k, backoff))
        for context, word in model.sentence_events(mapped, order)
    ]


def evaluate(
    model: NGramModel,
    sentences: Iterable[Sequence[str]],
    order: int | None = None,
    k: float = 0.0,
    backoff: bool = False,
) -> EvaluationResult:
    """Score ``sentences`` (token lists, *not* seen in training) with ``model``.

    The sentences are mapped through the training vocabulary and padded exactly as
    in training. ``order`` selects which of the model's orders to evaluate and
    ``k`` selects MLE (0) or add-k smoothing. ``backoff=True`` additionally lets
    the model use the longest context it has seen instead of an unseen one.
    """
    order = model.order if order is None else order
    k = validate_k(k)
    sentences = [list(sentence) for sentence in sentences if sentence]
    if not sentences:
        raise ValueError("cannot evaluate on an empty set of sentences")

    num_tokens = num_words = oov = zeros = 0
    nll = 0.0
    for sentence in sentences:
        num_words += len(sentence)
        oov += sum(1 for token in sentence if token not in model.vocab)
        for _context, _word, prob in score_sentence(model, sentence, order, k, backoff):
            num_tokens += 1
            if prob > 0:
                nll -= math.log(prob)
            else:
                zeros += 1

    if zeros:
        total_nll = cross_entropy = perplexity = math.inf
    else:
        total_nll = nll
        cross_entropy = nll / num_tokens
        perplexity = math.exp(cross_entropy)
    return EvaluationResult(
        label=model_label(order, k, backoff),
        order=order,
        k=k,
        backoff=backoff,
        num_sentences=len(sentences),
        num_tokens=num_tokens,
        num_words=num_words,
        oov_tokens=oov,
        oov_rate=oov / num_words if num_words else 0.0,
        zero_prob_tokens=zeros,
        total_nll=total_nll,
        cross_entropy=cross_entropy,
        cross_entropy_bits=cross_entropy / math.log(2),
        perplexity=perplexity,
    )


def compare_models(
    model: NGramModel, sentences: Iterable[Sequence[str]], k: float = 1.0
) -> list[EvaluationResult]:
    """Evaluate the standard line-up on the same sentences.

    Unsmoothed models of every order, their add-``k`` counterparts for order >= 2,
    and (as an extension) add-``k`` with the seen-context fallback.
    """
    sentences = [list(sentence) for sentence in sentences if sentence]
    k = validate_k(k)
    if k == 0:
        raise ValueError("compare_models needs k > 0 for the smoothed variants")
    results = [evaluate(model, sentences, order, 0.0) for order in range(1, model.order + 1)]
    results += [evaluate(model, sentences, order, k) for order in range(1, model.order + 1)]
    results += [
        evaluate(model, sentences, order, k, backoff=True) for order in range(2, model.order + 1)
    ]
    return results
