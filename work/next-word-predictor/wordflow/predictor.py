"""Next-word prediction on top of a trained :class:`NGramModel` (inference only)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

from .ngram_model import Context, NGramModel
from .preprocessing import EOS, UNK, context_tokens
from .smoothing import validate_k

#: tokens that are real model outcomes but are not useful as typed suggestions
HIDDEN_TOKENS = (UNK, EOS)


@dataclass(frozen=True)
class Suggestion:
    """One ranked candidate: the word, its probability and its raw training count."""

    word: str
    probability: float
    count: int


@dataclass(frozen=True)
class PredictionResult:
    """Everything needed to show *and explain* a prediction."""

    input_tokens: list[str]           # tokens of the sentence being typed
    unknown_words: list[str]          # input words that are not in the vocabulary
    requested_order: int              # order the caller asked for
    requested_context: list[str]      # context of that order (after <s>/<unk> mapping)
    used_order: int                   # order actually used after fallback
    used_context: list[str]           # context actually conditioned on
    backed_off: bool                  # True when a shorter context had to be used
    context_count: int                # C(used_context)
    distinct_followers: int           # number of different words seen after it
    smoothing: bool
    k: float
    vocab_size: int
    suggestions: list[Suggestion] = field(default_factory=list)
    hidden_mass: dict[str, float] = field(default_factory=dict)   # P(<unk>), P(</s>)
    explanation: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def rank_candidates(distribution: dict[str, float], counts) -> list[tuple[str, float]]:
    """Sort candidates by probability, then by count, then alphabetically.

    The explicit tie-break makes rankings fully deterministic, which matters under
    smoothing where every unseen word shares the same probability.
    """
    return sorted(distribution.items(), key=lambda item: (-item[1], -counts.get(item[0], 0), item[0]))


class NextWordPredictor:
    """Ranks next-word candidates for arbitrary user text."""

    def __init__(self, model: NGramModel) -> None:
        if not model.is_fitted:
            raise ValueError("the model must be trained before it can predict")
        self.model = model

    def suggest_next(
        self,
        text: str,
        top_k: int = 5,
        smoothing: bool = True,
        k: float = 1.0,
        order: int | None = None,
    ) -> PredictionResult:
        """Return the ``top_k`` most probable next words for ``text``.

        Steps: tokenise the sentence being typed, take its last ``order - 1`` words
        as context (unknown words become ``<unk>``, short histories are padded with
        ``<s>``), fall back to a shorter context if that one was never seen in
        training, then rank the candidates by ``P(word | context)``.

        ``smoothing=False`` uses MLE, so only words actually observed after the
        context are returned. ``smoothing=True`` uses add-``k``, which gives every
        vocabulary word a non-zero probability. Probabilities are never
        renormalised after hiding ``<unk>`` / ``</s>``; their mass is reported in
        ``hidden_mass`` instead.
        """
        model = self.model
        order = model.order if order is None else order
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError(f"top_k must be a positive integer, got {top_k!r}")
        k = validate_k(k) if smoothing else 0.0
        if smoothing and k == 0:
            raise ValueError("smoothing requires k > 0")

        tokens = context_tokens(text)
        unknown = sorted({token for token in tokens if token not in model.vocab})
        requested: Context = model.resolve_context(tokens, order)
        used: Context = model.backoff_context(requested)

        followers = model.followers(used)
        distribution = model.distribution(used, k)
        ranked = rank_candidates(distribution, followers)
        suggestions = [
            Suggestion(word, prob, followers.get(word, 0))
            for word, prob in ranked
            if word not in HIDDEN_TOKENS
        ][:top_k]
        hidden = {token: distribution.get(token, 0.0) for token in HIDDEN_TOKENS}

        result = PredictionResult(
            input_tokens=tokens,
            unknown_words=unknown,
            requested_order=order,
            requested_context=list(requested),
            used_order=len(used) + 1,
            used_context=list(used),
            backed_off=used != requested,
            context_count=model.context_count(used),
            distinct_followers=len(followers),
            smoothing=smoothing,
            k=k,
            vocab_size=model.vocab_size,
            suggestions=suggestions,
            hidden_mass=hidden,
        )
        return _with_explanation(result)


ORDER_NAMES = {1: "unigram", 2: "bigram", 3: "trigram"}


def order_name(order: int) -> str:
    return ORDER_NAMES.get(order, f"{order}-gram")


def format_context(context: list[str] | tuple[str, ...]) -> str:
    return " ".join(context) if context else "(no context)"


def _with_explanation(result: PredictionResult) -> PredictionResult:
    """Attach a plain-language account of how the top suggestion was computed."""
    parts: list[str] = []
    if result.unknown_words:
        parts.append(
            "Not in the vocabulary, treated as <unk>: " + ", ".join(result.unknown_words) + "."
        )
    if result.backed_off:
        parts.append(
            f"The {order_name(result.requested_order)} context "
            f"\"{format_context(result.requested_context)}\" never occurs in the training "
            f"data, so the model fell back to the {order_name(result.used_order)} context "
            f"\"{format_context(result.used_context)}\"."
        )
    if result.used_order == 1:
        parts.append(
            f"With no usable context the ranking is plain word frequency over "
            f"{result.context_count} training tokens."
        )
    else:
        parts.append(
            f"The context \"{format_context(result.used_context)}\" occurs "
            f"{result.context_count} time(s) in training, followed by "
            f"{result.distinct_followers} distinct word(s)."
        )
    if result.suggestions:
        top = result.suggestions[0]
        if result.smoothing:
            parts.append(
                f"\"{top.word}\" followed it {top.count} time(s), so "
                f"P = ({top.count} + {result.k:g}) / ({result.context_count} + "
                f"{result.k:g} x {result.vocab_size}) = {top.probability:.4f}."
            )
        else:
            parts.append(
                f"\"{top.word}\" followed it {top.count} time(s), so "
                f"P = {top.count} / {result.context_count} = {top.probability:.4f}."
            )
    else:
        parts.append("No word candidates are available for this context.")
    return replace(result, explanation=" ".join(parts))
