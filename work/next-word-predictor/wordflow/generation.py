"""Random text generation by sampling from the model's next-word distributions."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field

from .ngram_model import NGramModel
from .predictor import rank_candidates
from .preprocessing import EOS, UNK, context_tokens, ends_sentence, tokenize_sentences
from .smoothing import validate_k


@dataclass(frozen=True)
class GenerationStep:
    """One sampling decision, recorded so the UI can replay it."""

    index: int
    context: list[str]            # context actually conditioned on (after fallback)
    order_used: int
    backed_off: bool
    num_candidates: int
    chosen: str
    probability: float            # probability of the chosen token *after* temperature
    top_candidates: list[tuple[str, float]] = field(default_factory=list)


@dataclass(frozen=True)
class GenerationResult:
    seed_text: str
    generated_tokens: list[str]   # sampled tokens, may include </s>
    text: str                     # seed + generated words, rendered as readable text
    steps: list[GenerationStep]
    stop_reason: str              # "end_of_sentence" | "max_words" | "no_candidates"
    order: int
    temperature: float
    k: float
    seed: int | None

    def to_dict(self) -> dict:
        return asdict(self)


def apply_temperature(distribution: dict[str, float], temperature: float) -> dict[str, float]:
    """Reshape a distribution: ``p_i' = p_i^(1/T) / sum_j p_j^(1/T)``.

    ``T = 1`` leaves the model's probabilities unchanged, ``T < 1`` sharpens them
    towards the most likely word and ``T > 1`` flattens them. Temperature is a
    sampling-time extension; it is not part of Maximum Likelihood Estimation.
    Computed in log space so very small ``T`` cannot overflow.
    """
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError(f"temperature must be a number, got {temperature!r}")
    if math.isnan(temperature) or math.isinf(temperature) or temperature <= 0:
        raise ValueError(f"temperature must be finite and > 0, got {temperature!r}")
    positive = {word: p for word, p in distribution.items() if p > 0}
    if not positive:
        return {}
    scaled = {word: math.log(p) / temperature for word, p in positive.items()}
    peak = max(scaled.values())
    weights = {word: math.exp(value - peak) for word, value in scaled.items()}
    total = sum(weights.values())
    return {word: weight / total for word, weight in weights.items()}


def sample(ranked: list[tuple[str, float]], rng: random.Random) -> tuple[str, float]:
    """Inverse-CDF sampling: draw ``u ~ U[0, 1)`` and walk the cumulative sum."""
    if not ranked:
        raise ValueError("cannot sample from an empty distribution")
    total = sum(prob for _, prob in ranked)
    threshold = rng.random() * total
    cumulative = 0.0
    for word, prob in ranked:
        cumulative += prob
        if threshold < cumulative:
            return word, prob
    return ranked[-1]             # guards against floating-point round-off


class TextGenerator:
    """Generates text from a trained model, one sampled word at a time."""

    def __init__(self, model: NGramModel) -> None:
        if not model.is_fitted:
            raise ValueError("the model must be trained before it can generate text")
        self.model = model

    def generate_text(
        self,
        seed_text: str = "",
        max_words: int = 30,
        temperature: float = 1.0,
        order: int | None = None,
        seed: int | None = None,
        k: float = 0.0,
        stop_at_eos: bool = True,
    ) -> GenerationResult:
        """Continue ``seed_text`` by sampling up to ``max_words`` words.

        Each step conditions on the last ``order - 1`` tokens, falls back to a
        shorter context if that one is unseen, applies ``temperature`` and samples.
        ``k = 0`` samples from the MLE distribution (only continuations that were
        really observed); ``k > 0`` samples from the add-k distribution.
        ``<unk>`` is never emitted: it is removed and the rest renormalised.

        Generation stops at ``</s>`` when ``stop_at_eos`` is true; otherwise a new
        sentence is started. ``max_words`` bounds the loop in every case, and the
        same ``seed`` always reproduces the same text.
        """
        model = self.model
        order = model.order if order is None else order
        if isinstance(max_words, bool) or not isinstance(max_words, int) or max_words < 1:
            raise ValueError(f"max_words must be a positive integer, got {max_words!r}")
        k = validate_k(k)
        rng = random.Random(seed)

        history = context_tokens(seed_text)
        generated: list[str] = []
        steps: list[GenerationStep] = []
        stop_reason = "max_words"
        words_emitted = 0

        while words_emitted < max_words:
            requested = model.resolve_context(history, order)
            used = model.backoff_context(requested)
            distribution = model.distribution(used, k)
            distribution.pop(UNK, None)
            # A sentence may not end before it has a word, otherwise the output could
            # be empty or contain empty sentences.
            if not history:
                distribution.pop(EOS, None)
            tempered = apply_temperature(distribution, temperature)
            if not tempered:
                # Only reachable when the sole continuation was removed above; the
                # unigram distribution always contains at least one real word.
                tempered = apply_temperature(
                    {w: p for w, p in model.distribution((), k).items() if w not in (UNK, EOS)},
                    temperature,
                )
                used = ()
            if not tempered:                      # vocabulary has no real words at all
                stop_reason = "no_candidates"
                break
            ranked = rank_candidates(tempered, model.followers(used))
            chosen, prob = sample(ranked, rng)
            steps.append(
                GenerationStep(
                    index=len(steps) + 1,
                    context=list(used),
                    order_used=len(used) + 1,
                    backed_off=used != requested,
                    num_candidates=len(ranked),
                    chosen=chosen,
                    probability=prob,
                    top_candidates=ranked[:5],
                )
            )
            generated.append(chosen)
            if chosen == EOS:
                if stop_at_eos:
                    stop_reason = "end_of_sentence"
                    break
                history = []                      # next word starts a fresh sentence
            else:
                history.append(chosen)
                words_emitted += 1

        return GenerationResult(
            seed_text=seed_text,
            generated_tokens=generated,
            text=render_text(seed_text, generated),
            steps=steps,
            stop_reason=stop_reason,
            order=order,
            temperature=float(temperature),
            k=k,
            seed=seed,
        )


def render_text(seed_text: str, generated: list[str]) -> str:
    """Join the seed and sampled tokens into readable, sentence-cased text."""
    sentences: list[list[str]] = tokenize_sentences(seed_text) or [[]]
    closed = [True] * (len(sentences) - 1) + [ends_sentence(seed_text)]
    if closed[-1]:
        sentences.append([])
        closed.append(False)
    for token in generated:
        if token == EOS:
            closed[-1] = True
            sentences.append([])
            closed.append(False)
        else:
            sentences[-1].append(token)
    parts = []
    for words, is_closed in zip(sentences, closed):
        if not words:
            continue
        sentence = " ".join(words)
        sentence = sentence[0].upper() + sentence[1:]
        parts.append(sentence + ("." if is_closed else ""))
    return " ".join(parts)
