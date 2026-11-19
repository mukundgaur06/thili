"""Probability estimators: Maximum Likelihood and add-k (Laplace) smoothing.

These are the only two places in the project where a count becomes a probability.
The model, predictor, generator and evaluator all route through them.
"""

from __future__ import annotations

import math


def validate_k(k: float) -> float:
    """Return ``k`` as a float, raising ``ValueError`` unless it is finite and >= 0.

    ``k = 0`` means "no smoothing" (plain MLE); smoothing itself requires ``k > 0``.
    """
    if isinstance(k, bool) or not isinstance(k, (int, float)):
        raise ValueError(f"smoothing parameter k must be a number, got {k!r}")
    if math.isnan(k) or math.isinf(k) or k < 0:
        raise ValueError(f"smoothing parameter k must be finite and >= 0, got {k!r}")
    return float(k)


def mle_probability(count: int, context_count: int) -> float:
    """Maximum Likelihood Estimate: ``P(w | context) = C(context, w) / C(context)``.

    When the context was never observed the estimate is 0/0 -- undefined. We return
    0.0 so callers can treat it as "this model assigns no probability here"; the
    evaluator turns that into an infinite cross-entropy rather than hiding it.
    """
    if context_count <= 0:
        return 0.0
    return count / context_count


def add_k_probability(count: int, context_count: int, k: float, vocab_size: int) -> float:
    """Add-k estimate: ``P(w | context) = (C(context, w) + k) / (C(context) + k * V)``.

    ``k = 1`` is Laplace smoothing. Every one of the ``V`` outcomes receives ``k``
    pseudo-counts, so the ``V`` probabilities still sum to exactly 1, and an unseen
    context degrades gracefully to the uniform distribution ``1 / V``.
    """
    k = validate_k(k)
    if k == 0:
        raise ValueError("add-k smoothing requires k > 0 (k = 0 is plain MLE)")
    if vocab_size < 1:
        raise ValueError("vocab_size must be >= 1")
    return (count + k) / (context_count + k * vocab_size)


def probability(count: int, context_count: int, k: float, vocab_size: int) -> float:
    """Dispatch on ``k``: MLE when ``k == 0``, add-k otherwise."""
    if validate_k(k) == 0:
        return mle_probability(count, context_count)
    return add_k_probability(count, context_count, k, vocab_size)
