"""Bootstrap confidence intervals, single and paired.

**Percentile bootstrap.** Resample the n evaluation cases with replacement B times, recompute
the mean each time, and report the 2.5th and 97.5th percentiles of those B means as a 95%
interval.

**Paired bootstrap.** To compare two models on the *same* cases, resample case indices once
per replicate and apply the same indices to both models' per-case scores. The difference is
then computed on identical cases, so case-to-case difficulty cancels and the interval for
the difference is much tighter than two independent intervals would suggest. We report the
interval for the difference, and the share of replicates in which model A beats model B.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05,
                 stat: Callable[[np.ndarray], float] = np.mean) -> Dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    reps = v[idx].mean(axis=1) if stat is np.mean else np.array([stat(v[i]) for i in idx])
    lo, hi = np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"estimate": float(stat(v)), "lo": float(lo), "hi": float(hi), "n": int(len(v)),
            "n_boot": n_boot, "seed": seed}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0,
                     alpha: float = 0.05) -> Dict[str, float]:
    """Interval for mean(a) - mean(b) when a[i] and b[i] are scores on the same case i."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired bootstrap needs the same cases for both models")
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    reps = d[idx].mean(axis=1)
    lo, hi = np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"difference": float(d.mean()), "lo": float(lo), "hi": float(hi), "p_a_better": float((reps > 0).mean()),
            "n": int(len(d)), "n_boot": n_boot, "seed": seed}


def bootstrap_set_recall(hits: np.ndarray, n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    """CI for a recovery rate (share of true itemsets found), resampling the itemsets."""
    return bootstrap_ci(np.asarray(hits, dtype=float), n_boot, seed)
