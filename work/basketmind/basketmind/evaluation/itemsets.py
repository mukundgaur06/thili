"""Itemset-level evaluation: why is an itemset frequent, and how much of the truth was recovered?

A *planted* itemset is one we put into the data. A *frequent* itemset is anything the exact
miner finds above the support threshold. They are not the same thing: planting {A, B, C}
makes {A, B} frequent too (a "part"), popular items meet by chance ("chance"), and the planted
baskets also hold whatever else the shopper bought ("extension"). Every frequent itemset is
labelled so that recovery can be reported for each kind separately.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Iterable, List, Sequence

import numpy as np

LABELS = ("single", "planted", "part", "extension", "chance")


def classify(itemset: Iterable[int], planted) -> str:
    s = frozenset(itemset)
    if len(s) == 1:
        return "single"
    sets = [frozenset(p.items) for p in planted]
    if s in sets:
        return "planted"
    if any(s < p for p in sets):
        return "part"
    if any(len(s & p) >= 2 for p in sets):
        return "extension"
    return "chance"


def planted_recovery(found: Dict[FrozenSet[int], float], planted, min_support: float) -> List[dict]:
    """For each planted itemset: was it found frequent (support >= min_support) in `found`?"""
    rows = []
    for p in planted:
        s = found.get(frozenset(p.items), 0.0)
        rows.append({"items": list(p.items), "size": len(p.items), "kind": p.kind, "target_support": p.target_support,
                     "found_support": s, "recovered": s >= min_support})
    return rows


def recall_by_band(found: Dict[FrozenSet[int], float], truth: Dict[FrozenSet[int], float],
                   labels: Dict[FrozenSet[int], str], bands: Sequence[float], min_support: float,
                   only: Sequence[str] = ("planted", "part")) -> List[dict]:
    """Recall of true frequent itemsets with the given labels, per band of true support."""
    rows = []
    for lo, hi in zip(bands[:-1], bands[1:]):
        chosen = [s for s, v in truth.items() if lo <= v < hi and labels.get(s) in only and len(s) >= 2]
        hit = sum(found.get(s, 0.0) >= min_support for s in chosen)
        rows.append({"band": [lo, hi], "true": len(chosen), "found": hit,
                     "recall": hit / len(chosen) if chosen else float("nan")})
    return rows


def precision(found: Dict[FrozenSet[int], float], truth: Dict[FrozenSet[int], float], min_support: float,
              min_size: int = 2) -> float:
    """Of the multi-item itemsets called frequent, the share whose *true* support clears the bar."""
    called = [s for s, v in found.items() if v >= min_support and len(s) >= min_size]
    if not called:
        return float("nan")
    return float(np.mean([truth.get(s, 0.0) >= min_support for s in called]))


def support_error(found: Dict[FrozenSet[int], float], truth: Dict[FrozenSet[int], float],
                  keys: Iterable[FrozenSet[int]]) -> np.ndarray:
    return np.array([found.get(k, 0.0) - truth[k] for k in keys])
