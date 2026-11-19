"""Co-occurrence counting, and an association-rule recommender built on the exact miner.

``CoOccurrence`` keeps one n_items x n_items matrix C, where C[a, b] is the number of training
baskets holding both a and b (C[a, a] is a's own count). For a partial basket V it scores each
candidate c with one of:

* ``count``:      Σ_v C[v, c]                          raw co-occurrence
* ``confidence``: mean_v C[v, c] / C[v, v]             mean P(c | v)
* ``lift``:       mean_v P(c | v) / P(c)               how much v raises c above its base rate
* ``blend``:      log mean_v P(c | v) + β log P(c | v*) where v* is the most predictive visible item

``AssociationRules`` mines frequent itemsets with ``basketmind.mining.apriori`` and scores c by
the best rule A -> c whose antecedent A is contained in the basket (confidence, ties broken by
lift). Larger antecedents (pairs, triples) let it use more of the basket than pairwise counts
can.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Sequence

import numpy as np

from ..mining.apriori import apriori
from ..mining.rules import generate_rules
from .base import Recommender


class CoOccurrence(Recommender):
    def __init__(self, mode: str = "confidence", smoothing: float = 1.0):
        self.mode = mode
        self.smoothing = smoothing
        self.name = f"cooccurrence-{mode}"

    def fit(self, baskets, n_items):
        C = np.zeros((n_items, n_items), dtype=np.float32)
        for b in baskets:
            ix = np.array(sorted(set(b)))
            C[np.ix_(ix, ix)] += 1
        self.C = C
        self.n_baskets = len(baskets)
        self.item_count = np.diag(C).astype(np.float64)
        self.p = (self.item_count + self.smoothing) / (self.n_baskets + self.smoothing * n_items)
        return self

    def conditional(self, visible: Sequence[int]) -> np.ndarray:
        """P(c | v) for each visible v (rows) and candidate c (columns), smoothed."""
        v = np.asarray(visible)
        return (self.C[v] + self.smoothing * self.p) / (self.item_count[v, None] + self.smoothing)

    def score(self, visible):
        if len(visible) == 0:
            return np.log(self.p)
        if self.mode == "count":
            return self.C[np.asarray(visible)].sum(0).astype(np.float64)
        cond = self.conditional(visible)
        if self.mode == "confidence":
            return np.log(cond.mean(0))
        if self.mode == "lift":
            return np.log(cond.mean(0) / self.p)
        if self.mode == "blend":
            return np.log(cond.mean(0)) + 0.5 * np.log(cond.max(0))
        raise ValueError(self.mode)

    def evidence(self, visible: Sequence[int], c: int) -> dict:
        """Why c? The visible item that supports it most, with support, confidence and lift."""
        best = max(visible, key=lambda v: self.C[v, c] / max(1.0, self.item_count[v]))
        both = float(self.C[best, c])
        conf = both / max(1.0, self.item_count[best])
        sup_c = self.item_count[c] / self.n_baskets
        return {"because_of": int(best), "pair_support": both / self.n_baskets, "confidence": conf,
                "lift": conf / sup_c if sup_c else float("nan"), "pair_count": int(both)}


class AssociationRules(Recommender):
    name = "association-rules"

    def __init__(self, min_support: float = 0.001, min_confidence: float = 0.05, max_size: int = 4):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.max_size = max_size

    def fit(self, baskets, n_items):
        counts = apriori(baskets, n_items, self.min_support, self.max_size)
        n = len(baskets)
        sup = {k: v / n for k, v in counts.items()}
        self.rules = generate_rules(sup, self.min_confidence, max_consequent=1)
        by_ante: Dict[FrozenSet[int], List] = {}
        for r in self.rules:
            by_ante.setdefault(r.antecedent, []).append(r)
        self.by_antecedent = by_ante
        self.n_items = n_items
        self.fallback = np.zeros(n_items)
        for k, v in sup.items():
            if len(k) == 1:
                self.fallback[next(iter(k))] = v
        return self

    def score(self, visible):
        """Best matching rule per candidate: confidence, ties broken by lift.

        Only antecedents that are subsets of the basket can fire, so we enumerate the basket's
        subsets (up to max_size - 1 items) and look them up, rather than scanning every rule.
        """
        from itertools import combinations
        vis = sorted(set(visible))
        s = self.fallback * 1e-3                         # unmatched items: ordered by popularity, far below any rule
        best_rule = {}
        for k in range(1, min(len(vis), self.max_size - 1) + 1):
            for ante in combinations(vis, k):
                for r in self.by_antecedent.get(frozenset(ante), ()):
                    c = next(iter(r.consequent))
                    val = r.confidence + 1e-6 * min(r.lift, 1e3)
                    if val > s[c]:
                        s[c] = val
                        best_rule[c] = r
        self._last = best_rule
        return s

    def explain(self, visible, c):
        self.score(visible)
        r = self._last.get(c)
        return r.to_dict() if r else None
