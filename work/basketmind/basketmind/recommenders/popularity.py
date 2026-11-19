"""Popularity: recommend best sellers, ignoring the basket. The floor every method must beat."""

from __future__ import annotations

import numpy as np

from .base import Recommender


class Popularity(Recommender):
    name = "popularity"

    def fit(self, baskets, n_items):
        counts = np.ones(n_items)                        # +1 smoothing: unseen items score low, not -inf
        for b in baskets:
            counts[list(set(b))] += 1
        self.counts = counts
        self.n_baskets = len(baskets)
        self.support = (counts - 1) / max(1, len(baskets))
        self._log = np.log(counts)
        return self

    def score(self, visible):
        return self._log.copy()
