"""The common recommender interface: score every catalogue item for a partial basket."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


class Recommender:
    name = "base"

    def fit(self, baskets: Sequence[Sequence[int]], n_items: int) -> "Recommender":
        raise NotImplementedError

    def score(self, visible: Sequence[int]) -> np.ndarray:
        """A score per catalogue item (higher = more likely to be added next)."""
        raise NotImplementedError

    def score_many(self, cases: Sequence[Sequence[int]]) -> np.ndarray:
        return np.stack([self.score(v) for v in cases])

    def recommend(self, visible: Sequence[int], k: int = 10) -> List[Tuple[int, float]]:
        s = np.asarray(self.score(visible), dtype=float).copy()
        s[list(visible)] = -np.inf                       # never recommend what is already in the basket
        top = np.argsort(-s, kind="stable")[:k]
        return [(int(i), float(s[i])) for i in top]
