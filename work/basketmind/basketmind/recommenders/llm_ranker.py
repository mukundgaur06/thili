"""The language model as a recommender: rank products by log P(next item | basket)."""

from __future__ import annotations

import numpy as np

from ..language_model.inference import next_item_logprobs
from .base import Recommender


class LLMRanker(Recommender):
    def __init__(self, model, name: str = "llm", n_orders: int = 3, ordering: str = "fresh", seed: int = 0):
        self.model = model
        self.name = name
        self.n_orders = n_orders
        self.ordering = ordering
        self.seed = seed

    def fit(self, baskets, n_items):
        return self                                      # trained separately (continued pretraining)

    def score(self, visible):
        return self.score_many([visible])[0]

    def score_many(self, cases):
        return next_item_logprobs(self.model, cases, self.n_orders, self.ordering, seed=self.seed)

    def probability(self, visible) -> np.ndarray:
        return np.exp(self.score(visible))
