"""Next-item scores from the language model's actual probabilities.

For a partial basket V the model reads ``<basket> v1 v2 ... vk`` and its distribution over the
next token gives log P(next = product c | V) for all 1,000 products at once. A basket is a set,
so one order of V is arbitrary. We score ``n_orders`` random orders of V and average the
log-probabilities. Fixed seeds make this deterministic.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from ..data.verbalizer import encode
from .dataset import mask_invalid_next, pad_batch


@torch.no_grad()
def next_item_logprobs(model, visible_sets: Sequence[Sequence[int]], n_orders: int = 3, ordering: str = "fresh",
                       batch_size: int = 256, seed: int = 0) -> np.ndarray:
    """(n_cases, n_products) mean log P(next = product | visible), averaged over orders."""
    model.eval()
    vocab = model.vocab
    rng = np.random.default_rng(seed)
    orders = n_orders if ordering == "fresh" else 1
    seqs = [encode(v, vocab, ordering, rng, close=False) for v in visible_sets for _ in range(orders)]
    out = np.zeros((len(seqs), vocab.n_products), dtype=np.float32)
    for i in range(0, len(seqs), batch_size):
        ids, mask = pad_batch(seqs[i:i + batch_size], vocab.pad_id)
        logits = model(ids, attention_mask=mask).logits
        last = mask_invalid_next(logits[torch.arange(len(ids)), mask.sum(1) - 1], vocab)
        out[i:i + len(ids)] = torch.log_softmax(last, -1)[:, :vocab.n_products].numpy()
    return out.reshape(len(visible_sets), orders, vocab.n_products).mean(1)


@torch.no_grad()
def empty_basket_logprobs(model) -> np.ndarray:
    """log P(first item = product) after just ``<basket>``: the model's notion of popularity."""
    return next_item_logprobs(model, [[]], n_orders=1)[0]
