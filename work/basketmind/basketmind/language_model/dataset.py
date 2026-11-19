"""Turn baskets into padded training batches, and the next-token loss.

    tokens:   <basket>  p17  p402  p9  </basket>  <pad> <pad>
    inputs:   positions 0 .. T-2       (what the model reads)
    targets:  positions 1 .. T-1       (the next token at each position: labels shifted by one)
    loss:     cross-entropy over target positions whose token is not <pad>

GPT-2's attention is causal: position t attends only to positions <= t. ``attention_mask``
additionally stops real tokens from attending to padding. Padding is only ever on the right,
so it never sits *before* a real token, and the positions of real tokens are unaffected.
"""

from __future__ import annotations

from typing import Iterator, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..data.verbalizer import Vocabulary, encode


def pad_batch(seqs: Sequence[Sequence[int]], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.as_tensor(s, dtype=torch.long)
        mask[i, :len(s)] = 1
    return ids, mask


def mask_invalid_next(logits: torch.Tensor, vocab: Vocabulary) -> torch.Tensor:
    """<basket> and <pad> are never a valid next token: give them -inf before any softmax."""
    logits = logits.float().clone()
    logits[..., vocab.open_id] = float("-inf")
    logits[..., vocab.pad_id] = float("-inf")
    return logits


def next_token_loss(logits: torch.Tensor, ids: torch.Tensor, mask: torch.Tensor, reduction: str = "mean",
                    vocab: Vocabulary | None = None):
    """Cross-entropy of predicting token t+1 from tokens <= t, ignoring padded targets."""
    pred = logits[:, :-1].float()
    if vocab is not None:
        pred = mask_invalid_next(pred, vocab)
    target = ids[:, 1:]
    keep = mask[:, 1:].bool()
    loss = F.cross_entropy(pred[keep], target[keep], reduction="sum")
    n = keep.sum()
    return (loss / n) if reduction == "mean" else (loss, n)


class BasketBatches:
    """Endless shuffled batches of encoded baskets under an ordering policy.

    ``fresh`` draws a new permutation at every presentation; ``frozen`` uses a per-basket seed
    so a basket always reads the same; ``fixed`` sorts by product id.
    """

    def __init__(self, baskets: Sequence[Sequence[int]], vocab: Vocabulary, batch_size: int, ordering: str,
                 seed: int = 0, max_items: int = 40):
        self.baskets = [sorted(set(b))[:max_items] for b in baskets]
        self.vocab = vocab
        self.batch_size = batch_size
        self.ordering = ordering
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.epoch = 0
        self.presented = 0

    def state(self) -> dict:
        return {"rng": self.rng.bit_generator.state, "epoch": self.epoch, "presented": self.presented}

    def restore(self, state: dict) -> None:
        self.rng.bit_generator.state = state["rng"]
        self.epoch, self.presented = state["epoch"], state["presented"]

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        while True:
            order = self.rng.permutation(len(self.baskets))
            for start in range(0, len(order) - self.batch_size + 1, self.batch_size):
                idx = order[start:start + self.batch_size]
                seqs = [encode(self.baskets[i], self.vocab, self.ordering, self.rng, frozen_seed=self.seed * 1_000_003 + int(i))
                        for i in idx]
                self.presented += len(idx)
                yield pad_batch(seqs, self.vocab.pad_id)
            self.epoch += 1


def encode_eval(baskets: Sequence[Sequence[int]], vocab: Vocabulary, ordering: str, seed: int = 123) -> List[List[int]]:
    """Each basket encoded once (deterministically) for validation-loss measurement."""
    rng = np.random.default_rng(seed)
    return [encode(b, vocab, ordering, rng, frozen_seed=seed + k) for k, b in enumerate(baskets)]


@torch.no_grad()
def validation_loss(model, seqs: List[List[int]], pad_id: int, batch_size: int = 128) -> float:
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(seqs), batch_size):
        ids, mask = pad_batch(seqs[i:i + batch_size], pad_id)
        loss, n = next_token_loss(model(ids, attention_mask=mask).logits, ids, mask, reduction="sum", vocab=model.vocab)
        total += float(loss)
        count += int(n)
    return total / count
