"""Raw text -> training batches.

* Documents (separated by blank lines) are split into train / validation BEFORE the
  tokenizer is trained, so neither the BPE merges nor the model see validation text.
* Every document is followed by <|endoftext|>, so the model learns where text ends.
* A batch is B random windows of T+1 tokens: x = window[:-1], y = window[1:].
  Position t of x is trained to predict y[t] = the token right after it.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch

from .tokenizer import BPETokenizer


def read_documents(path: str | Path) -> List[str]:
    text = Path(path).read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    docs = [d.strip() for d in re.split(r"\n\s*\n", text)]
    return [d for d in docs if d]


def split_documents(docs: List[str], val_fraction: float = 0.1, seed: int = 0
                    ) -> Tuple[List[str], List[str]]:
    """Deterministic document-level split. Duplicate documents stay on one side."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0, 1)")
    unique = list(dict.fromkeys(docs))
    if len(unique) < 2:
        raise ValueError("need at least 2 distinct documents to make a validation split")
    r = random.Random(seed)
    order = unique[:]
    r.shuffle(order)
    n_val = max(1, round(len(order) * val_fraction))
    val_set = set(order[:n_val])
    train = [d for d in docs if d not in val_set]
    val = [d for d in docs if d in val_set]
    return train, val


def encode_documents(tok: BPETokenizer, docs: List[str]) -> torch.Tensor:
    ids: List[int] = []
    for d in docs:
        ids.extend(tok.encode_ordinary(d))
        ids.append(tok.eot_id)
    return torch.tensor(ids, dtype=torch.long)


@dataclass
class Batcher:
    data: torch.Tensor
    block_size: int
    batch_size: int
    seed: int = 0

    def __post_init__(self) -> None:
        if self.data.numel() < self.block_size + 1:
            raise ValueError(f"only {self.data.numel()} tokens; need more than block_size={self.block_size}")
        self.gen = torch.Generator().manual_seed(self.seed)

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        hi = self.data.numel() - self.block_size - 1
        starts = torch.randint(0, hi + 1, (self.batch_size,), generator=self.gen)
        x = torch.stack([self.data[s: s + self.block_size] for s in starts.tolist()])
        y = torch.stack([self.data[s + 1: s + 1 + self.block_size] for s in starts.tolist()])
        return x, y

    def sequential(self, max_batches: int | None = None):
        """Non-overlapping windows covering the data once (for a stable validation loss)."""
        T = self.block_size
        n = (self.data.numel() - 1) // T
        xs = [self.data[i * T: i * T + T] for i in range(n)]
        ys = [self.data[i * T + 1: i * T + T + 1] for i in range(n)]
        count = 0
        for i in range(0, n, self.batch_size):
            if max_batches is not None and count >= max_batches:
                return
            yield torch.stack(xs[i: i + self.batch_size]), torch.stack(ys[i: i + self.batch_size])
            count += 1


def unigram_cross_entropy(train_ids: torch.Tensor, val_ids: torch.Tensor, vocab_size: int) -> float:
    """Baseline: cross-entropy (nats) of an add-1 unigram model. A GPT must beat this."""
    counts = torch.bincount(train_ids, minlength=vocab_size).double() + 1.0
    logp = torch.log(counts / counts.sum())
    return float(-logp[val_ids[1:]].mean())
