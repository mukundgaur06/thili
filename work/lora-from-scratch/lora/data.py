"""SST-2 (Stanford Sentiment Treebank, binary) for fine-tuning.

Loaded from the local Hugging Face datasets cache (Arrow files) when present, otherwise
with ``datasets.load_dataset("stanfordnlp/sst2")``. GLUE's SST-2 test labels are hidden,
so the official *validation* split (872 sentences) is our untouched TEST set, and a dev
split for model selection is carved out of train.
"""

from __future__ import annotations

import glob
import os
import random
from typing import List, Tuple

import torch

Example = Tuple[str, int]


def _from_arrow() -> dict | None:
    root = os.environ.get("HF_DATASETS_CACHE") or os.path.join(os.path.expanduser("~"), ".cache",
                                                               "huggingface", "datasets")
    hits = sorted(glob.glob(os.path.join(root, "*sst2*", "**", "sst2-train.arrow"), recursive=True))
    if not hits:
        return None
    import pyarrow as pa
    folder = os.path.dirname(hits[-1])
    out = {}
    for split in ("train", "validation"):
        with pa.memory_map(os.path.join(folder, f"sst2-{split}.arrow")) as f:
            table = pa.ipc.open_stream(f).read_all()
        out[split] = list(zip(table.column("sentence").to_pylist(), table.column("label").to_pylist()))
    return out


def _from_hub() -> dict:
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2")
    return {s: list(zip(ds[s]["sentence"], ds[s]["label"])) for s in ("train", "validation")}


def load_sst2(n_train: int = 4000, n_dev: int = 500, seed: int = 0):
    raw = _from_arrow() or _from_hub()
    train = [(s.strip(), int(y)) for s, y in raw["train"]]
    test = [(s.strip(), int(y)) for s, y in raw["validation"]]
    test_texts = {s for s, _ in test}
    train = [ex for ex in train if ex[0] not in test_texts]      # no test sentence in training
    rng = random.Random(seed)
    rng.shuffle(train)
    dev = train[:n_dev]
    dev_texts = {s for s, _ in dev}
    rest = [ex for ex in train[n_dev:] if ex[0] not in dev_texts]
    return rest[:n_train], dev, test


class Batches:
    """Length-bucketed, right-padded batches (padding never reaches the last real token)."""

    def __init__(self, examples: List[Example], tokenizer, batch_size: int, max_len: int = 64,
                 shuffle: bool = True, seed: int = 0, pad_id: int = 50256) -> None:
        self.items = []
        for text, label in examples:
            ids = tokenizer.encode(text).ids[:max_len] or [pad_id]
            self.items.append((ids, label))
        self.batch_size, self.shuffle, self.pad_id = batch_size, shuffle, pad_id
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return (len(self.items) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        order = list(range(len(self.items)))
        if self.shuffle:
            self.rng.shuffle(order)
            # sort within chunks of 50 batches so batches have similar lengths (less padding)
            chunk = self.batch_size * 50
            order = [i for c in range(0, len(order), chunk)
                     for i in sorted(order[c:c + chunk], key=lambda j: len(self.items[j][0]))]
            batches = [order[i:i + self.batch_size] for i in range(0, len(order), self.batch_size)]
            self.rng.shuffle(batches)
        else:
            batches = [order[i:i + self.batch_size] for i in range(0, len(order), self.batch_size)]
        for b in batches:
            seqs = [self.items[i][0] for i in b]
            T = max(len(s) for s in seqs)
            x = torch.full((len(b), T), self.pad_id, dtype=torch.long)
            for r, s in enumerate(seqs):
                x[r, :len(s)] = torch.tensor(s)
            yield x, torch.tensor([len(s) for s in seqs]), torch.tensor([self.items[i][1] for i in b])
