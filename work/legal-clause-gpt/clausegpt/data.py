"""Lesson 2 and 3: load LEDGAR and turn each clause into a causal-LM training example."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from .labels import EOS, SEP, label_token

DATASET = ("coastalcph/lex_glue", "ledgar")


def load_ledgar():
    """LEDGAR from LexGLUE: 60k train / 10k validation / 10k test clauses, 100 types."""
    from datasets import load_dataset
    ds = load_dataset(*DATASET)
    return ds, ds["train"].features["label"].names


def sample(split, n: Optional[int], seed: int = 0):
    """A seeded random subset (the whole split when n is None or too large)."""
    if n is None or n >= len(split):
        return split
    idx = list(range(len(split)))
    random.Random(seed).shuffle(idx)
    return split.select(sorted(idx[:n]))


def encode(tokenizer, clause: str, label: str, max_clause_tokens: int = 128,
           loss_on_prompt: bool = False) -> Dict[str, List[int]]:
    """Token ids for ``clause<Label><Type><|endoftext|>``.

    * The clause is cut to its first ``max_clause_tokens`` tokens *before* the separator,
      so truncation can never cut the answer off.
    * ``labels`` copy the inputs (the model shifts them by one internally). Clause tokens
      are set to -100 unless ``loss_on_prompt``: the model is graded only on naming the
      type and ending the answer, not on predicting legal boilerplate.
    """
    clause_ids = tokenizer(clause.strip()).input_ids[:max_clause_tokens]
    tail = tokenizer.convert_tokens_to_ids([SEP, label_token(label), EOS])
    input_ids = clause_ids + tail
    if loss_on_prompt:
        labels = list(input_ids)
    else:
        labels = [-100] * (len(clause_ids) + 1) + tail[1:]     # predict <Type> and EOS
    return {"input_ids": input_ids, "labels": labels}


def prompt_ids(tokenizer, clause: str, max_clause_tokens: int = 128) -> List[int]:
    clause_ids = tokenizer(clause.strip()).input_ids[:max_clause_tokens]
    return clause_ids + [tokenizer.convert_tokens_to_ids(SEP)]


@dataclass
class Collator:
    """Right-pads a batch: pad id for inputs, -100 for labels, 0 in the attention mask."""
    pad_id: int

    def __call__(self, batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), self.pad_id, dtype=torch.long)
        lab = torch.full((len(batch), n), -100, dtype=torch.long)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            k = len(b["input_ids"])
            ids[i, :k] = torch.tensor(b["input_ids"])
            lab[i, :k] = torch.tensor(b["labels"])
            att[i, :k] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}
