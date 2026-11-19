"""Lesson 6 and 8: load the fine-tuned model and ask it to tag clauses.

Two ways to read the answer from the *same* next-token distribution after ``<Label>``:

* **free generation**: greedy decoding of one token over the whole 50k vocabulary, exactly
  what ``model.generate(max_new_tokens=1)`` returns. It can produce something that is not
  a clause type at all; we count those as wrong ("invalid").
* **constrained**: argmax over the 100 label tokens only, so the answer is always a valid
  type. It also gives a probability for each type (softmax over the label rows).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

from .data import prompt_ids
from .labels import label_token

LABELS_FILE = "clause_labels.json"


@dataclass
class Prediction:
    generated: str               # the raw token greedy decoding produced
    valid: bool                  # was it one of the 100 label tokens?
    label: str                   # constrained answer (always a clause type)
    top: List[tuple]             # [(type, probability), ...] over clause types


class ClauseTagger:
    def __init__(self, model: GPT2LMHeadModel, tokenizer, label_names: List[str],
                 max_clause_tokens: int = 64) -> None:
        self.model = model.eval()
        self.tok = tokenizer
        self.names = list(label_names)
        self.max_clause_tokens = max_clause_tokens
        self.label_ids = torch.tensor(tokenizer.convert_tokens_to_ids([label_token(n) for n in self.names]))
        self.id_to_name = {int(i): n for i, n in zip(self.label_ids, self.names)}

    # ------------------------------------------------------------------ persistence
    def save(self, out_dir: str) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out)
        self.tok.save_pretrained(out)
        (out / LABELS_FILE).write_text(json.dumps(
            {"labels": self.names, "max_clause_tokens": self.max_clause_tokens}, indent=1))

    @classmethod
    def load(cls, out_dir: str) -> "ClauseTagger":
        meta = json.loads((Path(out_dir) / LABELS_FILE).read_text())
        tok = AutoTokenizer.from_pretrained(out_dir)
        model = GPT2LMHeadModel.from_pretrained(out_dir)
        return cls(model, tok, meta["labels"], meta["max_clause_tokens"])

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def next_token_logits(self, clauses: List[str]) -> torch.Tensor:
        """Logits for the token after ``<Label>``, one row per clause (right-padded batch)."""
        seqs = [prompt_ids(self.tok, c, self.max_clause_tokens) for c in clauses]
        n = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), n), self.tok.pad_token_id, dtype=torch.long)
        att = torch.zeros_like(ids)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
            att[i, :len(s)] = 1
        hidden = self.model.transformer(input_ids=ids, attention_mask=att).last_hidden_state
        last = hidden[torch.arange(len(seqs)), att.sum(1) - 1]
        return self.model.lm_head(last)

    def predict(self, clauses: List[str], k: int = 3) -> List[Prediction]:
        logits = self.next_token_logits(clauses)
        free = logits.argmax(-1)
        probs = logits[:, self.label_ids].softmax(-1)
        out = []
        for i in range(len(clauses)):
            gid = int(free[i])
            p, j = probs[i].topk(k)
            top = [(self.names[int(jj)], float(pp)) for pp, jj in zip(p, j)]
            out.append(Prediction(generated=self.tok.decode([gid]),
                                  valid=gid in self.id_to_name, label=top[0][0], top=top))
        return out

    def tag(self, clause: str) -> str:
        """The headline API: clause text in, ``<Clause Type>`` out."""
        return label_token(self.predict([clause], k=1)[0].label)

    @torch.no_grad()
    def generate(self, clause: str, max_new_tokens: int = 2) -> str:
        """Plain ``model.generate`` (greedy), to show the tagger really is a language model."""
        ids = torch.tensor([prompt_ids(self.tok, clause, self.max_clause_tokens)])
        out = self.model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                                  do_sample=False, pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False)


def batched(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def predict_all(tagger: ClauseTagger, clauses: List[str], batch_size: int = 32,
                progress: Optional[Callable[[int], None]] = None) -> List[Prediction]:
    preds: List[Prediction] = []
    for chunk in batched(clauses, batch_size):
        preds.extend(tagger.predict(chunk))
        if progress:
            progress(len(preds))
    return preds
