"""The causal-LM loss, computed only where it is graded.

GPT-2's output layer is a 768 x 50,357 matrix. The stock forward pass applies it at
*every* position (about 40% of the forward FLOPs on 64-token clauses) and then throws
almost all of it away, because only the ``<Type>`` and ``<|endoftext|>`` positions carry a
label. This computes hidden states for the whole sequence, but logits only where
``labels != -100``. The loss is identical to ``model(**batch).loss`` (tested).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def answer_loss(model, input_ids: torch.Tensor, labels: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    hidden = model.transformer(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    targets = labels[:, 1:]                       # position t predicts token t + 1
    keep = targets != -100
    logits = model.lm_head(hidden[:, :-1][keep])  # (n_graded, vocab)
    return F.cross_entropy(logits.float(), targets[keep])
