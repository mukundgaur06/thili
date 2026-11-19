"""The domain vocabulary, and how its embeddings are initialised from GPT-2.

GPT-2 would spell "Meadow Farm Milk Regular" as about 5 BPE pieces. Here every product is one
token, so its id is preserved exactly and P(next item) for all 1,000 products comes from one
forward pass. The baskets use nothing else, so the 50,257-row GPT-2 vocabulary is replaced by
a 1,003-row one (1,000 products, <basket>, </basket>, <pad>). That saves 150 MB of
weights and gradients, which is what made training fit in this machine's memory.

The new rows do not start at random. Row ``<p0042>`` starts as the **mean of GPT-2's own
embeddings of the product's name** ("Meadow Farm Milk Regular"), and ``<basket>`` /
``</basket>`` start from " basket" and GPT-2's end-of-text token. GPT-2 ties its output layer
to these embeddings, so at step 0 the model already scores products by what it knows about
their names. That untrained model is the **off-the-shelf baseline**.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from ..data.catalogue import Product
from ..data.verbalizer import Vocabulary

GPT2 = "openai-community/gpt2"


def gpt2_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(GPT2)


@torch.no_grad()
def init_embeddings(old_wte: torch.Tensor, catalogue: List[Product], vocab: Vocabulary, tok=None) -> torch.Tensor:
    tok = tok or gpt2_tokenizer()
    E = torch.zeros(vocab.size, old_wte.shape[1])
    for p in catalogue:
        E[p.pid] = old_wte[tok(" " + p.name).input_ids].mean(0)
    E[vocab.open_id] = old_wte[tok(" basket").input_ids].mean(0)
    E[vocab.close_id] = old_wte[tok.eos_token_id]
    # Not zero: pretrained GPT-2's logits are large and negative for every token, so a zero row
    # would get logit 0 and soak up nearly all the probability mass. <pad> and <basket> are also
    # masked out of every softmax (see dataset.mask_invalid_next), so they can never be predicted.
    E[vocab.pad_id] = old_wte.mean(0)
    return E


def vocabulary_stats(catalogue: List[Product], tok=None) -> dict:
    tok = tok or gpt2_tokenizer()
    pieces = np.array([len(tok(" " + p.name).input_ids) for p in catalogue])
    return {"products": len(catalogue), "gpt2_pieces_per_name_mean": float(pieces.mean()),
            "gpt2_pieces_per_name_max": int(pieces.max()), "domain_tokens_per_product": 1,
            "domain_vocab_size": len(catalogue) + 3, "gpt2_vocab_size": tok.vocab_size}
