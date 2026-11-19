"""Lesson 4: teach GPT-2's tokenizer the new words, and grow the model to match."""

from __future__ import annotations

from typing import List, Tuple

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, PreTrainedTokenizerBase

from .labels import SEP, new_tokens

BASE = "openai-community/gpt2"


def extend_tokenizer(tokenizer: PreTrainedTokenizerBase, label_names: List[str]) -> int:
    """Add ``<Label>`` and one token per clause type. Returns how many were new.

    They are added as *special* tokens, so the BPE never splits them: ``<Governing Laws>``
    is one id, not the 5 pieces '<', 'Gover', 'ning', ' Laws', '>'.
    """
    added = tokenizer.add_special_tokens({"additional_special_tokens": new_tokens(label_names)})
    tokenizer.pad_token = tokenizer.eos_token            # GPT-2 ships without a pad token
    tokenizer.padding_side = "right"
    return added


@torch.no_grad()
def resize_and_init(model: GPT2LMHeadModel, tokenizer: PreTrainedTokenizerBase,
                    label_names: List[str]) -> Tuple[int, int]:
    """Grow the embedding matrix and give every new row a sensible starting point.

    A new row starts as the mean embedding of the label's *words* under the old tokenizer,
    so ``<Governing Laws>`` begins near ' Governing' and ' Laws', not at random noise.
    GPT-2 ties its output layer to the input embeddings, so this also initialises the
    output row the model uses to *predict* that label.
    """
    old_n = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    emb = model.get_input_embeddings().weight
    base_tok = AutoTokenizer.from_pretrained(BASE)
    sep_id = tokenizer.convert_tokens_to_ids(SEP)
    emb[sep_id] = emb[base_tok(" Type:").input_ids].mean(0)
    for name in label_names:
        tid = tokenizer.convert_tokens_to_ids(f"<{name}>")
        emb[tid] = emb[base_tok(" " + name).input_ids].mean(0)
    return old_n, emb.shape[0]


def load_base(label_names: List[str]):
    """Pretrained GPT-2 plus the extended tokenizer, ready for fine-tuning."""
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    extend_tokenizer(tokenizer, label_names)
    model = GPT2LMHeadModel.from_pretrained(BASE)
    resize_and_init(model, tokenizer, label_names)
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer
