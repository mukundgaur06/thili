"""BasketLM: pretrained GPT-2 with the domain vocabulary and LoRA adapters.

* The 12 pretrained GPT-2 transformer blocks (85M parameters) are loaded from the Hugging Face
  cache and **frozen**.
* Low-rank adapters (LoRA, h = W0 x + (α/r) B A x, with B = 0 at init) are added to every
  block's attention projection (``c_attn``) and first MLP layer (``c_fc``).
* The 1,003-row domain embedding (tied to the output layer) and the final LayerNorm are
  trained in full.

That is about 2.1M trainable parameters out of 88M. It is continued pretraining with the
plain next-token objective, made parameter-efficient so it fits in about 1 GB of RAM on a
CPU. ``full_finetune=True`` trains every weight instead, for machines with more memory.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import torch
import torch.nn as nn

from ..data.catalogue import Product
from ..data.verbalizer import Vocabulary
from .tokenizer import GPT2, init_embeddings


@dataclass
class LMConfig:
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_targets: tuple = ("c_attn", "c_fc")
    full_finetune: bool = False
    n_products: int = 1000

    def to_dict(self) -> dict:
        d = asdict(self)
        d["lora_targets"] = list(self.lora_targets)
        return d


class LoRAConv1D(nn.Module):
    """LoRA around GPT-2's Conv1D (weight stored as (in, out)): y = x W + b + (α/r) x Aᵀ Bᵀ."""

    def __init__(self, base: nn.Module, r: int, alpha: float):
        super().__init__()
        self.base = base
        d_in, d_out = base.weight.shape
        self.lora_A = nn.Parameter(torch.empty(r, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling


class BasketLM(nn.Module):
    def __init__(self, gpt2, vocab: Vocabulary, cfg: LMConfig):
        super().__init__()
        self.gpt2 = gpt2
        self.vocab = vocab
        self.cfg = cfg

    @classmethod
    def from_pretrained_gpt2(cls, catalogue: List[Product], cfg: LMConfig | None = None) -> "BasketLM":
        from transformers import GPT2LMHeadModel
        cfg = cfg or LMConfig(n_products=len(catalogue))
        vocab = Vocabulary(len(catalogue))
        gpt2 = GPT2LMHeadModel.from_pretrained(GPT2)
        E = init_embeddings(gpt2.transformer.wte.weight.detach(), catalogue, vocab)
        gpt2.transformer.wte = nn.Embedding(vocab.size, E.shape[1])
        gpt2.transformer.wte.weight.data.copy_(E)
        gpt2.lm_head = nn.Linear(E.shape[1], vocab.size, bias=False)
        gpt2.lm_head.weight = gpt2.transformer.wte.weight                 # tied, as in GPT-2
        gpt2.config.vocab_size = vocab.size
        gpt2.config.pad_token_id = vocab.pad_id
        gpt2.config.bos_token_id = vocab.open_id
        gpt2.config.eos_token_id = vocab.close_id
        model = cls(gpt2, vocab, cfg)
        model._setup_trainable()
        return model

    def _setup_trainable(self) -> None:
        if not self.cfg.full_finetune:
            for p in self.gpt2.parameters():
                p.requires_grad_(False)
            for block in self.gpt2.transformer.h:
                for name in self.cfg.lora_targets:
                    holder = block.attn if name in ("c_attn",) else block.mlp
                    setattr(holder, name, LoRAConv1D(getattr(holder, name), self.cfg.lora_rank, self.cfg.lora_alpha))
            self.gpt2.transformer.wte.weight.requires_grad_(True)
            for p in self.gpt2.transformer.ln_f.parameters():
                p.requires_grad_(True)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kw):
        """Logits (B, T, V). GPT-2 applies the causal mask itself; ``attention_mask`` hides padding."""
        return self.gpt2(input_ids=input_ids, attention_mask=attention_mask, **kw)

    def trainable_state(self) -> dict:
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().clone() for k, v in self.state_dict().items() if k in names}

    def counts(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "percent": 100 * trainable / total}

    def save(self, directory: Path, extra: dict | None = None) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.trainable_state(), directory / "adapter.pt")
        (directory / "config.json").write_text(json.dumps({"lm": self.cfg.to_dict(), "base": GPT2, **(extra or {})}, indent=1))

    @classmethod
    def load(cls, directory: Path, catalogue: List[Product]) -> "BasketLM":
        meta = json.loads((Path(directory) / "config.json").read_text())
        lm = meta["lm"]
        cfg = LMConfig(lm["lora_rank"], lm["lora_alpha"], tuple(lm["lora_targets"]), lm["full_finetune"], lm["n_products"])
        model = cls.from_pretrained_gpt2(catalogue, cfg)
        state = torch.load(Path(directory) / "adapter.pt", map_location="cpu")
        missing = model.load_state_dict(state, strict=False)
        if missing.unexpected_keys:
            raise KeyError(f"unexpected keys in checkpoint: {missing.unexpected_keys[:3]}")
        return model.eval()
