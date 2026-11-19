"""A compact GPT-2 that loads the real OpenAI/Hugging Face weights.

Module names mirror Hugging Face's ``GPT2Model`` exactly (``h.3.attn.c_attn`` ...), so
the official ``model.safetensors`` loads with a plain ``load_state_dict`` and LoRA targets
use the same names as in PEFT (``c_attn``, ``c_proj``, ``c_fc``).

Writing it out (instead of importing ``transformers``) keeps every layer visible and lets
this project run outside the workbench container too.
"""

from __future__ import annotations

import glob
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Conv1D


@dataclass
class GPT2Config:
    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    layer_norm_epsilon: float = 1e-5


def gelu_new(x: torch.Tensor) -> torch.Tensor:
    """GPT-2's tanh approximation of GELU."""
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


class Attention(nn.Module):
    def __init__(self, cfg: GPT2Config) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.c_attn = Conv1D(cfg.n_embd, 3 * cfg.n_embd)   # W_Q | W_K | W_V   <- LoRA target
        self.c_proj = Conv1D(cfg.n_embd, cfg.n_embd)       # W_O
        mask = torch.tril(torch.ones(cfg.n_positions, cfg.n_positions, dtype=torch.bool))
        self.register_buffer("causal", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        hd = C // self.n_head
        q, k, v = (t.view(B, T, self.n_head, hd).transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        att = att.masked_fill(~self.causal[:T, :T], float("-inf")).softmax(-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPT2Config) -> None:
        super().__init__()
        self.c_fc = Conv1D(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = Conv1D(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(gelu_new(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPT2Config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)
        self.attn = Attention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT2(nn.Module):
    def __init__(self, cfg: GPT2Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.n_positions, cfg.n_embd)
        self.h = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Returns final hidden states (B, T, n_embd)."""
        T = idx.size(1)
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        for block in self.h:
            x = block(x)
        return self.ln_f(x)

    def lm_logits(self, idx: torch.Tensor) -> torch.Tensor:
        return self.forward(idx) @ self.wte.weight.t()      # tied output head


class GPT2Classifier(nn.Module):
    """GPT-2 + a linear head on the hidden state of the LAST real token.

    GPT-2 is causal, so the last token is the only one that has seen the whole sentence.
    Sequences are right-padded; padding sits after the last real token and, thanks to the
    causal mask, cannot influence it, so no attention mask is needed.
    """

    def __init__(self, gpt2: GPT2, num_classes: int = 2) -> None:
        super().__init__()
        self.transformer = gpt2
        self.score = nn.Linear(gpt2.cfg.n_embd, num_classes, bias=False)
        nn.init.normal_(self.score.weight, std=0.02)

    def forward(self, idx: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        h = self.transformer(idx)
        last = h[torch.arange(idx.size(0), device=idx.device), lengths - 1]
        return self.score(last)


# ---------------------------------------------------------------------------- weights
def hf_cache_dir() -> Path:
    root = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")), "hub")
    return Path(root)


def find_file(name: str, repos=("models--openai-community--gpt2", "models--gpt2")) -> Optional[Path]:
    for repo in repos:
        hits = sorted(glob.glob(str(hf_cache_dir() / repo / "snapshots" / "*" / name)))
        if hits:
            return Path(hits[-1])
    return None


def locate_weights(path: Optional[str] = None, download: bool = True) -> Path:
    if path:
        return Path(path)
    found = find_file("model.safetensors")
    if found:
        return found
    if not download:
        raise FileNotFoundError("GPT-2 weights not found in the Hugging Face cache")
    from huggingface_hub import hf_hub_download   # installed in the lfllm workbench
    return Path(hf_hub_download("openai-community/gpt2", "model.safetensors"))


def load_gpt2(path: Optional[str] = None, download: bool = True) -> GPT2:
    from safetensors.torch import load_file
    sd = load_file(str(locate_weights(path, download)))
    sd = {k.removeprefix("transformer."): v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if not k.endswith((".attn.bias", ".attn.masked_bias"))}
    sd.pop("lm_head.weight", None)                          # tied to wte
    model = GPT2(GPT2Config())
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"weight mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    return model.eval()


def load_tokenizer():
    """GPT-2's byte-level BPE (50,257 tokens) from the cached tokenizer.json."""
    from tokenizers import Tokenizer
    p = find_file("tokenizer.json")
    if p is None:
        from huggingface_hub import hf_hub_download
        p = Path(hf_hub_download("openai-community/gpt2", "tokenizer.json"))
    return Tokenizer.from_file(str(p))
