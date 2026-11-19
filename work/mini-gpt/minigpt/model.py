"""A decoder-only GPT, written out layer by layer.

    token ids (B, T)
      -> token embedding (B, T, C)  +  positional encoding (T, C)
      -> N x [ x + Attention(LayerNorm(x)) ;  x + MLP(LayerNorm(x)) ]     (pre-norm blocks)
      -> final LayerNorm
      -> linear head (weights tied to the token embedding) -> logits (B, T, vocab)

Nothing here calls ``nn.MultiheadAttention``, ``nn.TransformerDecoder`` or
``F.scaled_dot_product_attention``: attention and layer norm are implemented by hand.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 512
    block_size: int = 64       # maximum context length T
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128          # C, must be divisible by n_head
    dropout: float = 0.1
    bias: bool = True
    pos_encoding: str = "learned"   # "learned" (GPT-2) or "sinusoidal" (Vaswani et al.)

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if self.pos_encoding not in ("learned", "sinusoidal"):
            raise ValueError("pos_encoding must be 'learned' or 'sinusoidal'")

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- pieces
class LayerNorm(nn.Module):
    """y = (x - mean) / sqrt(var + eps) * gamma + beta, over the last dimension."""

    def __init__(self, dim: int, bias: bool = True, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        y = (x - mean) / torch.sqrt(var + self.eps) * self.weight
        return y + self.bias if self.bias is not None else y


def sinusoidal_table(block_size: int, dim: int) -> torch.Tensor:
    """PE[pos, 2i] = sin(pos / 10000^(2i/d)),  PE[pos, 2i+1] = cos(pos / 10000^(2i/d))."""
    pos = torch.arange(block_size, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(block_size, dim)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)[:, : dim // 2]
    return pe


def causal_mask(T: int, device=None) -> torch.Tensor:
    """Lower-triangular boolean mask: position i may look at positions 0..i only."""
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                 mask: Optional[torch.Tensor] = None,
                                 dropout: Optional[nn.Module] = None
                                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """softmax(Q K^T / sqrt(d_k) + mask) V.  Returns (output, attention weights)."""
    d_k = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)          # (..., T, T)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))        # the future gets -inf
    weights = torch.softmax(scores, dim=-1)                       # -inf -> exactly 0
    if dropout is not None:
        weights_used = dropout(weights)
    else:
        weights_used = weights
    return weights_used @ v, weights


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)   # W_Q, W_K, W_V in one
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)      # W_O
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.register_buffer("mask", causal_mask(cfg.block_size), persistent=False)
        self.last_attention: Optional[torch.Tensor] = None
        self.keep_attention = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim): every head attends independently.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        y, weights = scaled_dot_product_attention(q, k, v, self.mask[:T, :T], self.attn_drop)
        if self.keep_attention:
            self.last_attention = weights.detach()
        y = y.transpose(1, 2).contiguous().view(B, T, C)            # concatenate heads
        return self.resid_drop(self.proj(y))


class FeedForward(nn.Module):
    """Position-wise MLP: Linear(C -> 4C) -> GELU -> Linear(4C -> C)."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Pre-norm transformer block with two residual connections."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = LayerNorm(cfg.n_embd, cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = LayerNorm(cfg.n_embd, cfg.bias)
        self.mlp = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))   # tokens talk to each other
        x = x + self.mlp(self.ln2(x))    # each token thinks on its own
        return x


# --------------------------------------------------------------------------- the GPT
class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        if cfg.pos_encoding == "learned":
            self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        else:
            self.pos_emb = None
            self.register_buffer("pos_table", sinusoidal_table(cfg.block_size, cfg.n_embd),
                                 persistent=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = LayerNorm(cfg.n_embd, cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight   # weight tying (GPT-2)

        self.apply(self._init_weights)
        # GPT-2: scale the residual projections by 1/sqrt(2 * n_layer).
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        # Tied weights are counted once by parameters().
        return sum(p.numel() for p in self.parameters())

    def positions(self, T: int, device) -> torch.Tensor:
        if self.pos_emb is not None:
            return self.pos_emb(torch.arange(T, device=device))
        return self.pos_table[:T]

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.cfg.block_size}")
        x = self.drop(self.tok_emb(idx) + self.positions(T, idx.device))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))                           # (B, T, vocab)
        loss = None
        if targets is not None:
            # Next-token cross-entropy: position t predicts token t+1.
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                                   ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def attention_maps(self, idx: torch.Tensor) -> List[torch.Tensor]:
        """Run a forward pass and return each layer's (B, n_head, T, T) attention weights."""
        for b in self.blocks:
            b.attn.keep_attention = True
        try:
            self.forward(idx)
            return [b.attn.last_attention for b in self.blocks]
        finally:
            for b in self.blocks:
                b.attn.keep_attention = False
                b.attn.last_attention = None

    def configure_optimizer(self, lr: float, weight_decay: float, betas=(0.9, 0.95)):
        """AdamW; weight decay only on matrices (not biases, LayerNorm gains or 1-D params)."""
        params = {n: p for n, p in self.named_parameters() if p.requires_grad}
        decay = [p for p in params.values() if p.dim() >= 2]
        no_decay = [p for p in params.values() if p.dim() < 2]
        return torch.optim.AdamW([
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ], lr=lr, betas=betas)
