"""LoRA from scratch: a frozen weight plus a trainable low-rank update.

    h = W0 x  +  (alpha / r) * B A x          W0: (out, in)  frozen
                                              A : (r, in)    trainable, random init
                                              B : (out, r)   trainable, ZERO init

Because B starts at zero, a freshly injected adapter changes nothing: the model starts
exactly at the pretrained function and training only learns the *difference*.
Trainable parameters per layer: r * (in + out) instead of in * out.

GPT-2 stores its projections as ``Conv1D`` (a linear layer whose weight is (in, out),
the transpose of ``nn.Linear``). ``Conv1D`` here is that layer, and ``LoRALinear`` wraps
either kind.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1D(nn.Module):
    """GPT-2's linear layer: y = x @ W + b with W of shape (in_features, out_features)."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.addmm(self.bias, x.reshape(-1, x.size(-1)), self.weight).view(*x.shape[:-1], -1)


def dense_weight(layer: nn.Module) -> torch.Tensor:
    """The (out, in) weight matrix of an nn.Linear or a GPT-2 Conv1D."""
    if isinstance(layer, nn.Linear):
        return layer.weight
    if isinstance(layer, Conv1D):
        return layer.weight.t()
    raise TypeError(f"unsupported layer {type(layer).__name__}")


def dims(layer: nn.Module) -> tuple[int, int]:
    if isinstance(layer, nn.Linear):
        return layer.in_features, layer.out_features
    if isinstance(layer, Conv1D):
        return layer.in_features, layer.out_features
    if hasattr(layer, "in_features") and hasattr(layer, "out_features"):
        return layer.in_features, layer.out_features
    raise TypeError(f"unsupported layer {type(layer).__name__}")


class LoRALinear(nn.Module):
    """Wraps a frozen base layer and adds a trainable rank-r update."""

    def __init__(self, base: nn.Module, r: int = 8, alpha: float = 16.0, dropout: float = 0.0) -> None:
        super().__init__()
        if r < 1:
            raise ValueError("rank r must be >= 1")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)                       # W0 is frozen
        self.in_features, self.out_features = dims(base)
        if r > min(self.in_features, self.out_features):
            raise ValueError(f"rank {r} exceeds min(in, out) = {min(self.in_features, self.out_features)}")
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))   # same init as nn.Linear / PEFT
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False
        self.enabled = True                                   # False -> behave like the base model

    # ------------------------------------------------------------------ forward
    def delta_weight(self) -> torch.Tensor:
        """ΔW = (alpha / r) · B A, shape (out, in)."""
        return (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.merged or not self.enabled:
            return y
        # Never materialise ΔW during training: (x A^T) B^T costs r·(in+out) per token.
        return y + (self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling

    # ------------------------------------------------------------------ merge / unmerge
    @torch.no_grad()
    def merge(self) -> None:
        """W0 <- W0 + ΔW. Afterwards inference costs exactly the same as the base model."""
        if self.merged:
            return
        if not hasattr(self.base, "weight") or self.base.weight.dtype not in (torch.float32, torch.float64,
                                                                              torch.bfloat16, torch.float16):
            raise TypeError("cannot merge into a quantized base layer; dequantize first")
        dense_weight(self.base).add_(self.delta_weight().to(self.base.weight.dtype))
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if not self.merged:
            return
        dense_weight(self.base).sub_(self.delta_weight().to(self.base.weight.dtype))
        self.merged = False

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, r={self.r}, alpha={self.alpha}, merged={self.merged}"


def to_plain(layer: LoRALinear) -> nn.Module:
    """Merge and return the bare base layer (the LoRA wrapper disappears completely)."""
    layer.merge()
    for p in layer.base.parameters():
        p.requires_grad_(False)
    return layer.base


def lora_param_count(in_features: int, out_features: int, r: int) -> int:
    return r * (in_features + out_features)


