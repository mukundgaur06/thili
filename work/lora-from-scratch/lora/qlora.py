"""QLoRA from scratch: a 4-bit NormalFloat (NF4) frozen base + LoRA in full precision.

QLoRA (Dettmers et al., 2023) observes that the frozen base never receives gradients, so it
can be stored in 4 bits. Three ideas:

1. NF4 data type. Pretrained weights are roughly normally distributed. Instead of 16 evenly
   spaced levels (int4), NF4 places its 16 levels at quantiles of N(0, 1), so each level is
   used about equally often. That is information-theoretically optimal for normal data.
2. Block-wise absmax scaling. Weights are cut into blocks of 64. Each block is divided by
   its own absolute maximum, so one outlier only hurts its own block.
3. Double quantization. The fp32 scale per 64 weights costs 0.5 bit/param. Quantizing the
   scales themselves to 8 bits (blocks of 256) brings that down to about 0.127 bit/param.

(The 4th idea, paged optimizers, moves optimizer state to CPU RAM on GPU memory spikes. It
is a CUDA unified-memory feature and has no meaning on this CPU-only workbench.)

Compute happens in fp32: the 4-bit weight is dequantized on the fly inside ``forward``.
Gradients flow through the dequantized weight into x (for earlier LoRA layers) but never
into the codes, which are integer buffers.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Conv1D, LoRALinear, dense_weight

# The table published with QLoRA / bitsandbytes, used to check our construction.
NF4_REFERENCE = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
    -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
    0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
]


def nf4_levels(offset: float = 0.9677083) -> torch.Tensor:
    """Build the 16 NF4 levels from normal quantiles (asymmetric, so 0 is exactly representable).

    8 positive levels from quantiles in [0.5, offset] and 7 negative ones, plus an exact 0,
    then rescaled to [-1, 1]. ``offset`` = 1/2 * (1 - 1/(2*15) + 1 - 1/(2*16)) avoids ±infinity.
    """
    nd = NormalDist()
    pos = [nd.inv_cdf(0.5 + (offset - 0.5) * i / 8) for i in range(1, 9)]      # 8 values
    neg = [-nd.inv_cdf(0.5 + (offset - 0.5) * i / 7) for i in range(1, 8)]     # 7 values
    v = torch.tensor(sorted(neg + [0.0] + pos), dtype=torch.float32)
    return v / v.abs().max()


INT4_LEVELS = torch.linspace(-1, 1, 16)   # the naive alternative, for comparison


def quantize_blockwise(w: torch.Tensor, levels: torch.Tensor, block: int = 64):
    """-> (codes uint8 in [0,15] with shape (n_blocks, block), absmax fp32 (n_blocks,), numel)."""
    flat = w.detach().float().reshape(-1)
    n = flat.numel()
    pad = (-n) % block
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.view(-1, block)
    absmax = blocks.abs().amax(dim=1).clamp_min(1e-12)
    normed = blocks / absmax[:, None]                                 # in [-1, 1]
    # nearest level: midpoints between sorted levels are the decision boundaries
    mids = (levels[1:] + levels[:-1]) / 2
    codes = torch.bucketize(normed, mids).to(torch.uint8)
    return codes, absmax, n


def dequantize_blockwise(codes: torch.Tensor, absmax: torch.Tensor, levels: torch.Tensor,
                         numel: int, shape) -> torch.Tensor:
    return (levels[codes.long()] * absmax[:, None]).reshape(-1)[:numel].view(shape)


def pack_4bit(codes: torch.Tensor) -> torch.Tensor:
    flat = codes.reshape(-1)
    if flat.numel() % 2:
        flat = torch.cat([flat, flat.new_zeros(1)])
    return (flat[0::2] << 4 | flat[1::2]).to(torch.uint8)            # two codes per byte


def unpack_4bit(packed: torch.Tensor, n: int) -> torch.Tensor:
    out = torch.stack([packed >> 4, packed & 0xF], dim=1).reshape(-1)
    return out[:n]


def quantize_absmax_8bit(x: torch.Tensor, block: int = 256):
    """Double quantization of the per-block scales: subtract the mean, int8 absmax per 256."""
    mean = x.mean()
    c = x - mean
    pad = (-c.numel()) % block
    cp = torch.cat([c, c.new_zeros(pad)]) if pad else c
    b = cp.view(-1, block)
    s = b.abs().amax(1).clamp_min(1e-12) / 127
    q = torch.round(b / s[:, None]).clamp(-127, 127).to(torch.int8)
    return q, s, mean, c.numel()


def dequantize_absmax_8bit(q, s, mean, n):
    return (q.float() * s[:, None]).reshape(-1)[:n] + mean


class Linear4bit(nn.Module):
    """A frozen linear layer stored as packed NF4 (or int4) codes."""

    def __init__(self, layer: nn.Module, dtype: str = "nf4", block: int = 64,
                 double_quant: bool = True) -> None:
        super().__init__()
        w = dense_weight(layer).detach()                               # (out, in)
        self.out_features, self.in_features = w.shape
        self.dtype_name, self.block, self.double_quant = dtype, block, double_quant
        levels = nf4_levels() if dtype == "nf4" else INT4_LEVELS.clone()
        codes, absmax, n = quantize_blockwise(w, levels, block)
        self.register_buffer("levels", levels)
        self.register_buffer("packed", pack_4bit(codes))
        self.numel = n
        if double_quant:
            q, s, mean, na = quantize_absmax_8bit(absmax)
            self.register_buffer("absmax_q", q)
            self.register_buffer("absmax_s", s)
            self.register_buffer("absmax_mean", mean.reshape(1))
            self.n_absmax = na
        else:
            self.register_buffer("absmax", absmax)
        bias = getattr(layer, "bias", None)
        self.register_buffer("bias", None if bias is None else bias.detach().clone())

    def scales(self) -> torch.Tensor:
        if self.double_quant:
            return dequantize_absmax_8bit(self.absmax_q, self.absmax_s, self.absmax_mean[0], self.n_absmax)
        return self.absmax

    def weight_fp32(self) -> torch.Tensor:
        codes = unpack_4bit(self.packed, self.numel + (-self.numel) % self.block).view(-1, self.block)
        return dequantize_blockwise(codes, self.scales(), self.levels, self.numel,
                                    (self.out_features, self.in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight_fp32(), self.bias)

    def storage_bytes(self) -> int:
        b = self.packed.numel()
        b += (self.absmax_q.numel() + 4 * self.absmax_s.numel() + 4) if self.double_quant else 4 * self.absmax.numel()
        return b + (0 if self.bias is None else 4 * self.bias.numel())  # bias stays fp32

    def extra_repr(self) -> str:
        return f"{self.in_features}->{self.out_features}, {self.dtype_name}, block={self.block}, double_quant={self.double_quant}"


def quantize_model(model: nn.Module, targets: Iterable[str] = ("c_attn", "c_proj", "c_fc"),
                   dtype: str = "nf4", double_quant: bool = True) -> List[str]:
    """Replace frozen linear layers (including the base inside LoRALinear) by 4-bit ones."""
    targets = tuple(targets)
    done = []
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear) and isinstance(module.base, (nn.Linear, Conv1D)):
            module.base = Linear4bit(module.base, dtype, double_quant=double_quant)
            done.append(name)
        elif isinstance(module, (nn.Linear, Conv1D)) and name.split(".")[-1] in targets:
            parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
            if isinstance(parent, LoRALinear):
                continue
            setattr(parent, name.rsplit(".", 1)[-1], Linear4bit(module, dtype, double_quant=double_quant))
            done.append(name)
    return done


def model_storage_bytes(model: nn.Module) -> int:
    total = 0
    for m in model.modules():
        if isinstance(m, Linear4bit):
            total += m.storage_bytes()
    seen = {id(b) for m in model.modules() if isinstance(m, Linear4bit) for b in m.buffers()}
    for p in model.parameters():
        total += p.numel() * p.element_size()
    for b in model.buffers():
        if id(b) not in seen and b.dtype != torch.bool:
            total += b.numel() * b.element_size()
    return total
