"""Quantization arithmetic, written out by hand.

    q     = clamp(round(x / scale) + zero_point, q_min, q_max)       quantize
    x_hat = scale * (q - zero_point)                                 dequantize

Affine (asymmetric): the float range [min, max] (widened to include 0) is mapped onto
[q_min, q_max]. Symmetric: the range is [-a, a] with a = max(|min|, |max|), and the zero
point is fixed (0 for signed int8). These are the same formulas PyTorch's MinMaxObserver
uses; the tests check them against ``torch.quantize_per_tensor``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

EPS = torch.finfo(torch.float32).eps


def qrange(bits: int = 8, signed: bool = False) -> Tuple[int, int]:
    """Integer range: uint8 -> (0, 255), int8 -> (-128, 127), int4 -> (-8, 7)."""
    if signed:
        return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    return 0, 2 ** bits - 1


@dataclass
class QParams:
    scale: float
    zero_point: int
    q_min: int
    q_max: int

    @property
    def float_range(self) -> Tuple[float, float]:
        """The real values that are exactly representable at the two ends."""
        return (self.scale * (self.q_min - self.zero_point), self.scale * (self.q_max - self.zero_point))


def calc_qparams(x_min: float, x_max: float, bits: int = 8, symmetric: bool = False,
                 signed: bool | None = None) -> QParams:
    """Scale and zero point that map [x_min, x_max] onto the integer grid.

    The range is always widened to include 0, so that 0.0 (padding, ReLU zeros) is exactly
    representable: x = 0 quantizes to q = zero_point with no error.
    """
    if signed is None:
        signed = symmetric                    # PyTorch defaults: affine uint8, symmetric int8
    q_min, q_max = qrange(bits, signed)
    lo, hi = min(float(x_min), 0.0), max(float(x_max), 0.0)
    if symmetric:
        a = max(-lo, hi)
        scale = max(a / ((q_max - q_min) / 2), EPS)
        zero_point = 0 if signed else (q_min + q_max + 1) // 2
    else:
        scale = max((hi - lo) / (q_max - q_min), EPS)
        zero_point = int(q_min - torch.round(torch.tensor(lo / scale)).item())
        zero_point = max(q_min, min(q_max, zero_point))
    return QParams(float(scale), int(zero_point), q_min, q_max)


def quantize(x: torch.Tensor, p: QParams) -> torch.Tensor:
    """Float -> integer codes (returned as int32 to keep the arithmetic exact)."""
    q = torch.round(x / p.scale) + p.zero_point          # torch.round: ties go to the even integer
    return q.clamp(p.q_min, p.q_max).to(torch.int32)


def dequantize(q: torch.Tensor, p: QParams) -> torch.Tensor:
    return p.scale * (q.to(torch.float32) - p.zero_point)


def fake_quantize(x: torch.Tensor, p: QParams) -> torch.Tensor:
    """Quantize then dequantize: the float tensor QAT trains on. It carries the rounding error."""
    return dequantize(quantize(x, p), p)


def error_report(x: torch.Tensor, p: QParams) -> dict:
    """How much was lost: MSE, worst-case error, share of values clipped, and SQNR in dB."""
    x_hat = fake_quantize(x, p)
    err = x - x_hat
    lo, hi = p.float_range
    clipped = ((x < lo - p.scale / 2) | (x > hi + p.scale / 2)).float().mean()
    signal = x.pow(2).mean()
    noise = err.pow(2).mean()
    return {"mse": float(noise), "max_abs_error": float(err.abs().max()),
            "half_step": p.scale / 2, "clipped_share": float(clipped),
            "sqnr_db": float(10 * torch.log10(signal / noise)) if noise > 0 else float("inf")}


def per_channel_qparams(w: torch.Tensor, axis: int = 0, bits: int = 8) -> list:
    """Symmetric int8 parameters for each output channel (each row of a weight matrix)."""
    w = w.transpose(0, axis).reshape(w.shape[axis], -1)
    return [calc_qparams(float(r.min()), float(r.max()), bits, symmetric=True) for r in w]


def sqnr(reference: torch.Tensor, approx: torch.Tensor) -> float:
    """Signal-to-quantization-noise ratio in dB between two tensors (higher is better)."""
    noise = (reference.float() - approx.float()).pow(2).mean()
    return float(10 * torch.log10(reference.float().pow(2).mean() / noise)) if noise > 0 else float("inf")


def pick_engine() -> str:
    """The best quantized backend this PyTorch build supports, and make it the active one.

    x86 and fbgemm are the usual server engines; some builds (e.g. this project's Windows wheel)
    only ship oneDNN; qnnpack is the ARM/mobile engine.
    """
    supported = torch.backends.quantized.supported_engines
    for engine in ("x86", "fbgemm", "onednn", "qnnpack"):
        if engine in supported:
            torch.backends.quantized.engine = engine
            return engine
    raise RuntimeError(f"no quantized engine available (supported: {supported})")
