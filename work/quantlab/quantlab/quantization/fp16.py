"""FP16 (and bf16) on the CPU: does this machine support it, and does it help?

Half precision halves the weight storage. Whether it is *faster* depends on the hardware:
GPUs and some ARM/Apple chips have native fp16 math, but most x86 CPUs do not, so PyTorch
either runs a slower fallback or raises "not implemented for 'Half'". We try it for real and
record what happened; nothing is assumed.
"""

from __future__ import annotations

import copy
from typing import Tuple

import torch
import torch.nn as nn


def to_half(model: nn.Module, dtype: torch.dtype = torch.float16) -> nn.Module:
    return copy.deepcopy(model).eval().to(dtype)


def check_support(model: nn.Module, dtype: torch.dtype = torch.float16) -> Tuple[bool, str]:
    """Run one batch in `dtype` on the CPU. Returns (supported, reason)."""
    try:
        half = to_half(model, dtype)
        with torch.no_grad():
            out = half(torch.randn(4, 1, 28, 28, dtype=dtype))
        if out.dtype != dtype:
            return False, f"output came back as {out.dtype}, not {dtype}"
        if not torch.isfinite(out).all():
            return False, "non-finite outputs"
        return True, f"{dtype} forward pass ran on CPU"
    except (RuntimeError, NotImplementedError) as e:
        return False, f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
