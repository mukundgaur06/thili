"""Post-training dynamic quantization.

Weights are converted to int8 once, ahead of time. Activations stay float in memory; at run
time each Linear computes a scale for its *current* input, quantizes it, does an int8 matmul,
and dequantizes the result. No calibration data is needed.

PyTorch supports it for Linear and recurrent layers only, not Conv2d, so in this CNN the two
convolutions stay fp32. The Linear layers hold about 97% of the parameters, so most of the
size saving still happens.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .utilities import pick_engine

SUPPORTED = (nn.Linear,)


def quantize_dynamic(model: nn.Module) -> nn.Module:
    pick_engine()
    return torch.ao.quantization.quantize_dynamic(copy.deepcopy(model).eval(), set(SUPPORTED), dtype=torch.qint8)


def quantized_layers(qmodel: nn.Module) -> dict:
    """Which layers ended up int8 and which stayed float."""
    out = {"int8": [], "float": []}
    for name, m in qmodel.named_modules():
        kind = type(m).__module__
        if isinstance(m, (nn.Conv2d, nn.Linear)) or "quantized" in kind:
            if "quantized" in kind and hasattr(m, "weight"):
                out["int8"].append(f"{name} ({type(m).__name__})")
            elif isinstance(m, (nn.Conv2d, nn.Linear)):
                out["float"].append(f"{name} ({type(m).__name__})")
    return out
