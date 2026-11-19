"""Post-training static quantization: weights *and* activations become int8.

Steps (eager mode):

1. **Fuse** Conv+BN+ReLU and Linear+ReLU (BN is folded into the conv weights).
2. **Prepare**: attach an observer after every quantized op. Observers do nothing to the
   values; they record the range of what flows past.
3. **Calibrate**: run real training images through the model. Every observer now holds a
   measured activation range. Nothing here is made up: the ranges come from the data.
4. **Convert**: from each range compute (scale, zero point) and swap every module for its
   int8 kernel. From now on activations travel between layers as uint8.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.ao.quantization import QConfig, convert, prepare
from torch.ao.quantization.observer import (HistogramObserver, MinMaxObserver, MovingAverageMinMaxObserver,
                                            ObserverBase, PerChannelMinMaxObserver)

from .utilities import pick_engine

ACTIVATION_OBSERVERS = {"minmax": MinMaxObserver, "moving_average": MovingAverageMinMaxObserver,
                        "histogram": HistogramObserver}


@dataclass
class StaticConfig:
    observer: str = "histogram"          # how activation ranges are estimated
    per_channel: bool = True             # one weight scale per output channel, or one per tensor
    calibration_images: int = 1024
    calibration_batch: int = 128
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def make_qconfig(cfg: StaticConfig) -> QConfig:
    act = ACTIVATION_OBSERVERS[cfg.observer].with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine,
                                                       reduce_range=False)
    if cfg.per_channel:
        weight = PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
    else:
        weight = MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
    return QConfig(activation=act, weight=weight)


def prepare_static(model: nn.Module, cfg: StaticConfig) -> nn.Module:
    pick_engine()
    m = copy.deepcopy(model).eval()
    m.fuse(qat=False)
    m.qconfig = make_qconfig(cfg)
    return prepare(m)


@torch.no_grad()
def calibrate(prepared: nn.Module, images: torch.Tensor, batch_size: int = 128) -> int:
    for i in range(0, len(images), batch_size):
        prepared(images[i:i + batch_size])
    return len(images)


def observer_stats(prepared: nn.Module) -> List[dict]:
    """What the observers measured during calibration, and the (scale, zero point) it implies."""
    rows = []
    for name, m in prepared.named_modules():
        if isinstance(m, ObserverBase) and name.endswith("activation_post_process"):
            scale, zp = m.calculate_qparams()
            layer = name.rsplit(".activation_post_process", 1)[0]
            rows.append({"layer": layer, "tensor": "output activation", "observer": type(m).__name__,
                         "min": float(m.min_val), "max": float(m.max_val),
                         "scale": float(scale.flatten()[0]), "zero_point": int(zp.flatten()[0])})
    for name, m in prepared.named_modules():
        w = getattr(m, "weight", None)
        if isinstance(w, torch.Tensor) and hasattr(m, "qconfig") and m.qconfig is not None and w.dim() > 1:
            obs = m.qconfig.weight()
            obs(w.detach())
            scale, zp = obs.calculate_qparams()
            rows.append({"layer": name, "tensor": f"weight ({'per-channel' if scale.numel() > 1 else 'per-tensor'})",
                         "observer": type(obs).__name__, "min": float(w.detach().min()), "max": float(w.detach().max()),
                         "scale": float(scale.mean()), "zero_point": int(zp.flatten()[0])})
    return rows


def convert_static(prepared: nn.Module) -> nn.Module:
    return convert(prepared.eval())


def quantize_static(model: nn.Module, calibration_images: torch.Tensor,
                    cfg: StaticConfig | None = None) -> Tuple[nn.Module, List[dict]]:
    cfg = cfg or StaticConfig()
    prepared = prepare_static(model, cfg)
    calibrate(prepared, calibration_images, cfg.calibration_batch)
    stats = observer_stats(prepared)
    return convert_static(prepared), stats
