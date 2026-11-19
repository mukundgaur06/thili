"""Quantization-aware training (QAT).

Post-training quantization rounds a model that never saw rounding. QAT instead inserts
*fake quantization* (quantize then immediately dequantize) on weights and activations
during training, so the forward pass already carries int8 rounding error and the weights
move to places where that error hurts less. Gradients pass through the rounding as if it
were the identity (the straight-through estimator).

Recipe used here, starting from the trained fp32 weights:

* fuse Conv+BN+ReLU in QAT mode (BN stays live), attach fake-quant modules;
* fine-tune for a few epochs at a small learning rate;
* from ``freeze_bn_epoch`` on, BN uses its running statistics (so folding at convert time
  matches training), and from ``freeze_observer_epoch`` on (never in the first epoch, so the
  observers have a range to keep), the ranges stop moving;
* convert to a real int8 model, exactly as static PTQ does.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from torch.ao.quantization import convert, get_default_qat_qconfig, prepare_qat

from ..data import Data
from ..train import train_epochs
from .utilities import pick_engine


@dataclass
class QATConfig:
    epochs: int = 2
    lr: float = 2e-4
    batch_size: int = 128
    freeze_bn_epoch: int = 2
    freeze_observer_epoch: int = 2
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def prepare_for_qat(model: nn.Module) -> nn.Module:
    engine = pick_engine()
    m = copy.deepcopy(model).train()
    m.fuse(qat=True)
    m.qconfig = get_default_qat_qconfig(engine)
    return prepare_qat(m)


def quantize_qat(model: nn.Module, data: Data, cfg: QATConfig | None = None,
                 on_epoch: Optional[Callable[[dict], None]] = None) -> Tuple[nn.Module, list]:
    cfg = cfg or QATConfig()
    torch.manual_seed(cfg.seed)
    qat_model = prepare_for_qat(model)

    def before_epoch(m: nn.Module, epoch: int) -> None:
        if epoch >= cfg.freeze_bn_epoch:
            m.apply(torch.ao.nn.intrinsic.qat.freeze_bn_stats)
        # Observers must see at least one epoch of data before they are frozen; frozen at the
        # start of epoch 1 they would hold no range at all and conversion would fail.
        if epoch >= cfg.freeze_observer_epoch and epoch > 1:
            m.apply(torch.ao.quantization.disable_observer)

    history = train_epochs(qat_model, data, cfg.epochs, cfg.lr, cfg.batch_size, seed=cfg.seed + 7,
                           on_epoch=on_epoch, before_epoch=before_epoch)
    return convert(qat_model.eval()), history
