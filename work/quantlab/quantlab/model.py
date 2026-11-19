"""A small CNN written so that PyTorch's eager-mode quantization can handle it.

    input 1x28x28
      QuantStub                      (fp32 -> int8 boundary; identity until converted)
      Conv 3x3 (1->c1) + BN + ReLU   -> MaxPool 2   (c1x14x14)
      Conv 3x3 (c1->c2) + BN + ReLU  -> MaxPool 2   (c2x7x7)
      Flatten -> Linear (c2*49 -> hidden) + ReLU -> Dropout -> Linear (hidden -> 10)
      DeQuantStub                    (int8 -> fp32 boundary)

Quantization-friendly choices: every ReLU is its own module (so Conv+BN+ReLU and Linear+ReLU can
be fused into one quantized kernel), there is no functional op between layers, and the
quantization boundaries are explicit stubs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List

import torch
import torch.nn as nn
from torch.ao.quantization import DeQuantStub, QuantStub

FUSE_GROUPS = [["conv1", "bn1", "relu1"], ["conv2", "bn2", "relu2"], ["fc1", "relu3"]]


@dataclass
class ModelConfig:
    channels: List[int] = field(default_factory=lambda: [16, 32])
    hidden: int = 128
    dropout: float = 0.2
    num_classes: int = 10

    def to_dict(self) -> dict:
        return asdict(self)


class QuantCNN(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or ModelConfig()
        self.cfg = cfg
        c1, c2 = cfg.channels
        self.quant = QuantStub()
        self.conv1 = nn.Conv2d(1, c1, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(c2 * 7 * 7, cfg.hidden)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc2 = nn.Linear(cfg.hidden, cfg.num_classes)
        self.dequant = DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.dropout(self.relu3(self.fc1(self.flatten(x))))
        return self.dequant(self.fc2(x))

    def fuse(self, qat: bool = False) -> "QuantCNN":
        """Fold BatchNorm into the convolution and merge ReLUs, in place.

        For post-training quantization the model must be in eval mode (BN is folded into the
        weights). For QAT it must be in train mode and BN stays live inside the fused module.
        """
        from torch.ao.quantization import fuse_modules, fuse_modules_qat
        (fuse_modules_qat if qat else fuse_modules)(self, FUSE_GROUPS, inplace=True)
        return self


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def layer_table(model: nn.Module) -> List[dict]:
    """One row per layer with weights: name, type, shape, parameter count."""
    rows = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            n = sum(p.numel() for p in m.parameters(recurse=False))
            rows.append({"layer": name, "type": type(m).__name__,
                         "weight shape": tuple(m.weight.shape), "parameters": n})
    return rows
