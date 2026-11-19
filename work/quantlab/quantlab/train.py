"""Training and validation loops, and checkpoints."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import Data, Split
from .model import ModelConfig, QuantCNN


@dataclass
class TrainConfig:
    epochs: int = 8
    batch_size: int = 128
    lr: float = 2e-3
    weight_decay: float = 1e-4
    seed: int = 0
    threads: int = 4


@torch.no_grad()
def validate(model: nn.Module, data: Data, split: Split, batch_size: int = 512) -> dict:
    model.eval()
    loss = correct = 0.0
    for x, y in data.batches(split, batch_size):
        out = model(x)
        loss += float(F.cross_entropy(out, y, reduction="sum"))
        correct += int((out.argmax(1) == y).sum())
    return {"loss": loss / len(split), "accuracy": correct / len(split)}


def train_epochs(model: nn.Module, data: Data, epochs: int, lr: float, batch_size: int = 128,
                 weight_decay: float = 1e-4, seed: int = 0,
                 on_epoch: Optional[Callable[[dict], None]] = None,
                 on_batch: Optional[Callable[[int, int], None]] = None,
                 before_epoch: Optional[Callable[[nn.Module, int], None]] = None) -> List[dict]:
    """AdamW with a cosine learning-rate schedule. Returns one row per epoch.

    `before_epoch(model, epoch)` lets QAT freeze observers / BatchNorm statistics late in training.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = math.ceil(len(data.train) / batch_size)
    total = epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    history = []
    step = 0
    for epoch in range(1, epochs + 1):
        if before_epoch:
            before_epoch(model, epoch)
        model.train()
        t0 = time.time()
        run_loss = run_correct = seen = 0
        for x, y in data.batches(data.train, batch_size, shuffle=True, seed=seed * 1000 + epoch):
            out = model(x)
            loss = F.cross_entropy(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            run_loss += loss.item() * len(y)
            run_correct += int((out.argmax(1) == y).sum())
            seen += len(y)
            if on_batch:
                on_batch(step, total)
        val = validate(model, data, data.val)
        row = {"epoch": epoch, "train_loss": run_loss / seen, "train_accuracy": run_correct / seen,
               "val_loss": val["loss"], "val_accuracy": val["accuracy"], "seconds": round(time.time() - t0, 1),
               "lr": sched.get_last_lr()[0]}
        history.append(row)
        if on_epoch:
            on_epoch(row)
    return history


def train_baseline(data: Data, model_cfg: ModelConfig | None = None, cfg: TrainConfig | None = None,
                   **callbacks) -> tuple:
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.threads)
    model = QuantCNN(model_cfg)
    history = train_epochs(model, data, cfg.epochs, cfg.lr, cfg.batch_size, cfg.weight_decay, cfg.seed, **callbacks)
    return model.eval(), history


def save_checkpoint(path: Path, model: QuantCNN, history: list, train_cfg: TrainConfig, dataset: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_config": model.cfg.to_dict(),
                "train_config": asdict(train_cfg), "history": history, "dataset": dataset}, path)


def load_checkpoint(path: Path) -> tuple:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = QuantCNN(ModelConfig(**ck["model_config"]))
    model.load_state_dict(ck["state_dict"])
    return model.eval(), ck
