"""Accuracy, loss, per-class metrics and the confusion matrix, for any model variant."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import Data, Split


@torch.no_grad()
def predict(model: nn.Module, data: Data, split: Split, batch_size: int = 512,
            dtype: torch.dtype = torch.float32) -> Dict[str, np.ndarray]:
    model.eval()
    logits = []
    for x, _ in data.batches(split, batch_size):
        logits.append(model(x.to(dtype)).float())
    out = torch.cat(logits)
    return {"logits": out.numpy(), "pred": out.argmax(1).numpy(), "y": split.y.numpy(),
            "loss": float(F.cross_entropy(out, split.y))}


def confusion_matrix(y: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    return cm


def per_class(cm: np.ndarray, classes: List[str]) -> List[dict]:
    rows = []
    for k, name in enumerate(classes):
        tp = cm[k, k]
        prec = tp / cm[:, k].sum() if cm[:, k].sum() else 0.0
        rec = tp / cm[k].sum() if cm[k].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({"class": name, "precision": float(prec), "recall": float(rec), "f1": float(f1),
                     "support": int(cm[k].sum())})
    return rows


def evaluate(model: nn.Module, data: Data, split: Split | None = None,
             dtype: torch.dtype = torch.float32) -> dict:
    split = split if split is not None else data.test
    p = predict(model, data, split, dtype=dtype)
    cm = confusion_matrix(p["y"], p["pred"], len(data.classes))
    return {"accuracy": float((p["pred"] == p["y"]).mean()), "loss": p["loss"],
            "confusion_matrix": cm.tolist(), "per_class": per_class(cm, data.classes),
            "pred": p["pred"].tolist()}


def agreement(pred_a, pred_b) -> float:
    """Share of test images on which two models predict the same class."""
    return float((np.asarray(pred_a) == np.asarray(pred_b)).mean())
