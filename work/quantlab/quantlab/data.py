"""Fashion-MNIST (or MNIST) as uint8 tensors, a leakage-free validation split, and batches.

Images stay as uint8 in memory (47 MB for all 70,000) and are normalised one batch at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Tuple

import torch

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# Mean and std of the training pixels (in [0, 1]); computed once by `pixel_stats`, then fixed.
STATS = {"fashion_mnist": (0.2860, 0.3530), "mnist": (0.1307, 0.3081)}
CLASSES = {
    "fashion_mnist": ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                      "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
    "mnist": [str(i) for i in range(10)],
}


@dataclass
class Split:
    x: torch.Tensor          # uint8, (N, 28, 28)
    y: torch.Tensor          # int64, (N,)

    def __len__(self) -> int:
        return len(self.y)


@dataclass
class Data:
    name: str
    train: Split
    val: Split
    test: Split
    classes: List[str]
    mean: float
    std: float

    def normalise(self, x_u8: torch.Tensor) -> torch.Tensor:
        return ((x_u8.float() / 255.0) - self.mean).div_(self.std).unsqueeze(1)

    def batches(self, split: Split, batch_size: int, shuffle: bool = False,
                seed: int = 0) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        n = len(split)
        order = torch.randperm(n, generator=torch.Generator().manual_seed(seed)) if shuffle else torch.arange(n)
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            yield self.normalise(split.x[idx]), split.y[idx]

    def calibration_set(self, n_images: int, seed: int = 0) -> torch.Tensor:
        """Normalised images drawn from the *training* split only, never from val or test."""
        idx = torch.randperm(len(self.train), generator=torch.Generator().manual_seed(seed))[:n_images]
        return self.normalise(self.train.x[idx])


def load(name: str = "fashion_mnist", val_size: int = 5000, seed: int = 0, root: Path = DATA_DIR) -> Data:
    """Official train split -> train + validation (seeded); official test split untouched."""
    from torchvision import datasets
    cls = {"fashion_mnist": datasets.FashionMNIST, "mnist": datasets.MNIST}[name]
    tr = cls(str(root), train=True, download=True)
    te = cls(str(root), train=False, download=True)
    perm = torch.randperm(len(tr.targets), generator=torch.Generator().manual_seed(seed))
    val_idx, train_idx = perm[:val_size], perm[val_size:]
    mean, std = STATS[name]
    return Data(name=name,
                train=Split(tr.data[train_idx], tr.targets[train_idx]),
                val=Split(tr.data[val_idx], tr.targets[val_idx]),
                test=Split(te.data, te.targets),
                classes=CLASSES[name], mean=mean, std=std)


def pixel_stats(x_u8: torch.Tensor) -> Tuple[float, float]:
    x = x_u8.float() / 255.0
    return float(x.mean()), float(x.std())


def synthetic(n_train: int = 512, n_test: int = 256, seed: int = 0) -> Data:
    """A tiny random dataset with the same shapes, for tests that must not touch the disk."""
    g = torch.Generator().manual_seed(seed)

    def split(n):
        y = torch.randint(0, 10, (n,), generator=g)
        x = torch.randint(0, 60, (n, 28, 28), generator=g, dtype=torch.uint8)
        for k in range(10):                         # a bright block whose position encodes the class
            r, c = divmod(k, 5)
            x[y == k, 4 + r * 10: 12 + r * 10, 2 + c * 5: 6 + c * 5] = 220
        return Split(x, y)

    return Data("synthetic", split(n_train), split(n_test // 2), split(n_test), [str(i) for i in range(10)], 0.2, 0.3)
