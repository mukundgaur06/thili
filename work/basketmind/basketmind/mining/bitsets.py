"""Vertical bitsets: for every item, one bit per basket.

Basket membership is stored item-wise ("vertically"). Item i's bitset is a uint64 array of
ceil(N / 64) words whose bit b is 1 when basket b holds item i. The support count of an
itemset is then the population count of the AND of its items' bitsets:

    count({a, b, c}) = popcount(bits[a] & bits[b] & bits[c])

For 100,000 baskets that is 1,563 words per item, and one AND plus one popcount instruction
per word. ``np.bitwise_count`` (NumPy >= 2.0) does the popcount; older NumPy falls back to a
byte lookup table.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def popcount(words: np.ndarray, axis: int = -1) -> np.ndarray:
    """Number of set bits along `axis` of a uint64 array."""
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(words).sum(axis=axis, dtype=np.int64)
    return _POP8[words.view(np.uint8)].reshape(*words.shape, 8).sum(axis=(axis, -1) if axis == -1 else axis, dtype=np.int64)


def build_bitsets(baskets: Sequence[Sequence[int]], n_items: int) -> np.ndarray:
    """A (n_items, n_words) uint64 matrix of membership bits."""
    n = len(baskets)
    n_words = (n + 63) // 64
    member = np.zeros((n_items, n_words * 64), dtype=bool)
    rows = np.concatenate([np.asarray(b, dtype=np.int64) for b in baskets]) if n else np.zeros(0, np.int64)
    cols = np.repeat(np.arange(n), [len(b) for b in baskets])
    member[rows, cols] = True
    packed = np.packbits(member, axis=1, bitorder="little")          # bit b of byte k = basket 8k + b
    return packed.view(np.uint64).reshape(n_items, n_words)


def support_count(bits: np.ndarray, items: Sequence[int]) -> int:
    acc = bits[items[0]].copy()
    for i in items[1:]:
        acc &= bits[i]
    return int(popcount(acc))
