"""Exact frequent-itemset mining: Apriori, level by level, on vertical bitsets.

* Level 1: count every item; keep those with count >= ceil(min_support * N).
* Level k+1: join two frequent k-itemsets that share their first k-1 items (both sorted),
  and **prune** the candidate unless every k-subset is frequent (the Apriori property: an
  itemset can only be frequent if all its subsets are). Count the survivors with one
  bitset AND + popcount each, vectorised over all candidates that share a prefix.

The result is exact: every itemset of at most `max_size` items with support >= min_support,
with its exact count. Tests compare it with brute-force enumeration.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Dict, FrozenSet, Sequence

import numpy as np

from .bitsets import build_bitsets, popcount


def min_count_for(min_support: float, n: int) -> int:
    return max(1, math.ceil(min_support * n - 1e-9))


def apriori(baskets: Sequence[Sequence[int]], n_items: int, min_support: float, max_size: int = 5,
            stats: dict | None = None) -> Dict[FrozenSet[int], int]:
    """{itemset: count} for every frequent itemset. `stats` (optional) collects per-level counts."""
    n = len(baskets)
    need = min_count_for(min_support, n)
    bits = build_bitsets(baskets, n_items)
    counts1 = popcount(bits)
    level = {(int(i),): bits[i] for i in np.flatnonzero(counts1 >= need)}
    found: Dict[FrozenSet[int], int] = {frozenset(k): int(counts1[k[0]]) for k in level}
    if stats is not None:
        stats.setdefault("levels", []).append({"size": 1, "candidates": n_items, "frequent": len(level)})
    size = 1
    while level and size < max_size:
        by_prefix: Dict[tuple, list] = {}
        for items in sorted(level):
            by_prefix.setdefault(items[:-1], []).append(items[-1])
        nxt = {}
        n_candidates = 0
        for prefix, lasts in by_prefix.items():
            for j, a in enumerate(lasts[:-1]):
                partners = []
                for b in lasts[j + 1:]:
                    cand = prefix + (a, b)
                    # Apriori pruning: every subset of size `size` must be frequent. The two that
                    # formed the candidate are frequent by construction; check the others.
                    if size >= 2 and any(cand[:i] + cand[i + 1:] not in level for i in range(len(cand) - 2)):
                        continue
                    partners.append(b)
                if not partners:
                    continue
                n_candidates += len(partners)
                base = level[prefix + (a,)]
                block = base[None, :] & bits[np.array(partners)]
                cnt = popcount(block)
                for b, c, row in zip(partners, cnt, block):
                    if c >= need:
                        nxt[prefix + (a, b)] = row
                        found[frozenset(prefix + (a, b))] = int(c)
        size += 1
        if stats is not None:
            stats["levels"].append({"size": size, "candidates": n_candidates, "frequent": len(nxt)})
        level = nxt
    return found


def brute_force(baskets: Sequence[Sequence[int]], min_support: float, max_size: int = 5) -> Dict[FrozenSet[int], int]:
    """Enumerate every sub-itemset of every basket and count it. Exponential: tests only."""
    n = len(baskets)
    need = min_count_for(min_support, n)
    counts: Dict[FrozenSet[int], int] = {}
    for b in baskets:
        items = sorted(set(b))
        for k in range(1, min(max_size, len(items)) + 1):
            for c in combinations(items, k):
                key = frozenset(c)
                counts[key] = counts.get(key, 0) + 1
    return {k: v for k, v in counts.items() if v >= need}


def supports(counts: Dict[FrozenSet[int], int], n: int) -> Dict[FrozenSet[int], float]:
    return {k: v / n for k, v in counts.items()}
