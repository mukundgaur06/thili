"""Shopping baskets with itemsets planted at known supports.

1. **Background baskets.** A shopper picks a number of different base products (2 plus a
   negative binomial, mean 6), drawn without replacement by popularity, then one SKU of each.
   A basket never holds two SKUs of the same base product. Nothing is related to anything:
   items meet only because they are popular.
2. **Planting.** For each planted itemset S with target support t, exactly round(t * N)
   baskets end up holding all of S. Baskets that already hold S by chance count; just enough
   other baskets are picked at random and get S added (replacing any other SKU of the same base
   product). Larger itemsets are planted first; overlapping itemsets share SKUs, and no planted
   itemset may contain another.
3. **Ground truth.** The planted itemsets, their target and realised supports, the probability
   with which a basket was chosen for planting, and their categories are written to a *separate*
   file. Models only ever see the baskets.

Everything is seeded: the same seed gives the same shop, the same baskets, the same truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .catalogue import Product, popularity

# Product groups that genuinely go together, by size. Base products, not SKUs.
NATURAL: Dict[int, List[Tuple[str, ...]]] = {
    2: [("Bread", "Butter"), ("Coffee", "Milk"), ("Pasta", "Pasta Sauce"), ("Smartphone", "Phone Case"),
        ("Laptop", "Wireless Mouse"), ("Toothpaste", "Toothbrush")],
    3: [("Tortillas", "Salsa", "Avocados"), ("Shampoo", "Conditioner", "Body Wash"),
        ("Running Shoes", "Sports Socks", "Water Bottle"), ("Coffee Maker", "Coffee Filters", "Coffee")],
    4: [("Burger Buns", "Cheddar Cheese", "Tomatoes", "Onions"), ("Laptop", "Laptop Bag", "Keyboard", "USB Cable"),
        ("Yoga Mat", "Gym Towel", "Water Bottle", "Protein Bar"), ("Novel", "Bookmark", "Tea", "Chocolate Bar")],
    5: [("Cereal", "Milk", "Bananas", "Orange Juice", "Eggs"), ("Flour", "Sugar", "Eggs", "Butter", "Yogurt"),
        ("Smartphone", "Screen Protector", "Power Bank", "Headphones", "USB Cable"),
        ("Rain Jacket", "Umbrella", "Socks", "Baseball Cap", "Jeans")],
}
SUPPORTS = (0.03, 0.01, 0.004, 0.0015)


@dataclass(frozen=True)
class Planted:
    items: Tuple[int, ...]          # product ids, sorted
    target_support: float
    kind: str                        # "natural" or "surprising"
    bases: Tuple[str, ...] = ()
    categories: Tuple[str, ...] = ()


@dataclass
class GeneratorConfig:
    n_baskets: int = 100_000
    seed: int = 0
    mean_size: float = 6.0
    min_size: int = 2
    dispersion: float = 4.0
    supports: Tuple[float, ...] = SUPPORTS
    max_chance_fraction: float = 0.3

    def to_dict(self) -> dict:
        return asdict(self)


def basket_sizes(n: int, rng, mean: float, minimum: int, dispersion: float) -> np.ndarray:
    extra = mean - minimum
    p = dispersion / (dispersion + extra)
    return minimum + rng.negative_binomial(dispersion, p, size=n)


class BaseIndex:
    """Base products, their shares, and a fast 'pick one SKU of this base' lookup."""

    def __init__(self, catalogue: List[Product], shares: np.ndarray):
        groups: Dict[str, List[int]] = {}
        for p in catalogue:
            groups.setdefault(p.base, []).append(p.pid)
        self.bases = list(groups)
        self.members = [np.array(groups[b]) for b in self.bases]
        self.share = np.array([shares[m].sum() for m in self.members])
        self.within = [shares[m] / shares[m].sum() for m in self.members]
        self.base_of = {p.pid: p.base for p in catalogue}

    def pick_skus(self, base_ids: np.ndarray, rng) -> np.ndarray:
        return np.array([rng.choice(self.members[b], p=self.within[b]) for b in base_ids])


def background_baskets(catalogue: List[Product], cfg: GeneratorConfig, seed: int) -> List[List[int]]:
    rng = np.random.default_rng(seed)
    idx = BaseIndex(catalogue, popularity(catalogue))
    sizes = np.minimum(basket_sizes(cfg.n_baskets, rng, cfg.mean_size, cfg.min_size, cfg.dispersion), len(idx.bases))
    log_share = np.log(idx.share)
    baskets = []
    for start in range(0, cfg.n_baskets, 4000):           # Gumbel top-k = sampling without replacement
        sz = sizes[start:start + 4000]
        scores = log_share + rng.gumbel(size=(len(sz), len(idx.bases)))
        order = np.argsort(-scores, axis=1)
        for row, k in zip(order, sz):
            baskets.append(sorted(int(x) for x in idx.pick_skus(row[:k], rng)))
    return baskets


def _holders(baskets: Sequence[Sequence[int]], items: Sequence[int]) -> np.ndarray:
    want = set(items)
    return np.array([want <= set(b) for b in baskets])


def plant(baskets: List[List[int]], p: Planted, base_of: Dict[int, str], rng) -> Tuple[List[List[int]], dict]:
    """Make exactly round(target * N) baskets hold all of p.items (or leave as is if already above)."""
    n = len(baskets)
    target = round(p.target_support * n)
    holds = _holders(baskets, p.items)
    already = int(holds.sum())
    info = {"chance_count": already, "target_count": target, "planting_probability": 0.0}
    if already >= target:
        return baskets, info
    pick = rng.choice(np.flatnonzero(~holds), size=target - already, replace=False)
    info["planting_probability"] = (target - already) / max(1, n - already)
    bases = {base_of[i] for i in p.items}
    out = list(baskets)
    for i in pick:
        kept = [x for x in out[i] if base_of[x] not in bases]
        out[i] = sorted(set(kept) | set(p.items))
    return out, info


def choose_planted(catalogue: List[Product], shares: np.ndarray, sample: List[List[int]], cfg: GeneratorConfig,
                   seed: int) -> List[Planted]:
    """Natural and surprising itemsets of sizes 2 to 5 at each support level.

    A base product uses one SKU in every itemset it appears in (so overlaps are deliberate).
    Rules: no itemset contains another; chance alone (measured on a background sample) must
    give at most `max_chance_fraction` of the target support; among the natural groups of one
    size, the one most common by chance gets the highest support.
    """
    rng = np.random.default_rng(seed)
    by_base: Dict[str, List[int]] = {}
    for p in catalogue:
        by_base.setdefault(p.base, []).append(p.pid)
    cat_of = {p.base: p.category for p in catalogue}
    sku: Dict[str, int] = {}

    def sku_for(base: str) -> int:
        if base not in sku:
            ids = np.array(by_base[base])
            sku[base] = int(rng.choice(ids, p=shares[ids] / shares[ids].sum()))
        return sku[base]

    index: Dict[int, set] = {}
    for k, b in enumerate(sample):
        for x in b:
            index.setdefault(x, set()).add(k)

    def chance(items) -> float:
        sets = sorted((index.get(i, set()) for i in items), key=len)
        return len(set.intersection(*sets)) / len(sample)

    def fits(items, support, existing) -> bool:
        if chance(items) > cfg.max_chance_fraction * support:
            return False
        s = set(items)
        return not any(s <= set(e.items) or set(e.items) <= s for e in existing)

    planted: List[Planted] = []
    supports = sorted(cfg.supports, reverse=True)
    for size in sorted(NATURAL, reverse=True):
        groups = [tuple(sorted(sku_for(b) for b in g)) for g in NATURAL[size]]
        groups.sort(key=lambda g: -chance(g))
        for support, items in zip(supports, groups):
            if not fits(items, support, planted):
                raise ValueError(f"natural set {items} is too common by chance or overlaps another at {support}")
            bases = tuple(sorted({b for b in by_base if sku.get(b) in items}))
            planted.append(Planted(items, support, "natural", bases, tuple(sorted({cat_of[b] for b in bases}))))
        for support in supports:
            for _ in range(1000):
                cats = rng.choice(sorted(set(cat_of.values())), size=size, replace=False)
                bases = tuple(str(rng.choice([b for b in by_base if cat_of[b] == c])) for c in cats)
                items = tuple(sorted(sku_for(b) for b in bases))
                if len(set(items)) == size and fits(items, support, planted):
                    planted.append(Planted(items, support, "surprising", tuple(sorted(bases)),
                                           tuple(sorted({cat_of[b] for b in bases}))))
                    break
            else:
                raise ValueError(f"no surprising itemset of size {size} fits at support {support}")
    return sorted(planted, key=lambda p: (-p.target_support, len(p.items), p.kind))


def generate(catalogue: List[Product], cfg: GeneratorConfig) -> Tuple[List[List[int]], List[Planted], List[dict]]:
    """Baskets, the planted itemsets, and a report of what was planted (the hidden truth)."""
    shares = popularity(catalogue)
    sample_cfg = GeneratorConfig(**{**cfg.to_dict(), "n_baskets": max(50_000, cfg.n_baskets // 2)})
    sample = background_baskets(catalogue, sample_cfg, seed=cfg.seed + 10_000)
    planted = choose_planted(catalogue, shares, sample, cfg, seed=cfg.seed)
    baskets = background_baskets(catalogue, cfg, seed=cfg.seed)
    base_of = {p.pid: p.base for p in catalogue}
    rng = np.random.default_rng(cfg.seed + 1)
    infos = {}
    for p in sorted(planted, key=lambda p: (-len(p.items), -p.target_support)):
        baskets, infos[p.items] = plant(baskets, p, base_of, rng)
    report = []
    for p in planted:
        realised = float(_holders(baskets, p.items).mean())
        report.append({"items": list(p.items), "names": [catalogue[i].name for i in p.items], "bases": list(p.bases),
                       "categories": list(p.categories), "size": len(p.items), "kind": p.kind,
                       "target_support": p.target_support, "realised_support": realised,
                       "intended_support_range": [0.9 * p.target_support, 1.25 * p.target_support],
                       **infos[p.items]})
    return baskets, planted, report
