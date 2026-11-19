"""Cross-sell and upsell, kept separate.

* **Cross-sell**: a *different* product that goes with the basket (bread → butter). Candidates
  exclude anything already in the basket and any other SKU of a base product already there
  (another brand of milk is not a complement to milk). Ranked by any recommender's score.
* **Upsell**: a *pricier version of something already chosen* (standard → premium coffee). In
  this synthetic shop a shopper picks one SKU per product with odds that ignore the rest of the
  basket, so the data contain no upsell signal to learn. Upsell here is a transparent business
  rule: higher-priced SKUs of the same base product, ordered by how well they sell. It is not
  evaluated against ground truth, because there is none.

A frequently bought item is not automatically a good complement. Bananas co-occur with
everything because they are in 7% of baskets. ``explain`` reports lift next to confidence so
that difference stays visible.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


def cross_sell(scores: np.ndarray, visible: Sequence[int], catalogue, k: int = 10) -> List[int]:
    have_bases = {catalogue[i].base for i in visible}
    s = np.asarray(scores, dtype=float).copy()
    blocked = [p.pid for p in catalogue if p.base in have_bases]
    s[blocked] = -np.inf
    order = np.argsort(-s, kind="stable")
    return [int(i) for i in order[:k] if np.isfinite(s[i])]


def upsell(visible: Sequence[int], catalogue, shares: np.ndarray, k: int = 5) -> List[dict]:
    out = []
    for v in visible:
        p = catalogue[v]
        better = [q for q in catalogue if q.base == p.base and q.price > p.price]
        better.sort(key=lambda q: -shares[q.pid])
        for q in better[:2]:
            out.append({"instead_of": p.pid, "suggest": q.pid, "price_from": p.price, "price_to": q.price,
                        "uplift_pct": 100 * (q.price / p.price - 1), "share_of_sales": float(shares[q.pid]),
                        "reason": f"{q.variant} {q.base} by {q.brand or 'loose'}: +{100 * (q.price / p.price - 1):.0f}% price, "
                                  f"a best seller among pricier {p.base.lower()} options"})
    out.sort(key=lambda r: -r["share_of_sales"])
    return out[:k]


def reason(ev: dict, names: Sequence[str]) -> str:
    """A plain-English reason from co-occurrence evidence."""
    lift = ev["lift"]
    strength = "much more often than chance" if lift >= 3 else "more often than chance" if lift > 1.2 else "about as often as chance"
    return (f"{ev['confidence']:.0%} of baskets with {names[ev['because_of']]} also have it "
            f"({ev['pair_count']} baskets, lift {lift:.1f}: {strength})")
