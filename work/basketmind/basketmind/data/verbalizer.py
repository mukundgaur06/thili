"""Turn a basket (a *set* of product ids) into a token sequence, and back.

Every product gets one stable special token, ``<p0042>`` for product id 42, so a basket of 7
products is 9 tokens: ``<basket> <p0042> <p0917> ... </basket>``. Product ids survive the
round trip exactly, and scoring every product as the next item takes one forward pass. For
reading, the same basket renders as ``<basket> Meadow Farm Milk Regular ; Oven Lane Bread
Classic ; ... </basket>``.

**Order.** A basket has no order, but a language model reads left to right, so an order must
be chosen. Three policies, all keeping the basket's *membership* unchanged:

* ``fixed``: sorted by product id, every time. The model can learn the artificial rule
  "the next id is larger than the last one". That is a next-token signal that exists only
  because of the presentation, and it says nothing about what sells together.
* ``frozen``: one random permutation per basket, drawn once and reused every epoch. It has
  no global rule, but a repeated basket is always seen in the same order, so it can be
  memorised.
* ``fresh``: a new random permutation every time a basket is shown. The only stable signal
  left is co-occurrence itself.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

BASKET_OPEN = "<basket>"
BASKET_CLOSE = "</basket>"
ORDERINGS = ("fixed", "frozen", "fresh")


def product_token(pid: int) -> str:
    return f"<p{pid:04d}>"


def token_pid(token: str) -> int | None:
    if len(token) == 7 and token.startswith("<p") and token.endswith(">") and token[2:6].isdigit():
        return int(token[2:6])
    return None


class Vocabulary:
    """Ids 0 .. n_products-1 are products, then <basket>, </basket>, <pad>."""

    def __init__(self, n_products: int):
        self.n_products = n_products
        self.open_id = n_products
        self.close_id = n_products + 1
        self.pad_id = n_products + 2
        self.size = n_products + 3

    def tokens(self) -> List[str]:
        return [product_token(i) for i in range(self.n_products)] + [BASKET_OPEN, BASKET_CLOSE, "<pad>"]

    def is_product(self, tid: int) -> bool:
        return 0 <= tid < self.n_products


def order(basket: Sequence[int], policy: str, rng: np.random.Generator | None = None,
          frozen_seed: int | None = None) -> List[int]:
    items = sorted(set(basket))
    if policy == "fixed":
        return items
    if policy == "frozen":
        return [items[i] for i in np.random.default_rng(frozen_seed).permutation(len(items))]
    if policy == "fresh":
        return [items[i] for i in (rng or np.random.default_rng()).permutation(len(items))]
    raise ValueError(f"unknown ordering {policy!r}; choose from {ORDERINGS}")


def encode(basket: Sequence[int], vocab: Vocabulary, policy: str = "fresh", rng=None, frozen_seed=None,
           close: bool = True) -> List[int]:
    ids = [vocab.open_id] + order(basket, policy, rng, frozen_seed)
    return ids + [vocab.close_id] if close else ids


def decode(ids: Iterable[int], vocab: Vocabulary) -> dict:
    """Parse a generated sequence: product ids (deduplicated, in order) and what was dropped."""
    products, seen, repeats, invalid, closed = [], set(), 0, 0, False
    for t in ids:
        t = int(t)
        if t == vocab.open_id:
            continue
        if t == vocab.close_id:
            closed = True
            break
        if not vocab.is_product(t):
            invalid += 1
            continue
        if t in seen:
            repeats += 1
            continue
        seen.add(t)
        products.append(t)
    return {"items": products, "repeats": repeats, "invalid": invalid, "closed": closed}


def render(basket: Sequence[int], names: Sequence[str]) -> str:
    return f"{BASKET_OPEN} " + " ; ".join(names[i] for i in basket) + f" {BASKET_CLOSE}"


def length_stats(baskets: Sequence[Sequence[int]]) -> dict:
    lens = np.array([len(set(b)) + 2 for b in baskets])
    return {"mean_tokens": float(lens.mean()), "median_tokens": float(np.median(lens)),
            "p95_tokens": float(np.percentile(lens, 95)), "max_tokens": int(lens.max())}
