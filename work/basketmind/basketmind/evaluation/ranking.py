"""Next-item ranking evaluation.

**Cases.** A test basket (never used for training) with n >= 2 distinct items is split at
random, with a fixed seed: the *visible* part is a random ceil(n / 2) items, at least 1, and
the *targets* are all the rest. The candidates are every catalogue item not already visible.

**Multi-item targets.** A basket usually has several correct next items, and any of them is a
good suggestion, so:

* **rank** = the rank of the *best-ranked* target (1 = top);
* **MRR** = mean of 1 / rank;
* **Hits@K** = share of cases where at least one target is in the top K;
* **Recall@K** = mean share of a case's targets that appear in the top K;
* **NDCG@K** with binary relevance over all targets.

Ties count against the target: a target tied with a non-target is ranked below it. A method
that returns the same score everywhere therefore gets the worst rank, not a lucky one.

**Order.** Which items are visible is decided by the seed, not by any order the data was
stored in. Language models, which read the visible items as a sequence, average their scores
over several random orders of the same visible set (``llm_ranker``).

``leave_one_out`` cases (hide exactly one item) are also provided, for comparison with the
reference implementation's protocol.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np


def make_cases(baskets: Sequence[Sequence[int]], n_cases: int | None = None, seed: int = 0,
               protocol: str = "half") -> List[dict]:
    rng = np.random.default_rng(seed)
    eligible = [i for i, b in enumerate(baskets) if len(set(b)) >= 2]
    if n_cases is not None and n_cases < len(eligible):
        eligible = sorted(rng.choice(eligible, n_cases, replace=False).tolist())
    cases = []
    for i in eligible:
        items = sorted(set(baskets[i]))
        perm = [items[j] for j in rng.permutation(len(items))]
        k = max(1, math.ceil(len(items) / 2)) if protocol == "half" else len(items) - 1
        cases.append({"basket": i, "visible": sorted(perm[:k]), "targets": sorted(perm[k:])})
    return cases


def ranks_of(scores: np.ndarray, visible: Sequence[int], targets: Sequence[int]) -> np.ndarray:
    """Pessimistic rank (1-based) of each target among all non-visible items."""
    s = np.asarray(scores, dtype=np.float64)
    keep = np.ones(len(s), dtype=bool)
    keep[list(visible)] = False
    cand = s[keep]
    # cand still contains the target itself, so "how many candidates score >= the target" is its rank
    return np.array([int((cand >= s[t]).sum()) for t in targets])


def case_metrics(scores: np.ndarray, visible: Sequence[int], targets: Sequence[int], ks=(1, 5, 10, 20)) -> dict:
    r = ranks_of(scores, visible, targets)
    best = int(r.min())
    out = {"rank": best, "rr": 1.0 / best}
    for k in ks:
        out[f"hits@{k}"] = float(best <= k)
        out[f"recall@{k}"] = float((r <= k).mean())
    dcg = sum(1 / math.log2(x + 1) for x in r if x <= 10)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(10, len(targets))))
    out["ndcg@10"] = dcg / idcg
    return out


def evaluate(score_rows: np.ndarray, cases: Sequence[dict], ks=(1, 5, 10, 20)) -> Dict[str, np.ndarray]:
    """Per-case metric arrays (kept per case so they can be bootstrapped and paired)."""
    rows = [case_metrics(s, c["visible"], c["targets"], ks) for s, c in zip(score_rows, cases)]
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def summarise(per_case: Dict[str, np.ndarray]) -> Dict[str, float]:
    out = {("mrr" if k == "rr" else k): float(v.mean()) for k, v in per_case.items() if k != "rank"}
    out["median_rank"] = float(np.median(per_case["rank"]))
    return out


def planted_mask(cases: Sequence[dict], planted) -> np.ndarray:
    """Cases where a planted itemset is split between the visible part and the targets."""
    sets = [set(p.items) for p in planted]
    return np.array([any((s & set(c["visible"])) and (s & set(c["targets"])) and
                         s <= set(c["visible"]) | set(c["targets"]) for s in sets) for c in cases])
