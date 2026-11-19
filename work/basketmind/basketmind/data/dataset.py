"""Build, save and load a dataset. The ground truth lives in its own folder.

    data/<name>/public/       catalogue.json, baskets.jsonl, splits.json, meta.json   (what models may read)
    data/<name>/ground_truth/ planted.json, truth_itemsets.json, config.json          (the answer key)

`truth_itemsets.json` is the *exact* miner's output on the finished baskets: every frequent
itemset, labelled by why it is frequent (see `basketmind.evaluation.itemsets.classify`).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from .catalogue import Product, make_catalogue, popularity
from .generator import GeneratorConfig, Planted, generate

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
PRESETS = {"dev": GeneratorConfig(n_baskets=10_000), "full": GeneratorConfig(n_baskets=100_000)}
SPLIT = (0.8, 0.1, 0.1)


@dataclass
class Dataset:
    name: str
    catalogue: List[Product]
    baskets: List[List[int]]
    splits: Dict[str, List[int]]          # basket indices per split
    meta: dict

    @property
    def n_items(self) -> int:
        return len(self.catalogue)

    def split(self, name: str) -> List[List[int]]:
        return [self.baskets[i] for i in self.splits[name]]

    def names(self, items) -> List[str]:
        return [self.catalogue[i].name for i in items]


def build(name: str = "dev", cfg: GeneratorConfig | None = None, min_support: float = 0.001, max_size: int = 5,
          directory: Path | None = None) -> Path:
    from ..evaluation.itemsets import classify
    from ..mining.apriori import apriori

    cfg = cfg or PRESETS[name]
    out = Path(directory or DATA / name)
    t = time.time()
    catalogue = make_catalogue(seed=cfg.seed)
    baskets, planted, report = generate(catalogue, cfg)
    gen_s = time.time() - t
    perm = np.random.default_rng(cfg.seed + 2).permutation(len(baskets))
    a, b = int(SPLIT[0] * len(baskets)), int((SPLIT[0] + SPLIT[1]) * len(baskets))
    splits = {"train": sorted(perm[:a].tolist()), "val": sorted(perm[a:b].tolist()), "test": sorted(perm[b:].tolist())}

    t = time.time()
    counts = apriori(baskets, len(catalogue), min_support, max_size)
    mine_s = time.time() - t
    n = len(baskets)
    truth = [{"items": sorted(s), "count": c, "support": c / n, "size": len(s), "label": classify(s, planted)}
             for s, c in sorted(counts.items(), key=lambda kv: (-kv[1], sorted(kv[0])))]

    (out / "public").mkdir(parents=True, exist_ok=True)
    (out / "ground_truth").mkdir(parents=True, exist_ok=True)
    shares = popularity(catalogue)
    (out / "public" / "catalogue.json").write_text(json.dumps(
        [{**p.to_dict(), "share": float(s)} for p, s in zip(catalogue, shares)], indent=0))
    (out / "public" / "baskets.jsonl").write_text("\n".join(json.dumps(b) for b in baskets) + "\n")
    (out / "public" / "splits.json").write_text(json.dumps(splits))
    sizes = np.array([len(b) for b in baskets])
    meta = {"name": name, "n_baskets": n, "n_items": len(catalogue), "seed": cfg.seed,
            "mean_basket_size": float(sizes.mean()), "median_basket_size": float(np.median(sizes)),
            "max_basket_size": int(sizes.max()), "split_sizes": {k: len(v) for k, v in splits.items()},
            "generation_seconds": round(gen_s, 1)}
    (out / "public" / "meta.json").write_text(json.dumps(meta, indent=1))
    (out / "ground_truth" / "planted.json").write_text(json.dumps(report, indent=1))
    (out / "ground_truth" / "truth_itemsets.json").write_text(json.dumps(truth))
    labels = {}
    for r in truth:
        labels[r["label"]] = labels.get(r["label"], 0) + 1
    (out / "ground_truth" / "config.json").write_text(json.dumps(
        {"generator": cfg.to_dict(), "min_support": min_support, "max_size": max_size, "n_truth": len(truth),
         "truth_by_label": labels, "mining_seconds": round(mine_s, 2)}, indent=1))
    return out


def load(name: str = "dev", directory: Path | None = None) -> Dataset:
    d = Path(directory or DATA / name) / "public"
    if not d.exists():
        raise FileNotFoundError(f"{d} not found: run `python -m basketmind.experiments.run generate --dataset {name}`")
    rows = json.loads((d / "catalogue.json").read_text())
    catalogue = [Product(**{k: v for k, v in r.items() if k != "share"}) for r in rows]
    baskets = [json.loads(line) for line in (d / "baskets.jsonl").read_text().splitlines() if line]
    return Dataset(name, catalogue, baskets, json.loads((d / "splits.json").read_text()),
                   json.loads((d / "meta.json").read_text()))


def load_truth(name: str = "dev", directory: Path | None = None):
    """The answer key. Only evaluation code should call this."""
    d = Path(directory or DATA / name) / "ground_truth"
    planted_rows = json.loads((d / "planted.json").read_text())
    planted = [Planted(tuple(r["items"]), r["target_support"], r["kind"], tuple(r["bases"]), tuple(r["categories"]))
               for r in planted_rows]
    truth = {frozenset(r["items"]): r["support"] for r in json.loads((d / "truth_itemsets.json").read_text())}
    labels = {frozenset(r["items"]): r["label"] for r in json.loads((d / "truth_itemsets.json").read_text())}
    return planted, planted_rows, truth, labels, json.loads((d / "config.json").read_text())
