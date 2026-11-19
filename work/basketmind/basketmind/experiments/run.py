"""Run the BasketMind experiments, stage by stage. Every stage writes JSON under artifacts/<dataset>/.

    python -m basketmind.experiments.run generate  --dataset full     # 1,000 products x 100,000 baskets
    python -m basketmind.experiments.run mine      --dataset full     # exact Apriori + brute-force check + ceiling
    python -m basketmind.experiments.run baselines --dataset full     # popularity, co-occurrence, rules
    python -m basketmind.experiments.run lm-eval   --dataset full --run offshelf
    python -m basketmind.experiments.run lm-train  --dataset full --run cpt-fresh --ordering fresh --steps 600
    python -m basketmind.experiments.run lm-eval   --dataset full --run cpt-fresh
    python -m basketmind.experiments.run probe     --dataset full --run cpt-fresh
    python -m basketmind.experiments.run report    --dataset full

``lm-train`` is resumable: with ``--max-minutes`` it stops at a step boundary and the same command
continues it. Use ``--dataset dev`` (10,000 baskets) for a quick run of everything.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"


def art(ds: str) -> Path:
    p = ART / ds
    (p / "per_case").mkdir(parents=True, exist_ok=True)
    return p


def write(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def cases_for(d, n_cases: int, seed: int = 0):
    from ..evaluation.ranking import make_cases
    return make_cases(d.split("test"), n_cases, seed=seed)


def save_per_case(ds: str, name: str, per_case: dict) -> None:
    np.savez(art(ds) / "per_case" / f"{name}.npz", **per_case)


def summarise_with_ci(per_case: dict, mask=None, n_boot: int = 2000) -> dict:
    from ..evaluation.bootstrap import bootstrap_ci
    out = {}
    for k, v in per_case.items():
        if k == "rank":
            continue
        vv = v if mask is None else v[mask]
        out["mrr" if k == "rr" else k] = bootstrap_ci(vv, n_boot=n_boot, seed=1)
    ranks = per_case["rank"] if mask is None else per_case["rank"][mask]
    out["median_rank"] = float(np.median(ranks))
    return out


# --------------------------------------------------------------------------------------------- stages
def stage_generate(a):
    from ..data.dataset import build
    t = time.time()
    out = build(a.dataset)
    print(f"built {a.dataset} in {time.time() - t:.0f}s -> {out.relative_to(ROOT)}")


def stage_mine(a):
    from ..data.dataset import load, load_truth
    from ..evaluation.bootstrap import bootstrap_ci
    from ..evaluation.itemsets import planted_recovery, recall_by_band
    from ..mining.apriori import apriori, brute_force
    from ..mining.rules import generate_rules

    d = load(a.dataset)
    planted, planted_rows, truth, labels, cfg = load_truth(a.dataset)
    n = len(d.baskets)
    stats = {}
    t = time.time()
    counts = apriori(d.baskets, d.n_items, cfg["min_support"], cfg["max_size"], stats)
    secs = time.time() - t
    sup = {k: v / n for k, v in counts.items()}
    assert set(sup) == set(truth), "mined itemsets differ from the stored truth"

    rng = np.random.default_rng(0)
    sample = [d.baskets[i] for i in rng.choice(n, 2000, replace=False)]
    t = time.time()
    agree = apriori(sample, d.n_items, 0.005, 4) == brute_force(sample, 0.005, 4)
    bf_secs = time.time() - t

    rules = generate_rules(sup, min_confidence=0.2, max_consequent=1)
    planted_sets = {frozenset(p.items) for p in planted}
    top_rules = [dict(r.to_dict(), planted_pair=any(r.antecedent | r.consequent <= s for s in planted_sets))
                 for r in rules[:200]]

    # The ceiling: how much of the truth does the *exact* miner recover from a sample of real baskets of
    # the same size as the model's generated sample? Anything a model recovers is bounded by sampling noise.
    ceiling = {}
    for size in (2000, 10000):
        rows = []
        for rep in range(20):
            idx = np.random.default_rng(100 + rep).choice(n, size, replace=False)
            sc = apriori([d.baskets[i] for i in idx], d.n_items, cfg["min_support"], 5)
            found = {k: v / size for k, v in sc.items()}
            rows.append([r["recovered"] for r in planted_recovery(found, planted, cfg["min_support"])])
        hits = np.array(rows, dtype=float)                       # (reps, n_planted)
        per_support = {}
        for s in sorted({p.target_support for p in planted}, reverse=True):
            cols = [i for i, p in enumerate(planted) if p.target_support == s]
            per_support[str(s)] = float(hits[:, cols].mean())
        ceiling[str(size)] = {"planted_recall": bootstrap_ci(hits.mean(1), seed=1), "by_support": per_support,
                              "repetitions": 20}

    bands = [0.001, 0.002, 0.005, 0.01, 0.03, 1.0]
    out = {"n_baskets": n, "min_support": cfg["min_support"], "max_size": cfg["max_size"], "seconds": round(secs, 2),
           "levels": stats["levels"], "n_frequent": len(counts), "by_label": cfg["truth_by_label"],
           "bruteforce_check": {"baskets": 2000, "min_support": 0.005, "max_size": 4, "identical": agree,
                                "seconds": round(bf_secs, 2)},
           "planted_on_full_data": planted_recovery(sup, planted, cfg["min_support"]),
           "planted_rows": planted_rows, "n_rules": len(rules), "top_rules": top_rules,
           "ceiling": ceiling, "truth_recall_bands": recall_by_band(sup, truth, labels, bands, cfg["min_support"])}
    write(art(a.dataset) / "mining.json", out)
    print(f"mined {len(counts)} frequent itemsets in {secs:.2f}s; brute force identical: {agree}; "
          f"ceiling recall (10k sample) {ceiling['10000']['planted_recall']['estimate']:.3f}")


def stage_baselines(a):
    from ..data.dataset import load, load_truth
    from ..evaluation.ranking import evaluate, planted_mask
    from ..recommenders.cooccurrence import AssociationRules, CoOccurrence
    from ..recommenders.popularity import Popularity

    d = load(a.dataset)
    planted = load_truth(a.dataset)[0]
    cases = cases_for(d, a.n_cases)
    pm = planted_mask(cases, planted)
    train = d.split("train")
    out = {"n_cases": len(cases), "n_planted_cases": int(pm.sum()), "models": {}}
    for rec in (Popularity(), CoOccurrence("count"), CoOccurrence("confidence"), CoOccurrence("lift"),
                CoOccurrence("blend"), AssociationRules(min_support=a.rules_min_support)):
        t = time.time()
        rec.fit(train, d.n_items)
        fit_s = time.time() - t
        t = time.time()
        pc = evaluate(rec.score_many([c["visible"] for c in cases]), cases)
        save_per_case(a.dataset, rec.name, pc)
        out["models"][rec.name] = {"all": summarise_with_ci(pc), "planted_cases": summarise_with_ci(pc, pm),
                                   "fit_seconds": round(fit_s, 2), "score_seconds": round(time.time() - t, 2)}
        s = out["models"][rec.name]
        print(f"{rec.name:24s} MRR {s['all']['mrr']['estimate']:.3f} [{s['all']['mrr']['lo']:.3f}, {s['all']['mrr']['hi']:.3f}]  "
              f"hits@10 {s['all']['hits@10']['estimate']:.3f}  planted-case MRR {s['planted_cases']['mrr']['estimate']:.3f}")
    write(art(a.dataset) / "baselines.json", out)


def _load_lm(a, d):
    from ..language_model.model import BasketLM
    if a.run == "offshelf":
        return BasketLM.from_pretrained_gpt2(d.catalogue)
    return BasketLM.load(art(a.dataset) / "runs" / a.run, d.catalogue)


def stage_lm_train(a):
    from ..data.dataset import load
    from ..language_model.model import BasketLM
    from ..language_model.training import TrainConfig, train

    d = load(a.dataset)
    tr = d.split("train")
    if a.train_subset:
        idx = np.random.default_rng(7).choice(len(tr), a.train_subset, replace=False)
        tr = [tr[i] for i in sorted(idx)]
    cfg = TrainConfig(ordering=a.ordering, steps=a.steps, batch_size=a.batch_size, lr=a.lr, eval_every=a.eval_every,
                      seed=a.seed, threads=a.threads, train_subset=a.train_subset)
    model = BasketLM.from_pretrained_gpt2(d.catalogue)
    out = art(a.dataset) / "runs" / a.run
    res = train(model, tr, d.split("val"), cfg, out, max_minutes=a.max_minutes)
    meta = {"run": a.run, "config": cfg.to_dict(), "done": res["done"], "step": res["step"], "params": model.counts(),
            "n_train_baskets": len(tr), "machine": platform.processor() or platform.machine()}
    write(out / "run.json", meta)
    print(("finished" if res["done"] else "paused") + f" at step {res['step']}/{cfg.steps}")


def stage_lm_eval(a):
    import torch
    from ..data.dataset import load, load_truth
    from ..evaluation.ranking import evaluate, planted_mask
    from ..language_model.inference import next_item_logprobs

    torch.set_num_threads(a.threads)
    d = load(a.dataset)
    planted = load_truth(a.dataset)[0]
    cases = cases_for(d, a.n_cases)
    pm = planted_mask(cases, planted)
    model = _load_lm(a, d)
    vis = [c["visible"] for c in cases]
    t = time.time()
    pc = evaluate(next_item_logprobs(model, vis, n_orders=a.n_orders, seed=0), cases)
    secs = time.time() - t
    save_per_case(a.dataset, f"llm-{a.run}", pc)
    # Order sensitivity: the same cases, visible items presented sorted by id (the "fixed" order).
    pc_fixed = evaluate(next_item_logprobs(model, vis, ordering="fixed"), cases)
    save_per_case(a.dataset, f"llm-{a.run}-fixedprefix", pc_fixed)
    from ..language_model.dataset import encode_eval, validation_loss
    val = d.split("val")[:1000]
    out = {"run": a.run, "n_cases": len(cases), "n_orders": a.n_orders, "score_seconds": round(secs, 1),
           "all": summarise_with_ci(pc), "planted_cases": summarise_with_ci(pc, pm),
           "fixed_order_prefix": summarise_with_ci(pc_fixed),
           "val_loss_random_order": validation_loss(model, encode_eval(val, model.vocab, "fresh"), model.vocab.pad_id),
           "val_loss_fixed_order": validation_loss(model, encode_eval(val, model.vocab, "fixed"), model.vocab.pad_id),
           "params": model.counts()}
    (art(a.dataset) / "lm").mkdir(exist_ok=True)
    write(art(a.dataset) / "lm" / f"eval-{a.run}.json", out)
    print(f"{a.run}: MRR {out['all']['mrr']['estimate']:.3f} [{out['all']['mrr']['lo']:.3f}, {out['all']['mrr']['hi']:.3f}]  "
          f"hits@10 {out['all']['hits@10']['estimate']:.3f}  | fixed-order prefix MRR {out['fixed_order_prefix']['mrr']['estimate']:.3f} "
          f"| val loss random {out['val_loss_random_order']:.3f} fixed {out['val_loss_fixed_order']:.3f}  ({secs:.0f}s)")


def stage_probe(a):
    import torch
    from ..data.dataset import load, load_truth
    from ..evaluation.bootstrap import bootstrap_ci
    from ..evaluation.itemsets import classify, planted_recovery, precision, recall_by_band
    from ..language_model.generation import sample_baskets
    from ..mining.apriori import apriori

    torch.set_num_threads(a.threads)
    d = load(a.dataset)
    planted, _, truth, labels, cfg = load_truth(a.dataset)
    model = _load_lm(a, d)
    t = time.time()
    gen, gstats = sample_baskets(model, a.n_generate, temperature=a.temperature, seed=0)
    gen_s = time.time() - t
    ms = cfg["min_support"]
    nonempty = [b for b in gen if b]
    counts = apriori(nonempty, d.n_items, ms, 5)
    found = {k: v / len(nonempty) for k, v in counts.items()}
    rec = planted_recovery(found, planted, ms)
    reps = []
    for r in range(a.n_boot_probe):
        idx = np.random.default_rng(1000 + r).integers(0, len(nonempty), len(nonempty))
        c = apriori([nonempty[i] for i in idx], d.n_items, ms, 5)
        f = {k: v / len(idx) for k, v in c.items()}
        reps.append(np.mean([x["recovered"] for x in planted_recovery(f, planted, ms)]))
    by_support = {}
    for s in sorted({p.target_support for p in planted}, reverse=True):
        rows = [x for x in rec if x["target_support"] == s]
        by_support[str(s)] = {"recovered": sum(x["recovered"] for x in rows), "of": len(rows),
                              "mean_found_support": float(np.mean([x["found_support"] for x in rows]))}
    multi = [k for k in found if len(k) >= 2]
    label_counts = {}
    for k in multi:
        lab = labels.get(k, "not frequent in real data")
        label_counts[lab] = label_counts.get(lab, 0) + 1
    cat_of = {p.pid: p.category for p in d.catalogue}
    same_cat = [len({cat_of[i] for i in k}) == 1 for k in multi if len(k) == 2]
    real_same_cat = [len({cat_of[i] for i in k}) == 1 for k in truth if len(k) == 2]
    # Item marginals: does the model sell items as often as the shop does?
    real_freq = np.bincount(np.concatenate([np.array(b) for b in d.split("train")]), minlength=d.n_items) / len(d.split("train"))
    gen_freq = np.bincount(np.concatenate([np.array(b) for b in nonempty]), minlength=d.n_items) / len(nonempty)
    out = {"run": a.run, "n_generated": len(gen), "nonempty": len(nonempty), "generation_seconds": round(gen_s, 1),
           "generation_stats": gstats, "min_support": ms,
           "planted_recall": {"estimate": float(np.mean([x["recovered"] for x in rec])),
                              "bootstrap": {"lo": float(np.percentile(reps, 2.5)), "hi": float(np.percentile(reps, 97.5)),
                                            "n_boot": a.n_boot_probe, "resampled": "generated baskets"}},
           "planted_by_support": by_support, "planted_detail": rec,
           "precision_vs_truth": precision(found, truth, ms),
           "recall_bands": recall_by_band(found, truth, labels, [0.001, 0.002, 0.005, 0.01, 0.03, 1.0], ms),
           "frequent_in_generated": len(counts), "multi_item_frequent_in_generated": len(multi),
           "labels_of_generated_itemsets": label_counts,
           "pair_same_category_share": {"generated": float(np.mean(same_cat)) if same_cat else None,
                                        "real": float(np.mean(real_same_cat)) if real_same_cat else None},
           "item_frequency_correlation": float(np.corrcoef(real_freq, gen_freq)[0, 1]),
           "mean_basket_size": {"generated": float(np.mean([len(b) for b in nonempty])),
                                "real": float(np.mean([len(set(b)) for b in d.baskets]))},
           "examples": [d.names(b) for b in gen[:12]]}
    write(art(a.dataset) / "lm" / f"probe-{a.run}.json", out)
    (art(a.dataset) / "lm" / f"generated-{a.run}.jsonl").write_text("\n".join(json.dumps(b) for b in gen[:2000]) + "\n")
    print(f"{a.run}: generated {len(gen)} baskets in {gen_s:.0f}s, mean size {out['mean_basket_size']['generated']:.2f} "
          f"(real {out['mean_basket_size']['real']:.2f}); planted recall {out['planted_recall']['estimate']:.3f} "
          f"[{out['planted_recall']['bootstrap']['lo']:.3f}, {out['planted_recall']['bootstrap']['hi']:.3f}]; "
          f"precision {out['precision_vs_truth']:.3f}; by support {by_support}")


def stage_report(a):
    from ..evaluation.bootstrap import paired_bootstrap
    p = art(a.dataset) / "per_case"
    names = sorted(f.stem for f in p.glob("*.npz"))
    data = {n: dict(np.load(p / f"{n}.npz")) for n in names}
    pairs = []
    focus = [n for n in names if n.startswith("llm-") and not n.endswith("fixedprefix")]
    baselines = [n for n in names if not n.startswith("llm-")]
    for x in focus:
        for y in baselines + [f for f in focus if f != x]:
            for metric in ("rr", "hits@10", "recall@10"):
                pairs.append({"a": x, "b": y, "metric": "mrr" if metric == "rr" else metric,
                              **paired_bootstrap(data[x][metric], data[y][metric], seed=3)})
    for metric in ("rr", "hits@10"):
        if "association-rules" in data and "cooccurrence-confidence" in data:
            pairs.append({"a": "association-rules", "b": "cooccurrence-confidence", "metric": "mrr" if metric == "rr" else metric,
                          **paired_bootstrap(data["association-rules"][metric], data["cooccurrence-confidence"][metric], seed=3)})
    write(art(a.dataset) / "comparisons.json", {"models": names, "paired": pairs})
    for r in pairs:
        if r["metric"] == "mrr":
            print(f"{r['a']:>22s} - {r['b']:<24s} MRR diff {r['difference']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]  P(a>b) {r['p_a_better']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["generate", "mine", "baselines", "lm-train", "lm-eval", "probe", "report"])
    ap.add_argument("--dataset", default="dev", choices=["dev", "full"])
    ap.add_argument("--run", default="cpt-fresh")
    ap.add_argument("--ordering", default="fresh", choices=["fixed", "frozen", "fresh"])
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--train-subset", type=int, default=None)
    ap.add_argument("--max-minutes", type=float, default=1e9)
    ap.add_argument("--n-cases", type=int, default=2000)
    ap.add_argument("--n-orders", type=int, default=3)
    ap.add_argument("--n-generate", type=int, default=10000)
    ap.add_argument("--n-boot-probe", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--rules-min-support", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    import basketmind  # noqa: F401  (environment fix-ups before transformers loads)
    {"generate": stage_generate, "mine": stage_mine, "baselines": stage_baselines, "lm-train": stage_lm_train,
     "lm-eval": stage_lm_eval, "probe": stage_probe, "report": stage_report}[a.stage](a)


if __name__ == "__main__":
    main()
