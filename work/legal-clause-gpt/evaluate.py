"""Lesson 7: how good is it? Tag unseen LEDGAR test clauses and score the answers.

    python evaluate.py                    # 2,000 test clauses, plus baselines
    python evaluate.py --n-test 10000     # the whole test split

Reports, on the same seeded test sample:

* **free generation accuracy**: greedy next token == the gold ``<Type>`` token. Anything
  that is not a clause type (e.g. a plain word) counts as wrong, and as "invalid".
* **constrained accuracy**: argmax restricted to the 100 type tokens; also top-3 accuracy
  and macro-F1 (every type weighted equally, so rare types count as much as common ones).
* **baselines**, for scale:
  * the majority class
  * GPT-2 *before* fine-tuning: new tokens initialised from their label words, constrained
  * TF-IDF + logistic regression, trained on the same clauses with the same 64-token cut
"""

from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

import torch

import clausegpt  # noqa: F401  (must come before transformers)
from clausegpt.data import load_ledgar, sample
from clausegpt.labels import label_token
from clausegpt.model import load_base
from clausegpt.tagger import ClauseTagger, predict_all

ROOT = Path(__file__).resolve().parent


def score(preds, gold, names):
    from sklearn.metrics import f1_score
    pred_c = [p.label for p in preds]
    n = len(gold)
    free_ok = sum(p.generated == label_token(g) for p, g in zip(preds, gold))
    top3 = sum(g in [t for t, _ in p.top[:3]] for p, g in zip(preds, gold))
    return {
        "n": n,
        "free_generation_accuracy": free_ok / n,
        "invalid_rate": sum(not p.valid for p in preds) / n,
        "constrained_accuracy": sum(a == b for a, b in zip(pred_c, gold)) / n,
        "top3_accuracy": top3 / n,
        "macro_f1": f1_score(gold, pred_c, labels=sorted(set(gold)), average="macro", zero_division=0),
    }


def per_class(preds, gold):
    from sklearn.metrics import precision_recall_fscore_support
    labels = sorted(set(gold))
    pred = [p.label for p in preds]
    p, r, f, s = precision_recall_fscore_support(gold, pred, labels=labels, zero_division=0)
    rows = [{"type": l, "precision": float(pp), "recall": float(rr), "f1": float(ff), "support": int(ss)}
            for l, pp, rr, ff, ss in zip(labels, p, r, f, s)]
    confusions = collections.Counter((g, q) for g, q in zip(gold, pred) if g != q).most_common(10)
    return rows, [{"gold": g, "predicted": q, "count": c} for (g, q), c in confusions]


def tfidf_baseline(train_texts, train_y, test_texts, test_y):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    clf = LogisticRegression(max_iter=2000, C=10.0)
    clf.fit(vec.fit_transform(train_texts), train_y)
    pred = clf.predict(vec.transform(test_texts))
    return {"accuracy": float((pred == test_y).mean()),
            "macro_f1": f1_score(test_y, pred, labels=sorted(set(test_y)), average="macro", zero_division=0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "outputs" / "clause-gpt2"))
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--n-train", type=int, default=None, help="default: the n_train used for training")
    ap.add_argument("--skip-base", action="store_true", help="skip the not-fine-tuned GPT-2 baseline")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    import numpy as np

    run = json.loads((ROOT / "results" / "train_run.json").read_text()) if (ROOT / "results" / "train_run.json").exists() else {}
    n_train = args.n_train or run.get("n_train", 4000)
    ds, names = load_ledgar()
    test = sample(ds["test"], args.n_test, args.seed)
    clauses, gold = test["text"], [names[l] for l in test["label"]]

    tagger = ClauseTagger.load(args.model)
    t0 = time.time()
    preds = predict_all(tagger, clauses, progress=lambda k: print(f"\r  tagged {k}/{len(clauses)}", end="", flush=True))
    secs = time.time() - t0
    print()
    model_rel = Path(args.model).resolve().relative_to(ROOT).as_posix() if Path(args.model).resolve().is_relative_to(ROOT) else args.model
    report = {"model": model_rel, "n_train": n_train, "fine_tuned": score(preds, gold, names),
              "clauses_per_second": round(len(clauses) / secs, 1)}

    # model.generate agrees with our one-pass "free generation"?
    k = 20
    agree = sum(tagger.generate(c, max_new_tokens=1) == preds[i].generated for i, c in enumerate(clauses[:k]))
    report["generate_agreement"] = f"{agree}/{k}"
    rows, conf = per_class(preds, gold)
    report["per_class"] = rows
    report["top_confusions"] = conf
    report["examples"] = [{"clause": c[:300], "gold": g, "generated": p.generated, "top": p.top}
                          for c, g, p in list(zip(clauses, gold, preds))[:12]]

    # ---------------------------------------------------------------- baselines
    train_counts = collections.Counter(ds["train"]["label"])
    majority = names[train_counts.most_common(1)[0][0]]
    report["majority_class"] = {"type": majority, "accuracy": sum(g == majority for g in gold) / len(gold)}

    cut = lambda texts: [tagger.tok.decode(tagger.tok(t.strip()).input_ids[:tagger.max_clause_tokens]) for t in texts]  # noqa: E731
    test_cut = cut(clauses)
    y_test = np.array(gold)
    tr = sample(ds["train"], n_train, run.get("seed", 0))
    report["tfidf_logreg_same_train"] = {"n_train": len(tr), **tfidf_baseline(cut(tr["text"]), [names[l] for l in tr["label"]], test_cut, y_test)}
    report["tfidf_logreg_all_train"] = {"n_train": len(ds["train"]), **tfidf_baseline(
        cut(ds["train"]["text"]), [names[l] for l in ds["train"]["label"]], test_cut, y_test)}

    if not args.skip_base:
        model, tok = load_base(names)
        base = ClauseTagger(model, tok, names, tagger.max_clause_tokens)
        report["gpt2_before_fine_tuning"] = score(predict_all(base, clauses), gold, names)
        del base, model

    out = ROOT / "results" / "eval.json"
    out.write_text(json.dumps(report, indent=1))
    ft = report["fine_tuned"]
    print(f"\nLEDGAR test, {ft['n']} clauses (fine-tuned on {n_train:,})")
    print(f"  free generation accuracy  {ft['free_generation_accuracy']:.3f}   (invalid outputs {ft['invalid_rate']:.3f})")
    print(f"  constrained accuracy      {ft['constrained_accuracy']:.3f}   top-3 {ft['top3_accuracy']:.3f}   macro-F1 {ft['macro_f1']:.3f}")
    print(f"  model.generate agrees with one-pass decoding on {report['generate_agreement']} clauses")
    print(f"  majority class ({majority})  {report['majority_class']['accuracy']:.3f}")
    for key in ("tfidf_logreg_same_train", "tfidf_logreg_all_train"):
        b = report[key]
        print(f"  TF-IDF + LogReg, {b['n_train']:>6,} train  acc {b['accuracy']:.3f}  macro-F1 {b['macro_f1']:.3f}")
    if "gpt2_before_fine_tuning" in report:
        b = report["gpt2_before_fine_tuning"]
        print(f"  GPT-2 before fine-tuning  constrained acc {b['constrained_accuracy']:.3f}  free {b['free_generation_accuracy']:.3f}")
    worst = sorted((r for r in rows if r["support"] >= 10), key=lambda r: r["f1"])[:5]
    print("  hardest types:", ", ".join(f"{r['type']} {r['f1']:.2f}" for r in worst))
    print("  top confusions:", "; ".join(f"{c['gold']} -> {c['predicted']} x{c['count']}" for c in conf[:5]))
    print(f"saved {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
