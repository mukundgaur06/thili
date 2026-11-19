"""Fine-tune real GPT-2 (124M) on SST-2 sentiment with different strategies.

    python finetune.py --mode lora --rank 8                # LoRA on attention (c_attn = Q,K,V)
    python finetune.py --mode lora --rank 8 --targets c_attn c_proj
    python finetune.py --mode qlora --rank 8               # NF4 4-bit frozen base + LoRA
    python finetune.py --mode head                         # linear probe: train the head only
    python finetune.py --mode full                         # every weight (needs ~2.5 GB RAM)

Every run trains the classification head too (PEFT calls this ``modules_to_save``), picks
the best step on a dev split carved from train, and reports accuracy on the untouched
SST-2 validation set (used as test). Results are appended to results/results.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from lora.data import Batches, load_sst2
from lora.gpt2 import GPT2Classifier, load_gpt2, load_tokenizer
from lora.inject import count_params, inject_lora, save_adapter
from lora.qlora import model_storage_bytes, quantize_model

ROOT = Path(__file__).resolve().parent


def rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**20
    except ImportError:  # pragma: no cover
        return float("nan")


@torch.no_grad()
def accuracy(model, batches) -> float:
    model.eval()
    right = total = 0
    for x, lengths, y in batches:
        right += int((model(x, lengths).argmax(-1) == y).sum())
        total += y.numel()
    model.train()
    return right / total


def build(mode: str, rank: int, alpha: float, targets, dropout: float, weights=None):
    model = GPT2Classifier(load_gpt2(weights))
    if mode in ("lora", "qlora"):
        wrapped = inject_lora(model, targets, r=rank, alpha=alpha, dropout=dropout)
    elif mode == "head":
        wrapped = []
        for p in model.parameters():
            p.requires_grad_(False)
    elif mode == "full":
        wrapped = []
        for p in model.parameters():
            p.requires_grad_(True)
    else:
        raise ValueError(mode)
    if mode == "qlora":
        quantize_model(model)              # every c_attn / c_proj / c_fc base becomes NF4
    model.score.weight.requires_grad_(True)
    return model, wrapped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["lora", "qlora", "head", "full"], default="lora")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--targets", nargs="+", default=["c_attn"])
    ap.add_argument("--lr", type=float, default=None, help="default: 5e-4 LoRA/QLoRA, 2e-3 head, 2e-5 full")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--eval-every", type=int, default=125)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--weights", default=None, help="path to GPT-2 model.safetensors (default: HF cache)")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    lr = a.lr or {"lora": 5e-4, "qlora": 5e-4, "head": 2e-3, "full": 2e-5}[a.mode]
    name = a.name or (f"{a.mode}-r{a.rank}-{'+'.join(a.targets)}" if a.mode in ("lora", "qlora") else a.mode)
    out_dir = ROOT / "adapters"
    out_dir.mkdir(exist_ok=True)

    tok = load_tokenizer()
    train, dev, test = load_sst2(a.n_train, seed=a.seed)
    train_b = Batches(train, tok, a.batch_size, shuffle=True, seed=a.seed)
    dev_b = Batches(dev, tok, 64, shuffle=False)
    test_b = Batches(test, tok, 64, shuffle=False)

    model, wrapped = build(a.mode, a.rank, a.alpha, a.targets, a.dropout, a.weights)
    params = count_params(model)
    print(f"[{name}] trainable {params['trainable']:,} / {params['total']:,} = {params['percent']:.3f}%"
          f"{' | LoRA on ' + str(len(wrapped)) + ' layers' if wrapped else ''}")
    zero_shot_dev = accuracy(model, dev_b)
    print(f"dev accuracy before training (random head): {zero_shot_dev:.3f}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0 if a.mode != "full" else 0.01)
    total_steps = math.ceil(a.epochs * len(train_b))
    warm = max(1, total_steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total_steps - s) / max(1, total_steps - warm)))

    best = {"dev": -1.0, "step": 0}
    best_state = None
    history = []
    step, t0, peak = 0, time.time(), rss_mb()
    model.train()
    while step < total_steps:
        for x, lengths, y in train_b:
            loss = F.cross_entropy(model(x, lengths), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
            step += 1
            peak = max(peak, rss_mb())
            if step % a.eval_every == 0 or step == total_steps:
                dev_acc = accuracy(model, dev_b)
                history.append({"step": step, "loss": round(loss.item(), 4), "dev_acc": dev_acc})
                print(f"step {step:4d}/{total_steps} | loss {loss.item():.4f} | dev {dev_acc:.3f} "
                      f"| {time.time() - t0:5.0f}s | rss {peak:.0f} MB")
                if dev_acc > best["dev"]:
                    best = {"dev": dev_acc, "step": step}
                    best_state = {k: v.detach().clone() for k, v in model.named_parameters() if v.requires_grad}
            if step >= total_steps:
                break
    train_time = time.time() - t0

    # restore the best checkpoint (by dev accuracy) and score it once on test
    with torch.no_grad():
        for k, v in model.named_parameters():
            if k in best_state:
                v.copy_(best_state[k])
    test_acc = accuracy(model, test_b)

    size = None
    if a.mode in ("lora", "qlora", "head"):
        meta = {"mode": a.mode, "rank": a.rank, "alpha": a.alpha, "targets": ",".join(a.targets),
                "base": "openai-community/gpt2", "task": "sst2", "test_acc": round(test_acc, 4)}
        size = save_adapter(str(out_dir / f"{name}.safetensors"), model, meta, extra=("score",))
    result = {
        "name": name, "mode": a.mode, "rank": a.rank if a.mode in ("lora", "qlora") else None,
        "alpha": a.alpha if a.mode in ("lora", "qlora") else None, "targets": a.targets if wrapped else None,
        "lr": lr, "epochs": a.epochs, "n_train": len(train), "trainable_params": params["trainable"],
        "trainable_percent": params["percent"], "best_dev_acc": best["dev"], "best_step": best["step"],
        "test_acc": test_acc, "train_time_s": round(train_time), "peak_rss_mb": round(peak),
        "adapter_bytes": size, "model_storage_mb": round(model_storage_bytes(model) / 2**20, 1),
        "history": history,
    }
    print(f"[{name}] best dev {best['dev']:.3f} at step {best['step']} | TEST accuracy {test_acc:.4f} | "
          f"{train_time:.0f}s | peak RSS {peak:.0f} MB" + (f" | adapter {size / 1024:.0f} KB" if size else ""))
    path = ROOT / "results" / "results.json"
    path.parent.mkdir(exist_ok=True)
    runs = json.loads(path.read_text()) if path.exists() else []
    runs = [r for r in runs if r["name"] != name] + [result]
    path.write_text(json.dumps(runs, indent=1))


if __name__ == "__main__":
    main()
