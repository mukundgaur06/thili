"""Training loop: next-token cross-entropy, AdamW, warmup + cosine LR, gradient clipping."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import torch

from .data import Batcher, encode_documents, read_documents, split_documents, unigram_cross_entropy
from .model import GPT, GPTConfig
from .tokenizer import BPETokenizer

PROJECT = Path(__file__).resolve().parents[1]


def portable(path) -> str:
    """Path relative to the project folder when it lies inside it, so logs and checkpoints
    never record the absolute location of the user's checkout."""
    try:
        return Path(path).resolve().relative_to(PROJECT).as_posix()
    except ValueError:
        return str(path)


@dataclass
class TrainConfig:
    corpus: str = "data/stories.txt"
    out_dir: str = "checkpoints"
    vocab_size: int = 512
    val_fraction: float = 0.1
    batch_size: int = 32
    max_steps: int = 1500
    eval_every: int = 100
    eval_batches: int = 10
    lr: float = 3e-3
    min_lr: float = 3e-4
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 1337
    threads: int = 8


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup, then cosine decay from lr to min_lr."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(1.0, progress)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model: GPT, batcher: Batcher, max_batches: int) -> float:
    model.eval()
    total, count = 0.0, 0
    for x, y in batcher.sequential(max_batches):
        _, loss = model(x, y)
        total += float(loss) * y.numel()
        count += y.numel()
    model.train()
    return total / count


def save_checkpoint(path: Path, model: GPT, tok: BPETokenizer, meta: dict) -> None:
    torch.save({
        "model_config": model.cfg.to_dict(),
        "state_dict": model.state_dict(),
        "tokenizer": tok.to_dict(),
        "meta": meta,
    }, path)


def load_checkpoint(path: str | Path, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    tok = BPETokenizer.from_dict(ckpt["tokenizer"])
    return model, tok, ckpt.get("meta", {})


def train(tcfg: TrainConfig, mcfg: GPTConfig, log: Callable[[str], None] = print) -> dict:
    torch.manual_seed(tcfg.seed)
    torch.set_num_threads(tcfg.threads)
    out = Path(tcfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. data: split first, then fit the tokenizer on training text only
    docs = read_documents(tcfg.corpus)
    train_docs, val_docs = split_documents(docs, tcfg.val_fraction, tcfg.seed)
    t0 = time.time()
    tok = BPETokenizer().train("\n\n".join(train_docs), tcfg.vocab_size)
    log(f"tokenizer: {tok.vocab_size} tokens ({len(tok.merges)} merges) in {time.time() - t0:.1f}s")
    train_ids = encode_documents(tok, train_docs)
    val_ids = encode_documents(tok, val_docs)
    train_chars = sum(len(d) for d in train_docs)
    log(f"docs: {len(train_docs)} train / {len(val_docs)} val | tokens: {train_ids.numel():,} train "
        f"/ {val_ids.numel():,} val | {train_chars / train_ids.numel():.2f} chars per token")

    # 2. model
    mcfg.vocab_size = tok.vocab_size
    model = GPT(mcfg)
    opt = model.configure_optimizer(tcfg.lr, tcfg.weight_decay)
    log(f"model: {model.num_params():,} parameters | {mcfg.n_layer} layers, {mcfg.n_head} heads, "
        f"d={mcfg.n_embd}, context={mcfg.block_size}")

    train_b = Batcher(train_ids, mcfg.block_size, tcfg.batch_size, tcfg.seed)
    train_eval_b = Batcher(train_ids, mcfg.block_size, tcfg.batch_size, tcfg.seed + 1)
    val_b = Batcher(val_ids, mcfg.block_size, tcfg.batch_size, tcfg.seed + 2)

    baselines = {
        "uniform": math.log(tok.vocab_size),
        "unigram": unigram_cross_entropy(train_ids, val_ids, tok.vocab_size),
    }
    log(f"baselines (val nats/token): uniform ln(V) = {baselines['uniform']:.3f}, "
        f"unigram = {baselines['unigram']:.3f}")

    history = {"step": [], "train_loss": [], "val_loss": [], "lr": [], "batch_loss": []}
    best_val = float("inf")
    meta_base = {
        "train_config": dict(asdict(tcfg), corpus=portable(tcfg.corpus), out_dir=portable(tcfg.out_dir)), "baselines": baselines,
        "data": {"train_docs": len(train_docs), "val_docs": len(val_docs),
                 "train_tokens": train_ids.numel(), "val_tokens": val_ids.numel(),
                 "chars_per_token": train_chars / train_ids.numel()},
        "num_params": model.num_params(),
    }

    model.train()
    t0 = time.time()
    for step in range(tcfg.max_steps + 1):
        if step % tcfg.eval_every == 0 or step == tcfg.max_steps:
            tr = estimate_loss(model, train_eval_b, tcfg.eval_batches)
            va = estimate_loss(model, val_b, tcfg.eval_batches)
            history["step"].append(step)
            history["train_loss"].append(tr)
            history["val_loss"].append(va)
            history["lr"].append(lr_at(step, tcfg))
            log(f"step {step:5d} | train {tr:.4f} | val {va:.4f} | ppl {math.exp(va):7.2f} "
                f"| lr {lr_at(step, tcfg):.2e} | {time.time() - t0:5.1f}s")
            meta = dict(meta_base, history=history, step=step, val_loss=va,
                        elapsed_s=time.time() - t0)
            if va < best_val:
                best_val = va
                meta["best_val_loss"] = va
                save_checkpoint(out / "best.pt", model, tok, meta)
            save_checkpoint(out / "last.pt", model, tok, dict(meta, best_val_loss=best_val))
            (out / "history.json").write_text(json.dumps(dict(meta, best_val_loss=best_val), indent=1))
        if step == tcfg.max_steps:
            break

        for g in opt.param_groups:
            g["lr"] = lr_at(step, tcfg)
        x, y = train_b.get()
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        opt.step()
        history["batch_loss"].append(round(loss.item(), 4))

    log(f"done in {time.time() - t0:.0f}s, best val loss {best_val:.4f} "
        f"(perplexity {math.exp(best_val):.2f}) -> {portable(out / 'best.pt')}")
    return {"best_val_loss": best_val, "history": history, "baselines": baselines}
