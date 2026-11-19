"""Continued pretraining, resumable in time-boxed segments.

AdamW on the trainable parameters, linear warm-up then cosine decay to 10% of the peak learning
rate, gradient clipping at 1.0. Every ``eval_every`` steps it records the validation loss with
baskets in a *random* order and in the *fixed* (sorted) order. The gap between the two shows
whether the model has learned something about presentation order rather than about baskets.

``max_minutes`` stops cleanly at a step boundary and saves everything needed to continue:
adapter weights, optimizer state, the batch iterator's RNG and history. Running the same
command again resumes exactly where it stopped. Long runs therefore survive in short pieces.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import torch

from .dataset import BasketBatches, encode_eval, next_token_loss, validation_loss
from .model import BasketLM


@dataclass
class TrainConfig:
    ordering: str = "fresh"
    steps: int = 600
    batch_size: int = 64
    lr: float = 1e-3
    warmup: int = 30
    weight_decay: float = 0.0
    eval_every: int = 50
    n_val: int = 512
    seed: int = 0
    threads: int = 8
    train_subset: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    progress = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


def train(model: BasketLM, train_baskets: Sequence[Sequence[int]], val_baskets: Sequence[Sequence[int]],
          cfg: TrainConfig, out_dir: Path, max_minutes: float = 1e9, log: Callable[[str], None] = print) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    batches = BasketBatches(train_baskets, model.vocab, cfg.batch_size, cfg.ordering, cfg.seed)
    val_random = encode_eval(val_baskets[:cfg.n_val], model.vocab, "fresh", seed=123)
    val_fixed = encode_eval(val_baskets[:cfg.n_val], model.vocab, "fixed")
    state_path = out_dir / "train_state.pt"
    history: List[dict] = []
    step = 0
    if state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"], strict=False)
        opt.load_state_dict(st["optimizer"])
        batches.restore(st["batches"])
        history, step = st["history"], st["step"]
        log(f"resumed at step {step}")
    elif step == 0:
        history.append({"step": 0, "train_loss": None, "val_loss_random": validation_loss(model, val_random, model.vocab.pad_id),
                        "val_loss_fixed": validation_loss(model, val_fixed, model.vocab.pad_id), "seconds": 0.0,
                        "presented": 0, "epoch": 0})
        log(f"step 0  val loss random {history[-1]['val_loss_random']:.3f}  fixed {history[-1]['val_loss_fixed']:.3f}")

    started = time.time()
    seconds_before = history[-1]["seconds"] if history else 0.0
    recent = []
    it = iter(batches)
    while step < cfg.steps:
        if (time.time() - started) / 60 > max_minutes:
            break
        model.train()
        ids, mask = next(it)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        loss = next_token_loss(model(ids, attention_mask=mask).logits, ids, mask, vocab=model.vocab)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1
        recent.append(loss.item())
        if step % cfg.eval_every == 0 or step == cfg.steps:
            row = {"step": step, "train_loss": sum(recent) / len(recent),
                   "val_loss_random": validation_loss(model, val_random, model.vocab.pad_id),
                   "val_loss_fixed": validation_loss(model, val_fixed, model.vocab.pad_id),
                   "seconds": round(seconds_before + time.time() - started, 1), "presented": batches.presented,
                   "epoch": batches.presented / len(train_baskets), "lr": lr_at(step - 1, cfg)}
            history.append(row)
            recent = []
            log(f"step {step}  train {row['train_loss']:.3f}  val random {row['val_loss_random']:.3f}  "
                f"val fixed {row['val_loss_fixed']:.3f}  epoch {row['epoch']:.2f}  {row['seconds']:.0f}s")
            torch.save({"model": model.trainable_state(), "optimizer": opt.state_dict(), "batches": batches.state(),
                        "history": history, "step": step}, state_path)

    done = step >= cfg.steps
    if not done:                      # paused by the time limit: save exactly where we are
        torch.save({"model": model.trainable_state(), "optimizer": opt.state_dict(), "batches": batches.state(),
                    "history": history, "step": step}, state_path)
    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    if done:
        model.save(out_dir, {"train": cfg.to_dict(), "steps_done": step, "params": model.counts()})
    return {"done": done, "step": step, "history": history}
