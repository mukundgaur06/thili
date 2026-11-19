"""Lesson 5: fine-tune all of GPT-2 on LEDGAR clauses with the Hugging Face Trainer.

    python train.py                       # 4,000 clauses, 1 epoch, ~40 min on a laptop CPU
    python train.py --n-train 60000       # the whole training split (GPU recommended)

Every weight of GPT-2 is trained (124M + the 101 new token rows). To fit in about 2 GB
of RAM on a CPU:

* **Adafactor** instead of AdamW. AdamW keeps two fp32 copies of every weight (about 1 GB
  for GPT-2). Adafactor keeps row and column averages instead, a few MB.
* **Logits only where graded.** ``clausegpt.loss.answer_loss`` replaces the stock loss,
  same value, about 40% less compute.
* **The first 64 tokens of each clause.** The median clause is 106 tokens; the opening
  words almost always say what the clause is about.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import clausegpt  # noqa: F401  (must come before transformers: environment fix-ups)
from transformers import Trainer, TrainerCallback, TrainingArguments
from clausegpt.data import Collator, encode, load_ledgar, sample
from clausegpt.loss import answer_loss
from clausegpt.model import load_base
from clausegpt.tagger import ClauseTagger

ROOT = Path(__file__).resolve().parent


class AnswerLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = answer_loss(model, inputs["input_ids"], inputs["labels"], inputs["attention_mask"])
        return (loss, (loss,)) if return_outputs else loss   # eval only needs the loss


class TimeLimit(TrainerCallback):
    """Stop cleanly after `minutes`, saving a full checkpoint (weights + optimizer + data position).

    Running the same command again resumes from it, so a long run can be done in short pieces
    (useful on a laptop where a long process risks being killed when memory runs low).
    """

    def __init__(self, minutes: float) -> None:
        self.minutes = minutes
        self.t0 = time.time()

    def on_step_end(self, args, state, control, **kw):
        if (time.time() - self.t0) / 60 > self.minutes:
            control.should_save = True
            control.should_training_stop = True


class LogToFile(TrainerCallback):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.t0 = time.time()

    def on_log(self, args, state, control, logs=None, **kw):
        line = {"step": state.global_step, "elapsed_s": round(time.time() - self.t0), **(logs or {})}
        with self.path.open("a") as f:
            f.write(json.dumps(line) + "\n")
        print(line, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-dev", type=int, default=400)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-clause-tokens", type=int, default=64)
    ap.add_argument("--loss-on-prompt", action="store_true",
                    help="also train on predicting the clause text (default: only the type)")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "clause-gpt2"))
    ap.add_argument("--max-minutes", type=float, default=float("inf"),
                    help="pause after this long; re-run the same command to resume")
    ap.add_argument("--fresh", action="store_true", help="ignore any checkpoint and start over")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    ds, names = load_ledgar()
    train = sample(ds["train"], args.n_train, args.seed)
    dev = sample(ds["validation"], args.n_dev, args.seed)
    enc = lambda split: [encode(tok, t, names[l], args.max_clause_tokens, args.loss_on_prompt)  # noqa: E731
                         for t, l in zip(split["text"], split["label"])]

    model, tok = load_base(names)
    train_ex, dev_ex = enc(train), enc(dev)
    print(f"{len(train_ex)} train / {len(dev_ex)} dev clauses, vocab {len(tok)} "
          f"(+{len(names) + 1} new tokens), {sum(p.numel() for p in model.parameters()):,} params, all trainable")

    out = Path(args.out)
    work = ROOT / "outputs" / "trainer"
    log = ROOT / "results" / "train_log.jsonl"
    log.parent.mkdir(exist_ok=True)
    from transformers.trainer_utils import get_last_checkpoint
    resume = None if args.fresh or not work.exists() else get_last_checkpoint(str(work))
    if resume and not (Path(resume) / "optimizer.pt").exists():
        resume = None                                  # an old weights-only checkpoint cannot be resumed
    if resume is None:
        log.write_text("")
        (work / "minutes.txt").unlink(missing_ok=True) if work.exists() else None
    print("resuming from " + Path(resume).name if resume else "starting a new run")
    steps = int(len(train_ex) / args.batch * args.epochs)
    eval_every = max(25, steps // 5)
    targs = TrainingArguments(
        output_dir=str(work), use_cpu=True, seed=args.seed,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=32,
        num_train_epochs=args.epochs, learning_rate=args.lr, optim="adafactor",
        lr_scheduler_type="linear", warmup_steps=int(0.05 * steps), weight_decay=0.0,
        eval_strategy="steps", eval_steps=eval_every, save_strategy="steps", save_steps=eval_every,
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, save_only_model=False, prediction_loss_only=True,
        logging_steps=10, report_to=[], dataloader_num_workers=0, remove_unused_columns=False,
        disable_tqdm=True,
    )
    trainer = AnswerLossTrainer(model=model, args=targs, train_dataset=train_ex, eval_dataset=dev_ex,
                                data_collator=Collator(tok.pad_token_id),
                                callbacks=[LogToFile(log), TimeLimit(args.max_minutes)])
    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume)
    work.mkdir(parents=True, exist_ok=True)
    spent = (time.time() - t0) / 60
    prev = float((work / "minutes.txt").read_text()) if (work / "minutes.txt").exists() else 0.0
    minutes = prev + spent
    (work / "minutes.txt").write_text(str(minutes))
    if trainer.state.global_step < trainer.state.max_steps:
        print(f"paused at step {trainer.state.global_step}/{trainer.state.max_steps}; run the same command to resume")
        return

    ClauseTagger(trainer.model, tok, names, args.max_clause_tokens).save(str(out))
    rel = lambda p: Path(p).resolve().relative_to(ROOT).as_posix() if p and Path(p).resolve().is_relative_to(ROOT) else p  # noqa: E731
    meta = {**vars(args), "out": rel(args.out), "train_minutes": round(minutes, 1), "steps": trainer.state.global_step,
            "best_eval_loss": trainer.state.best_metric, "best_checkpoint": rel(trainer.state.best_model_checkpoint)}
    (ROOT / "results" / "train_run.json").write_text(json.dumps(meta, indent=1))
    print(f"done in {minutes:.1f} min; best dev loss {trainer.state.best_metric:.4f}; saved to {out}")


if __name__ == "__main__":
    main()
