"""Does LoRA match full fine-tuning?  A fair, same-budget comparison.

Full fine-tuning of GPT-2 with AdamW needs weights + gradients + two Adam moments:
124M x 16 bytes, about 2 GB, plus activations. The machine this was built on had under 1 GB
free. So the head-to-head comparison runs on a model where all three strategies fit: the
0.87M-parameter mini-GPT from ../mini-gpt, pretrained on stories. All strategies use the
same optimizer, steps, data and seeds; only the set of trainable weights differs.

Task: pronoun consistency. Given a story opening, is the pronoun right?
  "Tom had a red ball. He loved the ball very much."   -> 1 (consistent)
  "Tom had a red ball. She loved the ball very much."  -> 0 (one pronoun flipped)
The answer needs information from a name many tokens back, so the model has to use
attention. That is exactly what LoRA on the attention projections adapts.

    python compare_full_ft.py            # head / LoRA r=1,4,8 / full, 3 seeds each
"""

from __future__ import annotations

import json
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
MINI = ROOT.parent / "mini-gpt"
sys.path.insert(0, str(MINI))

from minigpt.corpus import make_stories  # noqa: E402
from minigpt.trainer import load_checkpoint  # noqa: E402

from lora.inject import count_params, inject_lora  # noqa: E402

FLIP = {"He": "She", "She": "He", "he": "she", "she": "he"}


def make_examples(n: int, tok, max_tokens: int, seed: int):
    """Balanced consistent / flipped story openings that fit in the context window."""
    rng = random.Random(seed)
    out, seen = [], set()
    stories = make_stories(n * 3, seed=seed + 100)    # different seed from mini-GPT's training corpus
    for story in stories:
        sents = re.split(r"(?<=[.!?\"]) (?=[A-Z\"])", story)
        text = ""
        for s in sents:
            cand = (text + " " + s).strip()
            if len(tok.encode(cand)) > max_tokens:
                break
            text = cand
        words = text.split(" ")
        idx = [i for i, w in enumerate(words) if w in FLIP]
        if not idx or text in seen:
            continue
        seen.add(text)
        label = rng.random() < 0.5
        if not label:
            i = idx[-1]                       # flip the last subject pronoun
            words[i] = FLIP[words[i]]
        out.append((" ".join(words), int(label)))
        if len(out) >= n:
            break
    return out


class Classifier(nn.Module):
    def __init__(self, gpt) -> None:
        super().__init__()
        self.gpt = gpt
        self.score = nn.Linear(gpt.cfg.n_embd, 2, bias=False)
        nn.init.normal_(self.score.weight, std=0.02)

    def forward(self, idx, lengths):
        g = self.gpt
        x = g.drop(g.tok_emb(idx) + g.positions(idx.size(1), idx.device))
        for b in g.blocks:
            x = b(x)
        h = g.ln_f(x)
        return self.score(h[torch.arange(idx.size(0)), lengths - 1])


def batches(examples, tok, bs, shuffle, rng):
    order = list(range(len(examples)))
    if shuffle:
        rng.shuffle(order)
    for i in range(0, len(order), bs):
        chunk = [examples[j] for j in order[i:i + bs]]
        seqs = [tok.encode(t) for t, _ in chunk]
        T = max(map(len, seqs))
        x = torch.zeros(len(seqs), T, dtype=torch.long)
        for r, s in enumerate(seqs):
            x[r, :len(s)] = torch.tensor(s)
        yield x, torch.tensor([len(s) for s in seqs]), torch.tensor([y for _, y in chunk])


@torch.no_grad()
def accuracy(model, data, tok):
    model.eval()
    right = sum(int((model(x, l).argmax(-1) == y).sum()) for x, l, y in batches(data, tok, 128, False, None))
    model.train()
    return right / len(data)


def run(mode: str, rank: int, seed: int, data, tok, steps: int, lr: float):
    torch.manual_seed(seed)
    gpt, _, _ = load_checkpoint(MINI / "checkpoints" / "best.pt")
    gpt.cfg.dropout = 0.0
    for m in gpt.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
    model = Classifier(gpt)
    if mode == "lora":
        inject_lora(model, ("qkv",), r=rank, alpha=2 * rank)
    elif mode == "head":
        for p in model.parameters():
            p.requires_grad_(False)
    else:
        for p in model.parameters():
            p.requires_grad_(True)
    model.score.weight.requires_grad_(True)
    params = count_params(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    train, dev, test = data
    rng = random.Random(seed)
    step, best_dev, best_state, t0 = 0, -1.0, None, time.time()
    while step < steps:
        for x, l, y in batches(train, tok, 32, True, rng):
            loss = F.cross_entropy(model(x, l), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % 100 == 0 or step == steps:
                d = accuracy(model, dev, tok)
                if d > best_dev:
                    best_dev = d
                    best_state = {k: v.detach().clone() for k, v in model.named_parameters() if v.requires_grad}
            if step >= steps:
                break
    with torch.no_grad():
        for k, v in model.named_parameters():
            if k in best_state:
                v.copy_(best_state[k])
    return {"mode": mode, "rank": rank if mode == "lora" else None, "seed": seed,
            "trainable": params["trainable"], "percent": params["percent"],
            "dev": best_dev, "test": accuracy(model, test, tok), "time_s": round(time.time() - t0, 1)}


def main() -> None:
    torch.set_num_threads(4)
    _, tok, _ = load_checkpoint(MINI / "checkpoints" / "best.pt")
    ex = make_examples(5000, tok, max_tokens=60, seed=7)
    data = (ex[:3000], ex[3000:4000], ex[4000:])
    print(f"examples: {len(data[0])} train / {len(data[1])} dev / {len(data[2])} test, "
          f"{sum(y for _, y in ex) / len(ex):.2f} positive")
    configs = [("head", 0, 3e-3), ("lora", 1, 3e-3), ("lora", 4, 3e-3), ("lora", 8, 3e-3), ("full", 0, 1e-3)]
    results = []
    for mode, rank, lr in configs:
        for seed in (0, 1, 2):
            r = run(mode, rank, seed, data, tok, steps=600, lr=lr)
            r["lr"] = lr
            results.append(r)
            print(f"{mode:5s} r={rank or '-':>2} seed {seed}: trainable {r['trainable']:>7,} ({r['percent']:6.2f}%) "
                  f"| dev {r['dev']:.3f} | test {r['test']:.3f} | {r['time_s']}s", flush=True)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "full_vs_lora_minigpt.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
