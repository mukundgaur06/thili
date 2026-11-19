"""Train the mini-GPT.

    python train.py                      # default: generated stories, ~0.9M params, 1500 steps
    python train.py --max-steps 300      # quick smoke run
    python train.py --corpus my.txt --vocab-size 1024 --n-layer 6 --n-embd 192
"""

import argparse
from pathlib import Path

from minigpt.corpus import write_corpus
from minigpt.model import GPTConfig
from minigpt.trainer import TrainConfig, portable, train

ROOT = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    t, m = TrainConfig(), GPTConfig()
    p.add_argument("--corpus", default=str(ROOT / "data" / "stories.txt"))
    p.add_argument("--out-dir", default=str(ROOT / "checkpoints"))
    p.add_argument("--stories", type=int, default=3000, help="stories to generate if the default corpus is missing")
    for name, default in [("vocab_size", t.vocab_size), ("batch_size", t.batch_size), ("max_steps", t.max_steps),
                          ("eval_every", t.eval_every), ("warmup_steps", t.warmup_steps), ("seed", t.seed),
                          ("threads", t.threads), ("block_size", m.block_size), ("n_layer", m.n_layer),
                          ("n_head", m.n_head), ("n_embd", m.n_embd)]:
        p.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    for name, default in [("lr", t.lr), ("min_lr", t.min_lr), ("weight_decay", t.weight_decay),
                          ("dropout", m.dropout), ("val_fraction", t.val_fraction)]:
        p.add_argument("--" + name.replace("_", "-"), type=float, default=default)
    p.add_argument("--pos-encoding", choices=["learned", "sinusoidal"], default=m.pos_encoding)
    a = p.parse_args()

    corpus = Path(a.corpus)
    if not corpus.exists():
        if corpus.resolve() != (ROOT / "data" / "stories.txt").resolve():
            raise SystemExit(f"corpus not found: {corpus}")
        corpus.parent.mkdir(parents=True, exist_ok=True)
        size = write_corpus(corpus, a.stories, a.seed)
        print(f"generated {a.stories} stories ({size:,} chars) -> {portable(corpus)}")

    tcfg = TrainConfig(corpus=str(corpus), out_dir=a.out_dir, vocab_size=a.vocab_size,
                       val_fraction=a.val_fraction, batch_size=a.batch_size, max_steps=a.max_steps,
                       eval_every=a.eval_every, lr=a.lr, min_lr=a.min_lr, warmup_steps=a.warmup_steps,
                       weight_decay=a.weight_decay, seed=a.seed, threads=a.threads)
    mcfg = GPTConfig(block_size=a.block_size, n_layer=a.n_layer, n_head=a.n_head, n_embd=a.n_embd,
                     dropout=a.dropout, pos_encoding=a.pos_encoding)
    train(tcfg, mcfg)


if __name__ == "__main__":
    main()
