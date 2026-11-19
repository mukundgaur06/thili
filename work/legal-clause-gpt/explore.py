"""Lesson 2 to 4: meet LEDGAR, see a training example, and watch the tokenizer learn new words.

    python explore.py
"""

from __future__ import annotations

import collections

import numpy as np

import clausegpt  # noqa: F401  (must come before transformers)
from transformers import AutoTokenizer

from clausegpt.data import encode, load_ledgar
from clausegpt.labels import training_text
from clausegpt.model import BASE, extend_tokenizer


def main() -> None:
    ds, names = load_ledgar()
    print("LEDGAR (LexGLUE):", {k: len(v) for k, v in ds.items()}, f"| {len(names)} clause types")
    counts = collections.Counter(ds["train"]["label"])
    print("most common:", ", ".join(f"{names[k]} {v}" for k, v in counts.most_common(6)))
    print("rarest:     ", ", ".join(f"{names[k]} {v}" for k, v in counts.most_common()[-5:]))
    top = counts.most_common(1)[0]
    print(f"majority-class guess would score {top[1] / len(ds['train']):.1%}")

    tok = AutoTokenizer.from_pretrained(BASE)
    lens = np.array([len(tok(t).input_ids) for t in ds["train"].select(range(3000))["text"]])
    print("clause length in GPT-2 tokens (3,000 sample): "
          + ", ".join(f"p{p} {int(np.percentile(lens, p))}" for p in (50, 90, 99)))

    print("\nOne clause per type, first 160 characters:")
    seen = set()
    for t, l in zip(ds["train"]["text"][:400], ds["train"]["label"][:400]):
        if l not in seen and len(seen) < 6:
            seen.add(l)
            print(f"  <{names[l]}>  {t[:160]!r}")

    word = "<Governing Laws>"
    print(f"\nBefore extending the tokenizer, {word!r} is {len(tok(word).input_ids)} pieces: "
          f"{[tok.decode([i]) for i in tok(word).input_ids]}")
    added = extend_tokenizer(tok, names)
    print(f"after adding {added} tokens it is {tok(word).input_ids} (one id); vocab {len(tok)}")

    clause, label = ds["train"]["text"][0], names[ds["train"]["label"][0]]
    ex = encode(tok, clause, label, max_clause_tokens=24)
    print("\nTraining text:", repr(training_text(clause[:80] + " ...", label)))
    print("Encoded with a 24-token cut; only the last two positions are graded:")
    for i, l in zip(ex["input_ids"][-5:], ex["labels"][-5:]):
        print(f"  {tok.decode([i])!r:>20}  label {tok.decode([l]) if l != -100 else '-100 (ignored)'}")


if __name__ == "__main__":
    main()
