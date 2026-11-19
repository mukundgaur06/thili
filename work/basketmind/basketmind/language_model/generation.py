"""Have the model write baskets, and parse them.

Generation starts from ``<basket>`` and samples one token at a time, with a key/value cache, until
``</basket>`` or ``max_items``. ``<basket>`` and ``<pad>`` can never be sampled. Repeated
products and over-long baskets are allowed in sampling but *counted* when parsed, so the
statistics show how often the model breaks the rules of a basket.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from ..data.verbalizer import decode


@torch.no_grad()
def sample_baskets(model, n: int, temperature: float = 1.0, batch: int = 250, max_items: int = 30,
                   seed: int = 0) -> Tuple[List[List[int]], dict]:
    model.eval()
    vocab = model.vocab
    g = torch.Generator().manual_seed(seed)
    baskets, repeats, invalid, cut, sizes = [], 0, 0, 0, []
    for start in range(0, n, batch):
        size = min(batch, n - start)
        ids = torch.full((size, 1), vocab.open_id, dtype=torch.long)
        out = model(ids, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1]
        written = [[] for _ in range(size)]
        done = torch.zeros(size, dtype=torch.bool)
        for _ in range(max_items + 1):
            logits = logits.float() / temperature
            logits[:, vocab.open_id] = -float("inf")
            logits[:, vocab.pad_id] = -float("inf")
            nxt = torch.multinomial(torch.softmax(logits, -1), 1, generator=g).squeeze(1)
            for r in torch.nonzero(~done).flatten().tolist():
                written[r].append(int(nxt[r]))
            done |= nxt == vocab.close_id
            if bool(done.all()):
                break
            out = model(nxt[:, None], past_key_values=past, use_cache=True)
            past, logits = out.past_key_values, out.logits[:, -1]
        for w in written:
            parsed = decode(w, vocab)
            repeats += parsed["repeats"]
            invalid += parsed["invalid"]
            cut += not parsed["closed"]
            baskets.append(parsed["items"])
            sizes.append(len(parsed["items"]))
    written_items = sum(sizes) + repeats
    stats = {"baskets": len(baskets), "mean_items": float(np.mean(sizes)), "median_items": float(np.median(sizes)),
             "share_repeated": repeats / max(1, written_items), "invalid_tokens": invalid,
             "share_cut_off": cut / max(1, len(baskets)), "share_empty": float(np.mean([s == 0 for s in sizes])),
             "temperature": temperature, "seed": seed}
    return baskets, stats
