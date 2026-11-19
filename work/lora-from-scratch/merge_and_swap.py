"""Lesson 7: merge an adapter into the base weights, and hot-swap adapters at serve time.

    python merge_and_swap.py

1. Load GPT-2 once, inject empty LoRA layers, then load a trained adapter (a few hundred KB).
2. Merge:  W <- W + (alpha/r) B A.  Check the logits are unchanged, the LoRA branch is
   gone, and latency equals the plain base model's.
3. Unmerge and swap: several adapters share ONE frozen base. Switching costs a
   state-dict copy of the A/B matrices, not a 500 MB model reload.
4. Disable every adapter, and the base GPT-2 language model is bit-for-bit back.
"""

from __future__ import annotations

import glob
import json
import time
from pathlib import Path

import torch

from lora.data import Batches, load_sst2
from lora.gpt2 import GPT2Classifier, load_gpt2, load_tokenizer
from lora.inject import (AdapterBank, inject_lora, load_adapter, merge_lora, read_adapter,
                         set_lora_enabled, strip_lora, unmerge_lora)
from lora.layers import LoRALinear

ROOT = Path(__file__).resolve().parent


def timeit(fn, reps=5):
    fn()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t) / reps * 1000


@torch.no_grad()
def accuracy(model, batches):
    right = total = 0
    for x, l, y in batches:
        right += int((model(x, l).argmax(-1) == y).sum())
        total += y.numel()
    return right / total


def model_for(meta: dict) -> GPT2Classifier:
    m = GPT2Classifier(load_gpt2()).eval()
    inject_lora(m, meta["targets"].split(","), r=int(meta["rank"]), alpha=float(meta["alpha"]))
    return m


def main() -> None:
    torch.set_num_threads(8)
    files = sorted(f for f in glob.glob(str(ROOT / "adapters" / "lora-*.safetensors")))
    if not files:
        raise SystemExit("no adapters yet: run `python finetune.py --mode lora` first")
    tok = load_tokenizer()
    _, _, test = load_sst2()
    test_b = list(Batches(test, tok, 64, shuffle=False))
    x, lengths, _ = test_b[0]
    report = {}

    # ------------------------------------------------------------------ merge
    adapter, meta = read_adapter(files[0])
    print(f"adapter {Path(files[0]).name}: {Path(files[0]).stat().st_size / 1024:.0f} KB, "
          f"{sum(v.numel() for v in adapter.values()):,} numbers, meta {meta}")
    model = model_for(meta)
    load_adapter(model, adapter)
    with torch.no_grad():
        before = model(x, lengths)
    acc_unmerged = accuracy(model, test_b)
    t_unmerged = timeit(lambda: model(x, lengths))
    merge_lora(model)
    with torch.no_grad():
        after = model(x, lengths)
    diff = (before - after).abs().max().item()
    acc_merged = accuracy(model, test_b)
    t_merged = timeit(lambda: model(x, lengths))
    unmerge_lora(model)
    with torch.no_grad():
        diff_unmerge = (model(x, lengths) - before).abs().max().item()
    strip_lora(model)                                     # merged and wrapper removed entirely
    n_lora = sum(isinstance(m, LoRALinear) for m in model.modules())
    with torch.no_grad():
        diff_strip = (model(x, lengths) - before).abs().max().item()
    t_plain = timeit(lambda: model(x, lengths))
    print(f"merge: max |logit diff| = {diff:.2e}, test acc {acc_unmerged:.4f} -> {acc_merged:.4f}")
    print(f"unmerge restores W0: max diff {diff_unmerge:.2e} | stripped model has {n_lora} LoRA layers, diff {diff_strip:.2e}")
    print(f"latency for one batch of {x.shape[0]}x{x.shape[1]}: unmerged {t_unmerged:.0f} ms, "
          f"merged {t_merged:.0f} ms, plain GPT-2 {t_plain:.0f} ms")
    report["merge"] = {"adapter": Path(files[0]).name, "max_logit_diff": diff, "acc_unmerged": acc_unmerged,
                       "acc_merged": acc_merged, "unmerge_diff": diff_unmerge, "stripped_diff": diff_strip,
                       "ms_unmerged": round(t_unmerged, 1), "ms_merged": round(t_merged, 1),
                       "ms_plain": round(t_plain, 1)}
    del model

    # ------------------------------------------------------------------ swap
    same = [f for f in files if read_adapter(f)[1]["targets"] == meta["targets"]
            and read_adapter(f)[1]["rank"] == meta["rank"]]
    others = [f for f in files if f not in same]
    print(f"\nswap: {len(files)} adapters on disk; one base model serves those with the same LoRA layout")
    base = model_for(meta)
    bank = AdapterBank(base)
    ref_lm = None
    pure = load_gpt2().eval()
    prompt = torch.tensor([tok.encode("The movie was").ids])
    with torch.no_grad():
        ref_lm = pure.lm_logits(prompt)
    for f in same:
        bank.add(Path(f).stem, read_adapter(f)[0])
    swaps = {}
    for name in bank.adapters:
        t = time.perf_counter()
        bank.activate(name)
        swaps[name] = {"swap_ms": round((time.perf_counter() - t) * 1000, 2), "test_acc": accuracy(base, test_b)}
        print(f"  activate {name:28s} in {swaps[name]['swap_ms']:6.2f} ms -> test acc {swaps[name]['test_acc']:.4f}")
    bank.deactivate()
    with torch.no_grad():
        lm_diff = (base.transformer.lm_logits(prompt) - ref_lm).abs().max().item()
    print(f"  deactivate -> base GPT-2 LM logits differ from a fresh GPT-2 by {lm_diff:.2e}")
    t = time.perf_counter()
    _ = load_gpt2()
    reload_ms = (time.perf_counter() - t) * 1000
    print(f"  for comparison, reloading the whole base model takes {reload_ms:.0f} ms")
    report["swap"] = {"adapters": swaps, "base_lm_diff_when_disabled": lm_diff, "full_reload_ms": round(reload_ms),
                      "other_layouts": [Path(f).stem for f in others]}
    set_lora_enabled(base, True)
    (ROOT / "results" / "merge_and_swap.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
