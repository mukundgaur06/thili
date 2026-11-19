# LoRA from Scratch: parameter-efficient fine-tuning of real GPT-2

This project fine-tunes OpenAI's pretrained **GPT-2 (124M)** for SST-2 sentiment
classification. The pretrained weights stay frozen; only a tiny low-rank adapter is trained.
Everything LoRA-specific is written by hand:

* the `LoRALinear` layer
* injection into GPT-2's attention
* adapter save, load and merge
* hot-swapping adapters
* a 4-bit **NF4 QLoRA** base

No `peft`, no `bitsandbytes`, and no `transformers` modelling code is used.

Drill: *Build LoRA from Scratch* (LearnFast, Language Models track). It runs on CPU. It
needs `torch`, `safetensors` and `tokenizers` (all in the `lfllm` workbench), plus `pyarrow`
or `datasets` for SST-2.

```
            x ──────────────► W0 (frozen, 768×2304) ─────────┐
            │                                                ▼
            └─► A (r×768, random) ─► B (2304×r, ZERO) ─► ×α/r ─► (+) ─► h = W0x + (α/r)·BAx
```

## Where each lesson lives

| Lesson | Code |
|---|---|
| 1. Why not fine-tune everything? | memory maths below; `finetune.py --mode full` exists, but needs about 2.5 GB RAM |
| 2. The maths of low rank | `LoRALinear.delta_weight`. ΔW = (α/r)·B·A has rank ≤ r and r·(in+out) parameters instead of in·out. B = 0 at init, so training starts exactly at the pretrained model |
| 3. Build `LoRALinear` | `lora/layers.py`: wraps `nn.Linear` **or** GPT-2's transposed `Conv1D`; kaiming-init A, zero B, scaling α/r, dropout, `merge`/`unmerge` |
| 4. Inject it into GPT-2 | `lora/gpt2.py` loads the official `model.safetensors` into a compact GPT-2 with Hugging Face's module names. `lora/inject.py:inject_lora` wraps every `c_attn` (the fused Q, K, V projection) and freezes everything else |
| 5. Train just the adapter | `finetune.py`: AdamW on A, B and the 2-way classification head only. The head is trained in every mode (PEFT's `modules_to_save`) |
| 6. Does it match full fine-tuning? | `finetune.py` modes `head` / `lora` / `qlora` / `full`; `compare_full_ft.py` is a same-optimizer full-FT vs LoRA comparison |
| 7. Merge & swap | `merge_lora` / `unmerge_lora` / `strip_lora`, `AdapterBank` (one frozen base, many adapters); demo in `merge_and_swap.py` |
| 8. QLoRA | `lora/qlora.py`: NF4 levels **derived from normal quantiles** (they match the published table to 1e-7), block-wise absmax (64), 4-bit packing (2 codes per byte), double quantization of the scales, on-the-fly dequantization in `Linear4bit` |

## Results on real GPT-2 (SST-2)

Setup:

* **Data:** 4,000 training sentences, 500 dev sentences held out from train for picking the
  best step, and the 872-sentence official validation set used as the untouched **test** set.
  No test sentence appears in training.
* **Training:** 2 epochs (500 steps, batch 16) on CPU.

| Method | Trainable params | % of GPT-2 | Saved file | Test accuracy | Peak RAM |
|---|---:|---:|---:|---:|---:|
| Head only (linear probe) | 1,536 | 0.001% | 6 KB | 82.0% | 1.10 GB |
| **LoRA r = 8** on `c_attn` | 296,448 | **0.24%** | **582 KB** | 86.0% | 1.59 GB |
| **LoRA r = 1** on `c_attn` | 38,400 | 0.03% | 78 KB | **88.0%** | 1.37 GB |
| LoRA r = 8, only 1,000 train sentences | 296,448 | 0.24% | 582 KB | 81.7% | 1.51 GB |
| **QLoRA r = 8**: same LoRA on a 4-bit NF4 base | 296,448 | 0.24% | 582 KB | 86.7% | **1.20 GB** |

Adapters are stored in fp16. GPT-2 itself is 498 MB in fp32, so the r = 8 adapter is
**0.12%** of the model's size.

How to read this:

* LoRA beats training the head alone by 4 to 6 points. Adapting attention helps, even with
  0.03% of the weights.
* With a quarter of the training data, LoRA r = 8 drops from 86.0% to 81.7%, about the
  same as training the head alone on all 4,000 sentences.
* r = 1 doing slightly better than r = 8 matches the paper's finding: the update needed
  for a task like this has a very low intrinsic rank.
* These are single runs on 872 test sentences. One standard error is about ±1.2 points, so
  r = 1 and r = 8 are not clearly different.
* For reference, Hu et al. (2021, Table 2) report RoBERTa-base on SST-2:
  full fine-tuning 94.8% vs LoRA 95.1%, with 0.24% of the parameters. LoRA matches full
  fine-tuning at full scale.

### Why full fine-tuning was not run here (lesson 1, in numbers)

| | Weights | Gradients | AdamW m + v | Total, before activations |
|---|---:|---:|---:|---:|
| Full fine-tuning, fp32 | 498 MB | 498 MB | 996 MB | **≈ 2.0 GB** |
| LoRA r = 8 | 498 MB (frozen) | 1.2 MB | 2.4 MB | **≈ 0.50 GB** |
| QLoRA r = 8 (NF4 base) | ≈ 200 MB | 1.2 MB | 2.4 MB | **≈ 0.20 GB** |

The QLoRA row assumes the 85M linear weights are NF4 at 4.13 bits each, and that the 39M
embedding parameters stay fp32.

The machine this was built on had under 1 GB of free RAM. LoRA fit (1.4 to 1.6 GB peak,
including activations and the Python/PyTorch runtime); full fine-tuning would not. That
is the motivating problem of the whole technique.

`compare_full_ft.py` therefore runs the like-for-like comparison where all methods fit:

* **Model:** the 0.87M-parameter mini-GPT from `../mini-gpt`, already pretrained.
* **Task:** pronoun consistency ("Tom … He" vs "Tom … She").
* **Runs:** head-only, LoRA r = 1/4/8 and full fine-tuning, 3 seeds each, all with the same
  AdamW budget.

### Full fine-tuning vs LoRA, same budget (mini-GPT)

Data: 3,000 train / 1,000 dev / 875 test examples. Each method ran with 3 seeds and the
same AdamW budget (`results/full_vs_lora_minigpt.json`).

| Method | Trainable params | % of model | Test accuracy (3 seeds) |
|---|---:|---:|---:|
| Head only | 256 | 0.03% | 72.8 – 79.7% |
| LoRA r = 1 | 2,304 | 0.27% | 99.9 – 100% |
| LoRA r = 4 | 8,448 | 0.96% | 100% |
| LoRA r = 8 | 16,640 | 1.88% | 99.9 – 100% |
| Full fine-tuning | 867,328 | 100% | 100% |

LoRA matches full fine-tuning while training 0.27% of the weights; the head alone does
not. The task is easy enough that every adapted model saturates, so this shows parity,
not where the two would differ.

### Merge and swap on real GPT-2

`merge_and_swap.py` with the r = 1 adapter (`results/merge_and_swap.json`):

| Check | Result |
|---|---|
| Max logit change after merging | 6.9e-6 (float rounding) |
| Test accuracy, unmerged → merged | 87.96% → 87.96% |
| Unmerge restores W0 | max diff 3.8e-6 |
| LoRA layers left after `strip_lora` | 0 |
| Activate an adapter | 8.8 ms |
| Reload the whole base model instead | 2,349 ms |
| Adapters disabled vs a fresh GPT-2 (LM logits) | 0.0 (identical) |
| Latency per 64×45 batch: unmerged / merged / plain | 3.9 s / 3.3 s / 3.9 s |

The latency gap between merged and plain is timing noise. At r = 1 the unmerged LoRA
branch costs nothing measurable.

The demo only swaps one adapter. The other saved adapters are rank 8 and do not fit the
rank-1 layout the base was set up for.

### QLoRA on real GPT-2

`python finetune.py --mode qlora --rank 8` quantizes every `c_attn`, `c_proj` and `c_fc` weight of
GPT-2 to NF4 (with double-quantized scales), then trains the same rank-8 LoRA on top:

| | LoRA r = 8 (fp32 base) | QLoRA r = 8 (NF4 base) |
|---|---:|---:|
| Base model stored | 475.8 MB | **193.6 MB** |
| Peak RAM while training | 1.59 GB | **1.20 GB** |
| Test accuracy | 86.0% | 86.7% |
| Training time (500 steps) | 476 s | 876 s |

The 4-bit base costs no accuracy (the 0.7-point difference is well inside ±1.2 points of
noise). It saves 59% of the base model's storage and 0.39 GB of training memory. The price
is time: every weight is dequantized on the fly in each forward and backward pass, 1.84×
slower on this CPU. The memory maths above predicted about 0.20 GB for the NF4 base; 193.6 MB
was measured. The log is `results/log-qlora-r8.txt`.

`results/run_all.sh` reproduces every run in order (about 1 hour on CPU). Close other
memory-heavy apps first.

## Commands

```bash
cd work/lora-from-scratch
python finetune.py --mode lora --rank 8                 # LoRA on Q,K,V (c_attn)
python finetune.py --mode lora --rank 4 --targets c_attn c_proj c_fc
python finetune.py --mode qlora --rank 8                # 4-bit NF4 frozen base
python finetune.py --mode head                          # linear-probe baseline
python finetune.py --mode full                          # all 124M weights (~2.5 GB RAM)
python merge_and_swap.py                                # merge check + adapter hot-swap
python compare_full_ft.py                               # full FT vs LoRA, same budget
python -m unittest discover -s tests -t .
streamlit run dashboard.py                              # demo dashboard (needs streamlit, plotly)
```

## Dashboard

`streamlit run dashboard.py` opens four tabs:

* **Results:** every measured number from `results/`.
* **Try the adapters:** type sentences and classify them with any trained adapter, hot-swapped
  onto one frozen GPT-2 in a few milliseconds.
* **LoRA maths:** how rank and target layers set the parameter count and memory.
* **QLoRA / NF4:** quantize a weight tensor to 4 bits live, NF4 vs uniform int4, with
  outliers and block size as controls.

The four trained adapters in `adapters/` are committed (6–582 KB each), so the live tab works
without retraining. It only needs GPT-2 in the Hugging Face cache.

On first use, GPT-2's weights are fetched into the Hugging Face cache
(`openai-community/gpt2`, 548 MB). SST-2 is read from the local `datasets` cache, or
downloaded with `datasets`. Each run appends to `results/results.json` and saves its adapter
to `adapters/<name>.safetensors`.

## Merging and swapping

* **Merge.** `W0 ← W0 + (α/r)·B·A`. After `strip_lora`, the model is a plain GPT-2 again,
  with zero inference overhead and identical outputs, up to float rounding. `unmerge`
  subtracts the update back out.
* **Swap.** `AdapterBank` keeps one frozen base in memory and switches tasks by copying a
  few hundred KB of A/B matrices, instead of reloading 500 MB. Disabling every adapter gives
  back the exact base GPT-2. This is how multi-adapter serving works (e.g. Predibase/LoRAX).
* **Merging into a 4-bit base is not lossless.** The 4-bit codes cannot absorb a small
  fp32 update, so you dequantize first. The code refuses to merge into a quantized base
  and raises an error.

## QLoRA, briefly

* **NF4.** The 16 levels sit at quantiles of N(0, 1). Pretrained weights are roughly normal,
  so every code is used about equally often. The tests check that the code-usage entropy is
  above 3.8 bits for NF4 and lower for uniform int4. NF4's reconstruction error is also
  lower than int4's.
* **Block-wise absmax.** Each block of 64 weights has its own scale, so an outlier only hurts
  its own block. This is tested.
* **Double quantization.** The fp32 scales cost 0.5 bit/param. Quantizing them to int8 in
  blocks of 256 reduces that to about 0.13 bit/param. Total storage is about 4.13 bits/param:
  **12.9% of fp32**.
* **Compute.** Weights are dequantized on the fly. Gradients flow through the dequantized
  weight to earlier layers, but never into the frozen integer codes.
* **Paged optimizers** use CUDA unified memory to survive GPU memory spikes. They do not
  apply on this CPU-only setup.

## Tests

`python -m unittest discover -s tests -t .` runs 28 tests. One compares this GPT-2
against Hugging Face's `GPT2Model`; it runs where `transformers` is installed (the
workbench) and is skipped otherwise. The suite checks that:

* a zero-initialised adapter gives exactly the base output;
* `forward` equals `W0 + (α/r)BA`, and ΔW has rank r;
* only A and B receive gradients, and dL/dA = 0 on the first step;
* merge and unmerge round-trip, and `strip_lora` leaves no LoRA layers;
* `Conv1D` orientation is handled correctly;
* injection freezes everything else, and training changes only adapter weights;
* adapter save/load round-trips and loading rejects bad keys or shapes;
* `AdapterBank` swapping works, and deactivating gives back the base model;
* NF4 matches the published table, beats int4, isolates outliers, packs and unpacks
  correctly, and double quantization is accurate;
* QLoRA gradients reach LoRA while the codes stay frozen.

## Layout

```
lora/layers.py     LoRALinear, Conv1D
lora/gpt2.py       GPT-2 that loads the official weights, classifier head, tokenizer
lora/inject.py     inject, freeze, count, save/load, merge/strip, AdapterBank
lora/qlora.py      NF4, block-wise quantization, double quant, Linear4bit
lora/data.py       SST-2 loading and leakage-free split, length-bucketed batches
finetune.py  compare_full_ft.py  merge_and_swap.py
adapters/          trained adapters (fp16 safetensors, KB-sized)
results/           results.json, training logs, run_all.sh
```
