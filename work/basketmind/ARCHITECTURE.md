# Architecture

## Data flow

```
 data/catalogue.py ─► data/generator.py ─────────────► data/dataset.py
 1,000 SKUs,           background baskets + 32 planted     public/:       catalogue, baskets, splits (80/10/10)
 popularity            itemsets (seeded)                   ground_truth/: planted.json, truth_itemsets.json
                                                                 │                 ▲ (exact miner labels the truth)
                ┌────────────────────────────────────────────────┤                 │
                ▼                                                ▼                 │
   recommenders/popularity.py                       data/verbalizer.py      mining/bitsets.py
   recommenders/cooccurrence.py ◄── mining/apriori.py ── rules.py    <basket> <p0042> … </basket>
   (popularity, co-occurrence, association rules)        │                  fixed / frozen / fresh order
                │                                         ▼
                │                         language_model/tokenizer.py  (domain vocab, name-mean init)
                │                         language_model/model.py      (GPT-2 + LoRA, tied head)
                │                         language_model/dataset.py    (padding, shifted labels, loss)
                │                         language_model/training.py   (resumable continued pretraining)
                │                         language_model/inference.py  (next-item log-probs, order-averaged)
                │                         language_model/generation.py (sample baskets, KV cache)
                │                                         │
                ▼                                         ▼
        recommenders/base.py  ◄──────────  recommenders/llm_ranker.py      (one interface: score(visible))
                │
                ▼
   evaluation/ranking.py   (cases, MRR, Hits@K, Recall@K, NDCG)     evaluation/itemsets.py (labels, recovery)
   evaluation/bootstrap.py (percentile and paired CIs)
                │
                ▼
   experiments/run.py  ─►  artifacts/<dataset>/{mining, baselines, comparisons}.json, lm/*.json, per_case/*.npz
                │
                ▼
   app/ (Streamlit, 9 pages) reads artifacts and runs live recommendations and generation
```

## Separation of the answer key

`data/<name>/public/` is all any model or recommender ever reads. `data/<name>/ground_truth/` (the
planted itemsets, their target and realised supports, the probability a basket was chosen for
planting, and the labelled exact truth) is read only by `evaluation/` and `experiments/` for
scoring, and by the dashboard pages that explain the results. `dataset.load()` never touches it;
`dataset.load_truth()` is a separate call.

## Model

```
<basket> <p0042> <p0917> …             1,003-token domain vocabulary
   │  wte: Embedding(1003, 768)        rows initialised to the mean GPT-2 embedding of the product's name; trained
   │  wpe: pretrained positions        frozen
   ▼
12 × GPT-2 block (pretrained, frozen)  + LoRA r=16 on attn.c_attn and mlp.c_fc
   ▼
ln_f (trained) → lm_head = wteᵀ (tied) → logits over 1,003 tokens; <basket>/<pad> masked to −∞
```

**Why not the whole GPT-2 vocabulary and full fine-tuning?** Full fine-tuning of GPT-2 with Adam
needs about 2 GB before activations. On the build machine (under 1 GB free) a full-FT step paged
to 8–34 s. Replacing the 50,257-row vocabulary with the 1,003 tokens baskets use removes 150 MB of
weights and gradients. Freezing the blocks and training LoRA adapters brings a step to about
3–5 s at about 1 GB. `LMConfig(full_finetune=True)` trains every weight instead, on a machine
that can afford it.

## Recommender interface

```python
class Recommender:
    def fit(self, baskets, n_items) -> Recommender
    def score(self, visible) -> np.ndarray          # one score per catalogue item
    def score_many(self, cases) -> np.ndarray       # batched (the LM does one forward pass per batch)
    def recommend(self, visible, k) -> [(item, score)]   # never returns a visible item
```

Every recommender, classical or language model, is evaluated by the same code on the same cases.
`recommenders/crosssell.py` turns any score vector into a cross-sell list (no other SKU of a base
product already in the basket), and keeps upsell (a pricier SKU of the same base product) as a
separate, rule-based list.

## Evaluation pipeline

1. `make_cases`: held-out test baskets are split by a fixed seed into a visible half and a target
   half.
2. Each model scores all 1,000 products per case; visible items are excluded.
3. `ranking.evaluate` stores **per-case** metrics, saved in `per_case/<model>.npz`.
4. `bootstrap_ci` gives intervals per model; `paired_bootstrap` compares models on identical
   cases (`comparisons.json`).
5. `probe`: the trained model writes 10,000 baskets. They are mined with the same exact Apriori
   and compared with the planted itemsets and with the *ceiling*: the exact miner on a sample of
   real baskets of the same size.

## Resumable training

`training.train(..., max_minutes=…)` stops at a step boundary and saves the adapter weights,
optimizer state, the batch iterator's RNG state and the history. Re-running the same command
continues exactly where it stopped. The 45-minute main run was trained in pieces of under 9
minutes, so no single process ran long enough to be lost to a memory spike.
