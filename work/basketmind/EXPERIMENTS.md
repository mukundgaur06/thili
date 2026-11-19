# Experiments

Every number here was produced by `python -m basketmind.experiments.run …` on the `full` dataset
and is stored under `artifacts/full/`. Intervals are 95% percentile bootstraps (2,000 resamples,
seed 1) over the 2,000 evaluation cases unless stated otherwise. Hardware: one Intel laptop CPU,
no GPU.

## Hypotheses

1. **H1.** The exact miner recovers every planted itemset from the full data. This is a
   sanity check of the data and the miner.
2. **H2.** Counting recommenders beat popularity on next-item prediction, most of all on baskets
   that contain part of a planted itemset.
3. **H3.** Off-the-shelf GPT-2 knows little about this shop: its prior over product names should
   not beat popularity.
4. **H4.** Continued pretraining on verbalized baskets teaches GPT-2 the co-occurrence structure:
   it should beat popularity and the off-the-shelf model by a wide margin and approach the counting
   baselines.
5. **H5.** A fixed presentation order creates an artificial next-token signal. A model trained on
   it learns the order rather than the baskets.
6. **H6.** Baskets *generated* by the adapted model contain the planted itemsets at roughly
   their planted supports.

## Dataset design

* **Catalogue:** 1,000 SKUs = 95 base products × invented brands × variants, in 12 categories
  (Bakery, Dairy, Beverages, Pantry, Snacks, Produce, Electronics, Home Appliances, Beauty, Sports,
  Fashion, Books). Popularity follows a four-level hierarchy; the top 20% of SKUs make 56% of sales.
* **Baskets:** 100,000, seed 0. Background baskets draw 2 + NegBin (mean 6) base products without
  replacement by popularity, then one SKU each. After planting: mean 7.18 distinct products, max 31.
* **Planted itemsets:** 32. For each of four supports (3%, 1%, 0.4%, 0.15%) and each size 2–5,
  one *natural* group (Bread + Butter, Coffee + Milk, Smartphone + Phone Case, Laptop + Wireless
  Mouse, Pasta + Pasta Sauce, Shampoo + Conditioner + Body Wash, …) and one *surprising*
  cross-category group. Rules:
  * chance co-occurrence is at most 30% of the target support;
  * no planted set contains another;
  * a base product uses one SKU across every set.

  All 32 land exactly on target (`ground_truth/planted.json`).
* **Splits:** 80,000 train, 10,000 validation and 10,000 test baskets, by a seeded permutation.
  The 2,000 evaluation cases come from the test split only.
* **Answer key:** kept in `data/full/ground_truth/`, which no model reads.

## Exact mining (H1)

Apriori on vertical bitsets at min support 0.1% (100 baskets), max size 5: 5,953 frequent itemsets
in 7.4 s.

| Level | Candidates counted | Frequent |
|---|---:|---:|
| 1 | 1,000 | 928 |
| 2 | 430,128 | 2,073 |
| 3 | 17,474 | 1,341 |
| 4 | 1,323 | 1,099 |
| 5 | 540 | 512 |

* **Brute-force check:** on a 2,000-basket sample (support 0.5%, size ≤ 4), Apriori and brute
  force agree exactly.
* **Planted itemsets:** all 32 are found.
* **What else is frequent:** 300 parts of planted sets, 2,776 extensions (a planted set plus
  other items), 1,917 chance co-occurrences of popular items and 928 single items. So "frequent"
  and "planted" are different things, and recovery is scored on planted itemsets.
* **Top rules:** 162 of the 200 highest-lift rules come from planted sets.
* **Sampling ceiling:** the exact miner, given a random sample of real baskets, recovers
  96.1% [95.0, 97.3] of planted itemsets from 2,000 baskets and 98.8% [97.8, 99.5] from 10,000
  (20 samples each). At 0.15% support it misses some, because 15 expected baskets fall below
  the 10-basket threshold by chance. No model given a 10,000-basket sample can be expected to
  beat this.

## Next-item prediction

**Protocol.** Each test basket with ≥ 2 products is split by seed into a visible half (⌈n/2⌉
items) and targets (the rest). Every product not visible is a candidate.

* **MRR:** 1 / rank of the best-ranked target.
* **Hits@K:** at least one target in the top K.
* **Recall@K:** the share of the targets that land in the top K.
* **Ties:** count against the target.

The language models average log-probabilities over 3 random orders of the visible items, so no
single arbitrary order decides their answer. *Planted cases* (544 of 2,000) are cases where a
planted itemset is split between the visible items and the targets.

| Model | MRR | Hits@1 | Hits@5 | Hits@10 | Recall@10 | MRR, planted cases |
|---|---|---:|---:|---|---:|---|
| Popularity | 0.135 [0.124, 0.147] | 0.066 | 0.208 | 0.281 [0.262, 0.301] | 0.102 | 0.329 [0.300, 0.360] |
| Co-occurrence (count) | 0.265 [0.249, 0.283] | 0.211 | 0.316 | 0.373 [0.352, 0.394] | 0.165 | 0.817 [0.786, 0.842] |
| Co-occurrence (confidence) | 0.279 [0.262, 0.298] | 0.230 | 0.318 | 0.380 [0.360, 0.402] | 0.170 | 0.865 [0.839, 0.888] |
| Co-occurrence (lift) | 0.219 [0.203, 0.237] | 0.191 | 0.249 | 0.263 [0.245, 0.283] | 0.112 | 0.785 [0.753, 0.813] |
| Co-occurrence (blend) | 0.281 [0.264, 0.300] | 0.233 | 0.319 | 0.378 [0.357, 0.399] | 0.169 | 0.878 [0.854, 0.901] |
| **Association rules** | **0.297 [0.280, 0.316]** | **0.255** | **0.329** | 0.379 [0.358, 0.401] | **0.171** | **0.936 [0.918, 0.953]** |
| GPT-2 off-the-shelf | 0.027 [0.023, 0.032] | 0.008 | 0.024 | 0.047 [0.038, 0.057] | 0.014 | 0.028 [0.023, 0.034] |
| **GPT-2 + continued pretraining** | 0.255 [0.239, 0.273] | 0.193 | 0.311 | 0.366 [0.346, 0.388] | 0.158 | 0.724 [0.692, 0.756] |

**Findings:**

* **H2 is confirmed.** Every counting method beats popularity. On planted cases association
  rules reach MRR 0.94: it sees two items of a planted set and ranks the third first.
  Pure *lift* ranks worse than *confidence*, because it promotes rare items that co-occurred a
  few times.
* **H3 is confirmed, strongly.** Off-the-shelf GPT-2 is far *below* popularity. Product
  embeddings initialised from GPT-2's own embeddings of the names give it a prior, but that prior
  is about what words follow what, not what shoppers buy together. It does not help.
* **H4 is mostly confirmed.**
  * Continued pretraining moves GPT-2 from MRR 0.027 to 0.255, a 9.4× gain, past popularity, and
    to within about 0.03–0.04 of the counting baselines on all cases.
  * It still trails them clearly. Paired bootstrap over the same 2,000 cases (`comparisons.json`):

    | Comparison | MRR difference | 95% CI | P(adapted LM better) |
    |---|---:|---|---:|
    | adapted LM − association rules | −0.041 | [−0.053, −0.031] | 0.000 |
    | adapted LM − co-occurrence (confidence) | −0.024 | [−0.035, −0.014] | 0.000 |
    | adapted LM − co-occurrence (count) | −0.010 | [−0.020, +0.000] | 0.027 |
    | adapted LM − co-occurrence (lift) | +0.036 | [+0.023, +0.049] | 1.000 |
    | adapted LM − popularity | +0.120 | [+0.107, +0.133] | 1.000 |
    | adapted LM − off-the-shelf GPT-2 | +0.228 | [+0.210, +0.246] | 1.000 |

    It beats popularity, off-the-shelf GPT-2 and lift-ranked co-occurrence, is
    at most 0.02 behind raw co-occurrence counts (the interval just reaches 0), and loses to
    confidence-ranked co-occurrence and to association rules. On planted cases the gap to
    association rules is 0.21 MRR (0.72 vs 0.94).
  * The model saw 32,000 baskets (0.4 epochs, 500 steps of 64) in 34 minutes of CPU, and its
    validation loss was still falling (5.750 → 5.728 over the last 100 steps). The gap is
    therefore a *lower bound* on what this model can do, not a ceiling.
  * The counting methods read all 80,000 training baskets, exactly.
* **Order robustness.** The adapted model gives the same MRR whether the visible items are
  presented shuffled (0.255) or sorted (0.256), and its validation loss is the same either way
  (5.722 vs 5.721). It learned the baskets, not an order.

## Ordering ablation (H5)

Three runs, identical except for the ordering policy: the same 4,000 training baskets, 2 epochs
(250 steps of 32), the same seed and learning rate.

| Run | Val loss, shuffled | Val loss, sorted | MRR (shuffled prefixes) | MRR (sorted prefixes) | MRR, planted cases |
|---|---:|---:|---|---:|---|
| **fixed** (sorted by id) | **7.891** (start 7.220) | **4.682** | 0.103 [0.093, 0.115] | 0.090 | 0.282 |
| frozen (one random order per basket) | 5.869 | 5.850 | 0.143 [0.131, 0.155] | 0.145 | 0.344 |
| fresh (new order every time) | 5.869 | 5.864 | 0.144 [0.132, 0.155] | 0.143 | 0.368 |

**H5 is confirmed.** The fixed-order model's loss on *sorted* baskets falls to 4.68, far below
anything the shuffled models reach. Meanwhile its loss on *shuffled* baskets rises from 7.22 to
7.89, worse than not training at all.

Sorting plants a free signal: the next id is always larger than the last, and `</basket>`
comes after the largest. The model spends its capacity on that. Its next-item MRR is 28% below
the shuffled models', and below plain popularity (0.135). Even when it is given sorted prefixes
it does no better (0.090), because the hidden targets are not the "next larger ids".

Frozen and fresh orders are indistinguishable at 2 epochs: the frozen order is random, so there
is no global rule to learn, and two passes are too few to memorise per-basket orders. Fresh
permutations are the safe default for longer training, and the main run uses them.

## Generated baskets vs ground truth (H6)

The adapted model wrote 10,000 baskets from the bare prompt `<basket>` at temperature 1.0 (534 s). The
generated baskets were mined with the same exact Apriori at 0.1% support.

| | Generated by the LM | Real data |
|---|---:|---:|
| Mean basket size | 6.73 | 7.18 |
| Correlation of per-product frequency | 0.988 | – |
| Repeated items written (dropped) | 2.4% | 0 |
| Invalid tokens / baskets cut off / empty | 0 / 0% / 0.2% | – |
| Share of frequent pairs within one category | 14.5% | 13.5% |
| Multi-item frequent itemsets | 2,598 | 5,025 |
| …of which are really frequent in the shop (precision) | **70.5%** | 100% |

**Planted itemsets re-created** (support ≥ 0.1% among the generated baskets):

| Planted support | LM, 10,000 generated baskets | Exact miner on 10,000 real baskets (ceiling) |
|---|---:|---:|
| 3% | 7 / 8 | 100% |
| 1% | 2 / 8 | 100% |
| 0.4% | 0 / 8 | 100% |
| 0.15% | 0 / 8 | 95% |
| **All** | **9 / 32 = 28% [25%, 31%]** | **98.8% [97.8, 99.5]** |

The interval resamples the generated baskets 100 times and re-mines each sample.

**H6 is only partly supported.**

* The model reproduces the shop's **marginals** almost perfectly: product frequencies correlate
  0.988, and basket sizes are close.
* Most itemsets it treats as frequent really are frequent (70% precision).
* But it reproduces the planted **co-occurrences** far more weakly than they exist:
  * the 3% itemsets come out at a mean support of 0.45%, about 7× too rare;
  * nothing planted at 0.4% or below clears the threshold.

Its frequent itemsets, labelled by the answer key:

| Label | Count |
|---|---:|
| chance co-occurrences of popular items | 1,290 |
| extensions of planted sets | 383 |
| parts of planted sets | 150 |
| planted | 9 |
| not frequent in the real data | 766 |

The model has learned "what sells" well and "what sells *together*" only for the strongest
patterns. This matches the next-item results:

* it ranks well when a basket contains part of a strong planted set (MRR 0.72 on planted cases);
* it trails exact counting on the long tail.

Generating from a model trained for 0.4 epochs mostly samples its learned item popularity. Joint
structure needs more training: the reference's 800 steps of 16 on a GPU, or several epochs.

## Limitations

* **Synthetic-to-real shift.** Baskets are independent draws plus planted sets. Real baskets have
  seasonality, store layout, promotions, households and repeated purchases. Results say whether the
  method *can* learn planted co-occurrence, not how it performs on a real shop.
* **Training budget.** 0.4 epochs of LoRA continued pretraining on a laptop CPU, forced by
  memory. The classical baselines read every training basket exactly. More steps, a GPU or full
  fine-tuning (`LMConfig(full_finetune=True)`) would narrow the gap; how far is untested.
* **Ordering bias.** Shown above. Any fixed serialisation of a set leaks a spurious signal.
* **Catalogue vocabulary.** One token per product makes ids exact and scoring cheap, but every
  new SKU needs a new embedding row and training. A name-based verbalisation (the reference
  repo's choice) generalises to unseen SKUs but costs about 6 tokens per product and a far more
  expensive candidate scoring.
* **Hallucinated ids.** With a closed domain vocabulary the model cannot invent a non-existent
  product id. It can still repeat an item or end a basket early, and those cases are counted
  (`share_repeated`, `share_cut_off`). A name-based model can also misspell products; that
  failure mode does not exist here by construction.
* **Sparsity.** Itemsets at 0.15% support appear in about 120 training baskets. The adapted model
  saw about 48 of them; even the exact miner on a 10,000-basket sample loses some of them.
* **Uncertainty.** 2,000 cases give MRR intervals of about ±0.02. Differences smaller than that
  need the paired comparisons, which cancel case difficulty.
* **Upsell** has no ground truth in this data (variant choice ignores the rest of the basket), so
  it is a transparent rule, not a learned or evaluated model.

## Differences from the reference implementation (thili-ai/thilimarketbasket)

| | Reference | This project |
|---|---|---|
| Shop | Indian grocery, 32 planted itemsets at 5% / 1% / 0.2% / 0.05% | Hypermarket across 12 categories (groceries to electronics), 32 planted at 3% / 1% / 0.4% / 0.15% |
| Ground truth | stored alongside the baskets | separate `ground_truth/` folder that loaders do not read |
| Model | SmolLM2-135M, full continued pretraining with MLX (Apple GPU) | GPT-2 124M (already cached; no extra download), LoRA + domain vocabulary, PyTorch on CPU |
| Verbalisation | product names, about 6 BPE tokens each | one token per product id (exact ids, one-pass scoring), embeddings initialised from the names |
| Evaluation | leave-one-out, MRR / hits@K | half-basket split with multi-item targets (MRR, Hits@K, Recall@K, NDCG), order-averaged LM scores, bootstrap and paired-bootstrap CIs, planted-case subset |
| Ordering | fresh shuffle | fixed / frozen / fresh ablation |
| Baselines | popularity, item-to-item counting | popularity, four co-occurrence variants, association rules from the exact miner |
