# BasketMind: LLM-powered market basket intelligence

**Research question:** can a small pretrained language model learn which products are bought
together from transaction data alone, and become a useful cross-selling engine?

BasketMind builds a synthetic shop with a **known ground truth**:

* **Shop:** 1,000 products and 100,000 baskets, with 32 itemsets planted at known supports.
* **Answer key:** an **exact bitset Apriori miner**.
* **Classical recommenders:** popularity, co-occurrence counting and association rules.
* **Language models:** GPT-2 off the shelf, and GPT-2 after **continued pretraining** on the
  baskets written out as text.

Everything is scored on the same held-out cases with bootstrap confidence intervals. The adapted
model is also probed by having it write 10,000 baskets of its own.

It is a reimplementation of the experiment in the reference repository
[thili-ai/thilimarketbasket](https://github.com/thili-ai/thilimarketbasket), redesigned and
extended; [`EXPERIMENTS.md`](EXPERIMENTS.md) lists the differences. Every number below was
measured on a laptop CPU; nothing is illustrative.

## Results

### Next-item prediction

2,000 held-out test baskets. For each, half the basket is shown and the rest must be predicted.
95% bootstrap CI in brackets.

| Model | MRR | Hits@10 | MRR when a planted set is split |
|---|---|---|---|
| Popularity | 0.135 [0.124, 0.147] | 0.281 | 0.329 |
| Co-occurrence (confidence) | 0.279 [0.262, 0.298] | 0.380 | 0.865 |
| **Association rules** (exact miner) | **0.297 [0.280, 0.316]** | **0.379** | **0.936** |
| GPT-2 off-the-shelf | 0.027 [0.023, 0.032] | 0.047 | 0.028 |
| **GPT-2 + continued pretraining** | 0.255 [0.239, 0.273] | 0.366 | 0.724 |

* **Continued pretraining** lifts GPT-2 from well below popularity (MRR 0.027) to 0.255, a
  9.4× gain, and beats popularity by +0.120 [+0.107, +0.133] (paired bootstrap).
* **It does not beat exact counting.** Association rules lead by 0.041 [0.031, 0.053] MRR.
  That comes after 0.4 epochs of LoRA training on a CPU, with validation loss still falling.
* **Order.** The adapted model scores the same whether a basket is presented shuffled or sorted
  (MRR 0.255 vs 0.256): it learned the baskets, not an order.

### Ordering ablation

Same data, steps and seed; only the presentation order differs.

| Ordering | Val loss, shuffled | Val loss, sorted | MRR |
|---|---:|---:|---:|
| fixed (sorted by id) | **7.89** (worse than the untrained 7.22) | **4.68** | 0.103 |
| frozen random | 5.87 | 5.85 | 0.143 |
| fresh random each time | 5.87 | 5.86 | 0.144 |

A fixed order plants a fake next-token signal ("the next id is larger"). The model learns that
instead of co-occurrence, and its recommendations fall below popularity.

### Generated baskets

The adapted model wrote 10,000 baskets, which were mined with the exact miner:

* **Product frequencies:** correlate 0.988 with the real shop.
* **Precision:** 70% of its frequent itemsets are truly frequent.
* **Planted itemsets:** only **9 of 32** re-created [25%, 31%], mostly the strongest (3%) ones,
  against **98.8%** for the exact miner on 10,000 real baskets.

It learned *what sells*, but *what sells together* only for the strongest patterns.

## What is in the box

| Deliverable | Where |
|---|---|
| Synthetic shop: 1,000-product catalogue, 100,000 baskets (plus a 10,000-basket dev set), planted itemsets, seeded | `basketmind/data/catalogue.py`, `generator.py`, `dataset.py` |
| Hidden ground truth, stored apart from the data models read | `data/<name>/ground_truth/` vs `data/<name>/public/` |
| Exact Apriori on vertical bitsets, checked against brute force | `basketmind/mining/bitsets.py`, `apriori.py` |
| Association rules (support, confidence, lift, leverage, conviction) | `basketmind/mining/rules.py` |
| Common recommender interface; popularity, co-occurrence (count, confidence, lift, blend), association rules | `basketmind/recommenders/` |
| Transaction verbalisation: one token per product, `<basket>` delimiters, fixed / frozen / fresh order | `basketmind/data/verbalizer.py` |
| Off-the-shelf GPT-2 baseline, scored with real next-token probabilities | `basketmind/language_model/tokenizer.py`, `inference.py` |
| Continued pretraining: causal mask, padding mask, shifted labels, cross-entropy, AdamW, checkpoints, validation loss, resumable | `basketmind/language_model/model.py`, `dataset.py`, `training.py` |
| Basket generation, parsing and probing with the exact miner | `basketmind/language_model/generation.py`, `experiments/run.py probe` |
| Cross-sell (with evidence) and upsell, kept separate | `basketmind/recommenders/crosssell.py` |
| MRR, Hits@K, Recall@K, NDCG; percentile and paired bootstrap | `basketmind/evaluation/` |
| 9-page Streamlit dashboard | `app/` |
| 29 tests | `tests/test_basketmind.py` |
| Docs | [`EXPERIMENTS.md`](EXPERIMENTS.md), [`MATHEMATICS.md`](MATHEMATICS.md), [`ARCHITECTURE.md`](ARCHITECTURE.md) |

## Setup

In the LearnFast **lfretail** workbench (it already provides torch, transformers, numpy and
pandas):

```bash
docker compose -f lfretail/docker-compose.yml up
# in the workbench terminal:
cd work/basketmind && pip install streamlit plotly
```

Or anywhere with Python 3.10+:

```bash
cd work/basketmind
pip install -e ".[train,app]"          # numpy, pandas, torch, transformers, streamlit, plotly
```

GPT-2 (`openai-community/gpt2`, 548 MB) is read from the Hugging Face cache. The package sets
`HF_HUB_OFFLINE=1`, so on a machine that has never downloaded it, fetch it once first:

```bash
HF_HUB_OFFLINE=0 python -c "from transformers import GPT2LMHeadModel; GPT2LMHeadModel.from_pretrained('openai-community/gpt2')"
```

## Reproduce

```bash
cd work/basketmind
R="python -m basketmind.experiments.run"
$R generate  --dataset full                 # 1 min: shop, baskets, ground truth, exact truth
$R mine      --dataset full                 # exact Apriori, brute-force check, sampling ceiling
$R baselines --dataset full                 # popularity, co-occurrence, rules, with bootstrap CIs
$R lm-eval   --dataset full --run offshelf  # GPT-2 before any training
# Ordering ablation: 4,000 baskets, 2 epochs each (about 13 min each on a laptop CPU)
for o in fixed frozen fresh; do
  $R lm-train --dataset full --run abl-$o --ordering $o --train-subset 4000 --steps 250 --batch-size 32
  $R lm-eval  --dataset full --run abl-$o
done
# Main continued-pretraining run (about 35 min on a laptop CPU). Resumable: add --max-minutes 8
# and re-run the same command until it says "finished".
$R lm-train --dataset full --run cpt-fresh --ordering fresh --steps 500 --batch-size 64
$R lm-eval  --dataset full --run cpt-fresh
$R probe    --dataset full --run cpt-fresh  # generate 10,000 baskets, mine, compare (about 10 min)
$R report   --dataset full                  # paired bootstrap comparisons
python -m unittest discover -s tests -t .   # 29 tests (7 need GPT-2 in the HF cache)
streamlit run app/app.py                    # the dashboard
```

Use `--dataset dev` (10,000 baskets) for a fast end-to-end run. The trained adapter for the main
run is committed (`artifacts/full/runs/cpt-fresh/adapter.pt`, 8 MB), so the dashboard's live
recommendations and generation work without retraining. The datasets themselves are regenerated
deterministically by `generate` and are not committed.

The lfretail container publishes only Jupyter's port, and its compose file is left unchanged.
Run the dashboard on the host, or add `"8501:8501"` to that file's `ports`.

## The dashboard

| Page | What it shows |
|---|---|
| **Executive Overview** | products, baskets, planted patterns, itemsets recovered, the selected model and its metrics, the pipeline diagram |
| **Retail Simulator** | catalogue browser, category shares, sales concentration, basket sizes, product popularity, a co-occurrence graph, a basket builder |
| **Association Mining** | levels and candidates, live Apriori at any support with exact counts and answer-key labels, a rule explorer (confidence, lift, item filter) |
| **LLM Training Lab** | model, tokenizer and vocabulary statistics, training configuration, loss curves (train, shuffled and sorted validation), checkpoint and hardware |
| **Prediction Benchmark** | MRR, Hits@1/5/10, Recall@10 and NDCG@10 with CIs for every model; all or planted-only cases; paired comparisons |
| **Basket Generation Lab** | the 10,000-basket probe against the ceiling by support level, per planted itemset; generate new baskets live |
| **Cross-Sell Studio** | build a basket and see every recommender's suggestions with support, confidence, lift, LLM probability and a plain-English reason; upsell shown separately |
| **Experiment Analytics** | the ordering ablation, classical vs LLM, generated vs ground truth |
| **Research & Mathematics** | support, confidence and lift worked on the live data, Apriori and bitsets, the causal LM loss, ranking metrics, the bootstrap |

All nine pages were run headlessly with Streamlit's `AppTest` with no exceptions, including live
generation and the Cross-Sell Studio with the language model loaded.

## Tests

`python -m unittest discover -s tests -t .` runs 29 tests. Metrics and mining are checked against
hand-computed values or brute force, never against the code under test. They cover:

* the catalogue (size, coverage, determinism);
* basket generation (one SKU per product, seeded);
* planting: realised supports counted independently, design rules;
* bitset counts against direct counting;
* Apriori against brute force, on random data and on generated baskets, and the Apriori property
  of its output;
* rule metrics by hand;
* itemset labels;
* verbalisation round trips, orderings and parsing;
* ranking metrics by hand (including ties and NDCG);
* case construction;
* bootstrap coverage and a paired-vs-unpaired check;
* recommender validity and evidence;
* the language model (with GPT-2 cached): name-based embedding initialisation, the causal mask
  (future tokens don't leak), padding invariance, label shifting and loss masking by hand, a
  training step lowering the loss and learning a pattern, generation validity, and save/load.

## Limitations

The main ones (details in [`EXPERIMENTS.md`](EXPERIMENTS.md#limitations)):

* the data is synthetic, so this says nothing yet about real-world performance;
* the training budget was set by a CPU with under 2 GB of free RAM (LoRA and a domain
  vocabulary instead of full fine-tuning, 0.4 epochs);
* any fixed order of a basket leaks a spurious signal;
* one-token-per-product does not generalise to unseen SKUs;
* rare itemsets are sparse;
* upsell has no ground truth here, so it is a rule, not a model.
