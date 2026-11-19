# Legal-Clause GPT: fine-tuning GPT-2 to tag contract clauses

This project fine-tunes OpenAI's pretrained **GPT-2 (124M)** on **LEDGAR**: about 80,000
real contract provisions from SEC filings, each labelled with one of 100 clause types.
Given a clause, the model answers with its type:

```
"This Agreement shall be governed by the laws of the State of Delaware."  ->  <Governing Laws>
```

Drill: *Fine-Tune GPT-2 to Tag Legal Contract Clauses* (LearnFast, Language Models track).
It runs on CPU and needs `torch`, `transformers`, `datasets` and `scikit-learn` (the last
only for the baselines in `evaluate.py`).

## Results

GPT-2 was fine-tuned with the Trainer on 4,000 training clauses for 1 epoch: 500 steps of 8,
39 minutes on a laptop CPU, with every weight trained. Dev loss fell from 1.14 at step 100 to
0.56 at step 500. Scores are on 2,000 clauses from the official LEDGAR **test** split, which
was never used for training or model selection (`results/eval.json`).

| Model | Accuracy | Top-3 | Macro-F1 |
|---|---:|---:|---:|
| Majority class (always `<Governing Laws>`) | 5.4% | – | – |
| GPT-2 with the new tokens, **before** fine-tuning | 5.8% | 11.4% | 0.04 |
| **GPT-2 fine-tuned, free generation** | **73.2%** | – | – |
| **GPT-2 fine-tuned, constrained to the 100 types** | **73.4%** | **89.0%** | **0.56** |
| TF-IDF + logistic regression, same 4,000 clauses, same 64-token cut | 76.4% | – | 0.62 |
| TF-IDF + logistic regression, all 60,000 clauses | 86.8% | – | 0.79 |

* **It tags clauses.** Free generation (greedy over all 50k tokens) answers with a valid
  clause type 99.55% of the time. `model.generate` agrees with the one-pass decoding on 20 of
  20 checked clauses.
* **Before fine-tuning, GPT-2 never produces a type token** (0% free-generation accuracy) and
  is near chance when constrained.
* **Easy and hard types.** The best types are formulaic, with F1 0.92–0.99 for Counterparts,
  Severability, Notices, Use Of Proceeds, Confidentiality and Headings. The worst have
  near-synonyms among the labels:
  * Applicable Laws, F1 0.18 (13 of its errors are predicted as Governing Laws);
  * Tax Withholdings, F1 0.00 (predicted as Withholdings);
  * Miscellaneous and General, too vague to identify from the text.

  Many "errors" are LEDGAR's own label overlap.
* **It does not beat a linear baseline at this data size.** TF-IDF + logistic regression
  trained on the same 4,000 clauses is 3 points better, and with all 60,000 clauses it
  reaches 86.8%. This run used 7% of the training data, one epoch and the first 64 tokens
  per clause, all to fit a CPU. LexGLUE reports about 88% micro-F1 for fully trained legal
  transformers. More data and epochs (`python train.py --n-train 60000 --epochs 2`, GPU
  recommended) is the obvious next step.

Training is resumable: `python train.py --max-minutes 8` stops cleanly with a full
checkpoint, and running the same command again continues it. This run was done in 6 such
pieces.

## How classification becomes generation

Each clause type becomes **one new token**, and a separator token `<Label>` marks where the
answer starts. A training example is just text:

```
Except as otherwise set forth in this Debenture, the Company ... waives presentment ...<Label><Waivers><|endoftext|>
```

* **Tokenizer.** Before extending it, `<Governing Laws>` is 6 BPE pieces (`<`, `G`,
  `over`, `ning`, ` Laws`, `>`). After adding the 101 tokens it is a single id, 50305; the
  vocabulary grows from 50,257 to 50,358.
* **Embeddings.** `resize_token_embeddings` adds 101 rows. Each new row starts as the
  average of its label's words: `<Governing Laws>` starts at the mean of ' Governing' and
  ' Laws', not random noise. GPT-2 ties its output layer to its input embeddings, so this
  also initialises the row the model uses to *predict* that label.
* **Loss.** Only the two answer positions are graded, `<Type>` and `<|endoftext|>`. The
  clause tokens get label -100. Pass `--loss-on-prompt` to also train on the clause text.
* **Answering.** One forward pass gives the distribution over the next token after
  `<Label>`. There are two ways to read it:
  * **free generation**: greedy over all 50k tokens, exactly what `model.generate`
    returns (a test checks this). It can answer with something that is not a type.
  * **constrained**: argmax over the 100 type tokens only. The answer is always valid,
    and you get a probability for each type.

## Fitting full fine-tuning on a laptop CPU

Every one of GPT-2's weights is trained. Three choices keep that to about 1.9 GB of RAM and
about 4.5 s per batch of 8:

| Choice | Why |
|---|---|
| **Adafactor** (`optim="adafactor"` in the Trainer) | AdamW stores two extra fp32 copies of every weight, about 1 GB for GPT-2. Adafactor stores row and column averages instead, a few MB. |
| **Logits only at graded positions** (`clausegpt/loss.py`) | The stock loss multiplies every position by the 768 × 50,358 output matrix and then ignores nearly all of it. Same loss value (tested against `model(**batch).loss`), and a 20 s step became 4.5 s together with the next row. |
| **First 64 tokens of each clause** | The median clause is 106 tokens; the opening words usually say what the clause is about. The cut is applied *before* `<Label>`, so the answer is never cut off. |

The Trainer evaluates dev loss every 100 steps and keeps the best checkpoint
(`load_best_model_at_end`).

## LEDGAR at a glance

From `python explore.py` (full output in `results/explore.txt`):

* **Splits:** 60,000 train, 10,000 validation and 10,000 test clauses; 100 types.
* **Class balance:** the most common type, Governing Laws, is 5.3% of training, so always
  guessing it scores about 5%. The rarest, Books, has only 23 training examples.
* **Length:** in GPT-2 tokens, the median clause is 106, the 90th percentile 292 and the
  99th percentile 616.

## What `evaluate.py` reports

It scores the same seeded sample of test clauses (2,000 by default) with:

* **free-generation accuracy**, and how often the output is not a clause type at all;
* **constrained accuracy**, top-3 accuracy and **macro-F1** (rare types count as much as
  common ones);
* per-type F1, the most frequent confusions, and a check that `model.generate` agrees with
  the one-pass decoding.

Baselines, for scale:

* the majority class;
* GPT-2 with the new tokens but *before* fine-tuning;
* TF-IDF + logistic regression, trained on the same clauses with the same 64-token cut,
  and on all 60,000.

LexGLUE (Chalkidis et al., 2022) reports about 88% micro-F1 on LEDGAR for fully trained
legal BERT models. That is a reference point for a model trained on all 60,000 clauses
for several epochs, not for this 4,000-clause CPU run.

## Commands

```bash
cd work/legal-clause-gpt
python explore.py                          # lessons 2-4: the data, the new tokens, one example
python train.py                            # lesson 5: 4,000 clauses, 1 epoch, Adafactor
python train.py --n-train 60000 --epochs 2 # the whole training split (GPU recommended)
python tag.py "This Agreement may be executed in counterparts."      # lesson 6
python evaluate.py                         # lesson 7: accuracy, macro-F1, baselines
python app.py                              # lesson 8: web page on http://127.0.0.1:8610
python -m unittest discover -s tests -t .
```

The fine-tuned model, its tokenizer and `clause_labels.json` are saved together in
`outputs/clause-gpt2/`. That folder is about 500 MB and is not committed;
`ClauseTagger.load` reads it back.

The package hides a broken `torchaudio` from `transformers`. If the installed torchaudio
was built for a different torch, `transformers` crashes on import even for text models.
Where torchaudio works or is absent, this has no effect.

## Tests

`python -m unittest discover -s tests -t .` runs 12 tests on a tiny randomly initialised
GPT-2. They need the GPT-2 tokenizer in the local Hugging Face cache, but no network. They
check that:

* each label is a single new token, and the tokenizer splits it before the extension;
* new embedding rows start at the mean of their label's words, and the output layer stays
  tied to the input embeddings;
* only `<Type>` and `<|endoftext|>` are graded, and truncation never cuts off the answer;
* right-padding sets labels to -100 and the attention mask to 0 on padded positions;
* the graded-positions loss equals the stock loss, gradients included;
* free generation equals `model.generate`, batched equals one-by-one, and save/reload
  gives the same logits;
* a tiny model fine-tuned for 150 steps tags its four training clauses correctly.

## Layout

```
clausegpt/labels.py   label tokens, <Label> separator, prompt and training text
clausegpt/model.py    extend the tokenizer, resize and initialise embeddings
clausegpt/data.py     LEDGAR loading, encoding with answer-only labels, padding collator
clausegpt/loss.py     causal-LM loss computed only at graded positions
clausegpt/tagger.py   ClauseTagger: save/load, predict, tag, generate
train.py  evaluate.py  explore.py  tag.py  app.py  web/index.html
results/              explore.txt, train_log.jsonl, train_run.json, eval.json
```
