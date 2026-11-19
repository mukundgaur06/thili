# WordFlow — Statistical Next-Word Prediction Engine

**WORD FLOW · Statistical Language Modeling Laboratory**

A next-word predictor built from first principles: unigram, bigram and trigram language
models estimated by counting words, with Maximum Likelihood Estimation, add-k (Laplace)
smoothing, backoff, random text generation, and held-out cross-entropy / perplexity
evaluation. It comes with an interactive 8-page dashboard and a command-line interface.

No neural networks, no pretrained models or embeddings, no Transformers, no external APIs,
no GPU — and **no third-party dependencies at all**: the engine, the web server and the
dashboard use only the Python standard library and vanilla HTML/CSS/JS.

```
Input:  "I love machine"
         1. learning   P = 0.0032   (seen 1x)      ← computed from real counts, add-1
```

---

## Features

| Area | What is implemented |
|---|---|
| Preprocessing | lower-casing, apostrophe normalisation, punctuation handling, sentence splitting (with abbreviation guard), per-sentence `<s>`/`</s>` padding, `<unk>` mapping, configurable vocabulary (`min_count`, `max_vocab_size`) |
| Counting | generalised n-gram tables for every order `1..n` (`Counter` + `defaultdict`); no n-gram ever spans two sentences |
| Estimation | MLE `C(ctx,w)/C(ctx)`; add-k `(C(ctx,w)+k)/(C(ctx)+kV)` with any `k > 0` |
| Prediction | `suggest_next(text, top_k, smoothing, k, order)` with backoff to the longest seen context, unknown-word handling, counts and a plain-English explanation |
| Generation | `generate_text(seed_text, max_words, temperature, order, seed, k)`: seeded inverse-CDF sampling, temperature, fallback, bounded loop, full step trace |
| Evaluation | NLL, cross-entropy (nats and bits), perplexity, token count, OOV rate, zero-probability count; held-out vs training clearly separated; k sweep |
| Data | original 309-sentence sample corpus, `.txt` upload with validation, deterministic leakage-free split, retraining |
| Interface | 8-page dashboard + CLI (`main.py`) + JSON API |
| Tests | 93 `unittest` tests (also run under pytest) |

### The dashboard

| Page | What you can do |
|---|---|
| 1 Overview | project, language-model primer, architecture diagram, live corpus/vocabulary/n-gram stats, quick-start cards |
| 2 Live Prediction Studio | type and get suggestions as you type; **Tab** accepts the top word, **Alt+1…9** picks one; order / top-K / smoothing / k controls; probability chart, context used, fallback, counts, explanation; **sentence surprise heatmap** (per-word surprisal in bits); "continue writing" |
| 3 N-gram Explorer | browse unigram/bigram/trigram counts with search, min-count and boundary-token filters; click any row to see that context's next-word distribution |
| 4 Probability Laboratory | pick a context and a k; worked MLE and add-k calculation with real numbers, side-by-side distributions, unseen-word mass, normalisation checks |
| 5 Text Generation Lab | seed, order, length, temperature, random seed, sampling k; generate / regenerate; animated output and a step-by-step table of every sampling decision |
| 6 Model Evaluation | held-out cross-entropy and perplexity for unigram/bigram/trigram, MLE vs add-k vs backoff; perplexity-vs-k chart; training-set metrics shown separately |
| 7 Learn the Mathematics | eight sections (unigrams → perplexity) whose worked examples are computed live by the engine on an editable toy corpus |
| ◆ Corpus & Training | upload a `.txt`, choose sentence layout, order, min count, test fraction and seed, retrain |

Keyboard: `1`–`7` switch pages, `/` focuses the main input.

---

## Quick start

Requires Python 3.10+ (tested on 3.10.11 and 3.12.6). Nothing to install.

```bash
cd work/next-word-predictor

python app.py                 # dashboard at http://127.0.0.1:8501
python app.py --open          # ...and open the browser
python app.py --port 8600     # if 8501 is taken
python app.py --corpus my_book.txt   # start with your own corpus
```

Command line:

```bash
python main.py demo                         # split → train → predict → generate → evaluate
python main.py split                        # corpus/corpus.txt → corpus/train.txt + corpus/test.txt
python main.py train --order 3              # fit on train.txt, save models/wordflow_model.json
python main.py predict "the weather is" --no-smoothing
python main.py predict "i love machine" --k 0.1 --top-k 3
python main.py generate "the model" --seed 7 --temperature 0.8
python main.py evaluate --k 0.1             # held-out metrics on test.txt
```

`predict`, `generate` and `evaluate` only load the saved model — training and inference are
separate steps.

### Inside the LearnFast `lfllm` Docker workbench

`work/` is mounted into the container at `/home/learnfast/work`, and the image ships
Python 3.12, so no changes to the Docker setup are needed.

```bash
# from the repository root
docker compose -f lfllm/docker-compose.yml up        # JupyterLab at http://127.0.0.1:8889/?token=learnfast
```

Then open a terminal in JupyterLab:

```bash
cd ~/work/next-word-predictor
python -m unittest discover -s tests
python main.py demo
```

The compose file only publishes Jupyter's port, so to use the dashboard from the container
publish its port for a one-off run (the compose file stays untouched):

```bash
docker compose -f lfllm/docker-compose.yml run --rm -p 8501:8501 workbench \
  python /home/learnfast/work/next-word-predictor/app.py --host 0.0.0.0
# open http://127.0.0.1:8501
```

---

## Project structure

```
next-word-predictor/
├── app.py                 dashboard launcher (trains, then serves web/ + JSON API)
├── main.py                command-line interface
├── wordflow/              the engine (standard library only)
│   ├── preprocessing.py   normalise, split sentences, tokenize, pad, Vocabulary
│   ├── ngram_model.py     NGramModel: counting, lookups, backoff, save/load
│   ├── smoothing.py       MLE and add-k: the only place counts become probabilities
│   ├── predictor.py       suggest_next()
│   ├── generation.py      generate_text(), temperature, sampling
│   ├── evaluation.py      cross-entropy, perplexity, OOV, model comparison
│   ├── corpus.py          upload validation, cleaning, train/test split
│   ├── service.py         Workspace: the single trained model behind the UI
│   └── server.py          http.server JSON API
├── web/                   index.html, styles.css, app.js (no CDN, works offline)
├── corpus/                corpus.txt (sample) + train.txt / test.txt (generated split)
├── models/                saved model JSON (git-ignored, rebuilt on demand)
├── tests/                 93 unittest tests
├── README.md · MATHEMATICS.md · ARCHITECTURE.md
└── requirements.txt       (empty: no dependencies)
```

> The folder is called `corpus/` rather than `data/` because the workbench's root
> `.gitignore` ignores every `data/` directory, and the sample corpus must be committed.
> The engine package is `wordflow/` rather than `src/` so that imports read `from wordflow import …`.

---

## Usage examples (Python)

```python
from wordflow import NGramModel, NextWordPredictor, TextGenerator, evaluate, tokenize_sentences

sentences = tokenize_sentences("I love Python\nI love machine learning\nI love AI")
model = NGramModel(order=3).fit(sentences)

model.ngram_counts(2)[("i", "love")]            # 3
model.probability("machine", ("love",))          # 1/3            (MLE)
model.probability("machine", ("love",), k=1)     # (1+1)/(3+1·8) = 2/11

r = NextWordPredictor(model).suggest_next("I love", top_k=3, smoothing=False)
[(s.word, round(s.probability, 3)) for s in r.suggestions]   # [('ai', .333), ('machine', .333), ('python', .333)]

TextGenerator(model).generate_text("I love", seed=1).text
evaluate(model, [["i", "love", "ai"]], order=2, k=0).perplexity   # 3 ** 0.25
```

---

## Dataset

* `corpus/corpus.txt` — 309 original sentences (everyday English plus sentences about
  language models), one per line. Written for this project; no licence restrictions.
* Default split: 20 % test, seed 42 → **247 training / 62 test sentences**, V = 619.
* **Replace or extend it**: drop in any public-domain English text (e.g. a Project Gutenberg
  book saved as `.txt`). Use the dashboard upload with "Wrapped prose", or
  `python app.py --corpus book.txt`, or `python main.py split --corpus book.txt`.
  Nothing is ever downloaded automatically.
* Uploads must be `.txt`, non-empty, ≤ 8 MB, not binary, with ≥ 2 sentences. Control
  characters are stripped and line endings normalised.
* Leakage prevention: the split is made at sentence level *before* counting; sentences with
  identical tokens are grouped so a test sentence can never appear in training; the
  vocabulary comes from the training split only.

---

## Results on the sample corpus (held-out, 62 sentences, N = 564 tokens)

From `python main.py demo` (k = 1) and the dashboard's k sweep:

| Model | Perplexity | Notes |
|---|---:|---|
| Unigram / Bigram / Trigram, MLE | ∞ | 70 / 319 / 428 test tokens get P = 0 |
| Unigram + add-1 | 215.95 | |
| Bigram + add-1 | 358.21 | add-1 moves too much mass to unseen words |
| Trigram + add-1 | 461.32 | |
| Bigram + add-1 + backoff | 292.26 | |
| **Bigram + add-0.03** | **≈ 201.5** | best k for the bigram in the sweep |
| Trigram + add-0.01 | ≈ 297.5 | best k for the trigram |

OOV rate on the test split: 13.94 %. On a corpus this small, higher orders are too sparse
to beat a well-tuned bigram — exactly the data-sparsity lesson of n-gram models. Training-set
perplexities are much lower and are reported separately; they are not a quality measure.

---

## Mathematical foundations (summary)

| Quantity | Formula (as implemented) |
|---|---|
| Unigram | `P(w) = C(w) / N`, N = words + one `</s>` per sentence |
| Conditional MLE | `P(w \| ctx) = C(ctx, w) / C(ctx)` |
| Add-k / Laplace | `P(w \| ctx) = (C(ctx, w) + k) / (C(ctx) + k·V)`, V = words + `<unk>` + `</s>` |
| Temperature | `p'(w) = p(w)^(1/T) / Σ p(v)^(1/T)` — a sampling extension, not part of MLE |
| Cross-entropy | `H = −(1/N) Σ ln P(wᵢ \| ctxᵢ)` (nats; ÷ ln 2 for bits) |
| Perplexity | `PP = exp(H)` |

Full derivations and worked examples: [MATHEMATICS.md](MATHEMATICS.md).
Design and data flow: [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Testing

```bash
python -m unittest discover -s tests      # standard library, no install
python -m pytest tests                    # if pytest is available
```

93 tests in `tests/` cover tokenization, sentence boundaries, unigram/bigram/trigram
counts (including the spec example `I love: 3, love Python: 1, …`), MLE values,
normalisation of every distribution, Laplace/add-k values, unseen contexts, unknown
words, empty and one-word input, backoff, temperature, reproducible generation, bounded
generation, hand-computed cross-entropy and perplexity, infinite perplexity on zero
probabilities, OOV rate, train/test separation and determinism, corpus validation,
invalid parameters, and the API/service layer (including upload-and-retrain).

---

## Limitations

* Fixed, short context: a trigram sees two words; anything earlier is ignored.
* Data sparsity: most possible n-grams are never observed; longer orders get sparser fast.
* Add-k is a crude smoother; with a large V it over-smooths (see the results above).
  Kneser-Ney or interpolation would do much better.
* Vocabulary is closed; every new word is the same `<unk>`.
* Counts only co-occurrence — no notion of meaning, synonymy or grammar.
* The dashboard server is a single-user development server (Python's `http.server`).

## Future improvements

Linear interpolation with held-out-tuned weights; Kneser-Ney smoothing; Katz/stupid
backoff; character-level or sub-word models for OOV words; a beam-search autocomplete;
tuning k on a separate validation split rather than the test split.

---

## How this submission is checked, and how to submit

From the workbench's own README files: the `work/` folder **is** the submission. You push
your template copy of the repository to GitHub and paste its URL on the drill page. The
workbench states that *how you build is part of what gets assessed* and that *a steady
commit history is part of how your build is assessed* — so small, meaningful commits
matter, not just the final code. The exact rubric is not published in this repository;
expect a reviewer to read the code and docs, run it, and look at the commit history.
A good way to self-check before submitting:

1. `python -m unittest discover -s tests` passes.
2. `python main.py demo` runs end to end.
3. `python app.py` starts and every page works.
4. The README explains what was built and how to run it.

```bash
git status                                    # review what changed
git add work/
git commit -m "Describe the change"
git push origin main                          # your own template copy on GitHub
```

If the repository has no remote yet, create an empty repository on GitHub (or use
"Use this template"), then `git remote add origin https://github.com/<you>/<repo>.git`
followed by `git push -u origin main`.
