# WordFlow Architecture

## Data flow

```
 corpus.txt / upload.txt
        │  corpus.py: decode_bytes → clean_text → detect_line_mode → split_sentences
        ▼
 sentences (str) ──► split_train_test(test_fraction, seed)   ── deterministic, duplicate-safe
        │                         │
        │ train                   │ test (never counted)
        ▼                         │
 preprocessing.tokenize           │
        ▼                         │
 Vocabulary.build(train only)     │
        ▼                         │
 NGramModel.fit                   │
   pad each sentence: <s>… w … </s>
   tables[m][context] = Counter(next word)   for m = 1..n
        ▼                         │
 smoothing.probability(count, context_count, k, V)
   k = 0 → MLE      k > 0 → add-k
        │                         │
   ┌────┴─────────────┐           │
   ▼                  ▼           ▼
 predictor          generation   evaluation
 suggest_next()     generate_text()   evaluate()/compare_models()
   │                  │               │  H, PP, OOV on held-out tokens
   └──────────┬───────┴───────────────┘
              ▼
 service.Workspace (one TrainedState = single source of truth)
              ▼
 server.py JSON API  ◄──►  web/app.js dashboard        main.py CLI (loads saved model)
```

**Corpus → Preprocessing → Tokenization → N-gram Counting → Probability Estimation →
Smoothing → Prediction / Generation → Evaluation.**

## Modules

| Module | Responsibility | Depends on |
|---|---|---|
| `preprocessing.py` | normalisation, sentence splitting, tokenization, padding, special tokens, `Vocabulary` | — |
| `smoothing.py` | `mle_probability`, `add_k_probability`, `validate_k` — the only place a count becomes a probability | — |
| `ngram_model.py` | `NGramModel`: count tables for every order, lookups, distributions, context resolution, backoff, JSON persistence | preprocessing, smoothing |
| `predictor.py` | `NextWordPredictor.suggest_next`, ranking, explanations | ngram_model |
| `generation.py` | `TextGenerator.generate_text`, temperature, inverse-CDF sampling | ngram_model, predictor |
| `evaluation.py` | `evaluate`, `score_sentence`, `compare_models` | ngram_model |
| `corpus.py` | upload validation, cleaning, layout detection, `split_train_test` | preprocessing |
| `service.py` | `Workspace`: trains, holds and swaps the active `TrainedState`; JSON-ready query methods | everything above |
| `server.py` | `http.server` API: parameter validation, routing, static files | service |
| `app.py` / `main.py` | entry points: dashboard and CLI | service / engine |

## Design decisions

* **One model holds every order.** An order-n `NGramModel` stores the tables for orders
  1..n, each built exactly as a standalone model of that order would build it (its own
  `<s>` padding). The unigram, bigram and trigram views in the UI therefore come from one
  trained object, and the bigram table inside a trigram model is identical to a pure
  bigram model's (tested).
* **Totals are derived from the tables.** `C(context)` is the sum of its follower
  counts, so MLE distributions normalise exactly by construction.
* **One estimator path.** The model, predictor, generator and evaluator all call
  `smoothing.probability`, so a formula exists exactly once.
* **Training and evaluation share `sentence_events`**, which guarantees identical padding
  and context extraction on both sides.
* **Training is separate from inference.** `Workspace.train` builds an immutable
  `TrainedState` and swaps it in under a lock. Queries only read. The CLI's
  `predict`/`generate`/`evaluate` load the saved JSON and never touch the corpus.
* **The UI always reflects the current model.** Every API response carries the model
  `version`. If the dashboard sees a version it does not expect, it re-fetches the state
  and re-renders. The sidebar always shows the active corpus and version.
* **No dependencies.** The `lfllm` image has no Streamlit or web framework, and adding one
  would mean rebuilding a multi-GB image. The standard-library server and a vanilla-JS
  front end run unchanged in the container and offline.
* **Honest numbers.** Zero probabilities produce infinite perplexity instead of a hidden
  epsilon. `<unk>` and `</s>` are never shown as suggestions, and their probability mass is
  reported rather than renormalised away. The JSON encoder runs with `allow_nan=False`.
* **Leakage-free split.** Sentence-level, seeded, with duplicate sentences grouped. The
  vocabulary is built after the split, from training only.

## Why n-gram models are limited

* **Fixed context length.** A trigram sees two words. In "The keys that I left on the
  table *are*…", the verb agrees with a noun six words back.
* **Data sparsity.** The number of possible n-grams grows as Vⁿ, but a corpus contains only
  a tiny fraction of them. In the sample corpus 1,977 distinct trigrams occur among
  2,343 trigram events, so most are seen once. Smoothing patches the zeros but cannot
  invent knowledge.
* **Vocabulary growth.** New words keep appearing (Heaps' law). Every one of them is the
  same `<unk>`, and V, and with it the model's memory, grows with the corpus.
* **No semantic understanding.** "cat" and "kitten" are unrelated symbols. Seeing
  "the cat sleeps" tells the model nothing about "the kitten sleeps".
* **Limited generalisation.** Probabilities come only from exact matches of the context.
  There is no notion of similar contexts.

## Why neural language models were developed

Neural language models (Bengio et al., 2003, then RNNs/LSTMs and Transformers) replace
count tables with learned functions:

* Words become **dense vectors**, so similar words share statistical strength and the model
  generalises to word combinations it has never seen.
* The context is encoded by a network rather than looked up exactly, so it can be **much
  longer** (attention in Transformers covers thousands of tokens) without an exponential
  table.
* Sub-word tokenization removes most of the out-of-vocabulary problem.

The cost is opacity, compute and data. An n-gram model is still the clearest way to see
what a language model *is*: a conditional probability distribution over the next token,
judged by cross-entropy on held-out text. Neural models are trained and evaluated by
exactly the same objective.
