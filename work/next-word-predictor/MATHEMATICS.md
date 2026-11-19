# The Mathematics of WordFlow

Every formula here is the one the code implements; the function that implements it is
named next to it. The running example is the toy corpus used in the tests and on the
dashboard's *Learn the Mathematics* page:

```
I love Python
I love machine learning
I love AI
```

After tokenization: 3 sentences, 10 words, vocabulary
`{ai, i, learning, love, machine, python}` (6 words).

---

## 0. Tokens, padding and V

Each sentence is padded **independently** (`preprocessing.pad_sentence`): an order-n model
prepends `n − 1` start tokens `<s>` and appends one end token `</s>`.

```
bigram : <s> i love python </s>
trigram: <s> <s> i love python </s>
```

* `<s>` only ever appears in a context. It is never predicted.
* `</s>` is a real outcome: the model must predict where a sentence ends.
* `<unk>` replaces every word outside the training vocabulary.

Because sentences are padded separately, no n-gram spans two sentences
(`python i` is never counted).

The number of possible outcomes is

```
V = |words| + 1 (<unk>) + 1 (</s>) = 6 + 2 = 8          (Vocabulary.size)
```

---

## 1. Unigram model

```
P(w) = C(w) / N
```

N is the number of predicted tokens: every word plus one `</s>` per sentence,
N = 10 + 3 = 13.

| w | C(w) | P(w) |
|---|---:|---:|
| i | 3 | 3/13 = 0.2308 |
| love | 3 | 0.2308 |
| `</s>` | 3 | 0.2308 |
| python, machine, learning, ai | 1 each | 0.0769 each |

Σ P = 13/13 = 1.

---

## 2. Bigram and trigram models (Markov assumption)

By the chain rule, `P(w₁ … w_m) = Π P(wᵢ | w₁ … wᵢ₋₁)`. An n-gram model assumes only
the last `n − 1` words matter:

```
bigram : P(wᵢ | wᵢ₋₁)            = C(wᵢ₋₁, wᵢ)        / C(wᵢ₋₁)
trigram: P(wᵢ | wᵢ₋₂, wᵢ₋₁)      = C(wᵢ₋₂, wᵢ₋₁, wᵢ)  / C(wᵢ₋₂, wᵢ₋₁)
```

Bigram counts of the toy corpus (the specification's example):

| bigram | count |
|---|---:|
| `<s>` i | 3 |
| i love | 3 |
| love python | 1 |
| love machine | 1 |
| love ai | 1 |
| machine learning | 1 |
| python `</s>`, learning `</s>`, ai `</s>` | 1 each |

`C(context)` is the number of times the context occurred **as a history**, which equals the
sum of its follower counts (`NGramModel.context_count`). Storing it that way guarantees
`Σ_w C(ctx, w) = C(ctx)` exactly.

---

## 3. Conditional probability

```
P(B | A) = P(A, B) / P(A)
```

With relative frequencies `P(A, B) = C(A, B)/N₂` and `P(A) = C(A)/N₂`, the total `N₂`
cancels:

```
P(machine | love) = [C(love, machine)/N₂] / [C(love)/N₂] = C(love, machine) / C(love) = 1/3
```

---

## 4. Maximum Likelihood Estimation

For one context with follower counts `c_w` (total `C = Σ c_w`) we look for the
distribution `p` that makes the training data most likely:

```
maximise   L(p) = Σ_w c_w · ln p_w        subject to   Σ_w p_w = 1
```

Lagrangian: `Λ = Σ c_w ln p_w − λ (Σ p_w − 1)`

```
∂Λ/∂p_w = c_w / p_w − λ = 0      ⇒  p_w = c_w / λ
Σ_w p_w = 1                      ⇒  λ = Σ c_w = C
                                 ⇒  p_w = c_w / C
```

So the MLE is the relative frequency (`smoothing.mle_probability`):

```
P(python | love) = P(machine | love) = P(ai | love) = 1/3      Σ = 1
P(love | i) = 3/3 = 1
P(python | i) = 0/3 = 0          ← never observed, so impossible under MLE
```

**The zero-probability problem.** Every unseen event gets probability 0. If a test sentence
contains a single unseen n-gram, its probability is 0, `ln 0 = −∞`, and cross-entropy and
perplexity are infinite. An unseen *context* is worse: `C(ctx) = 0`, so the MLE is
`0/0` and undefined. (The code returns 0 there and the evaluator reports infinity rather
than inventing a floor.)

---

## 5. Add-k (Laplace) smoothing

Pretend each of the V possible outcomes was seen `k` extra times after every context
(`smoothing.add_k_probability`):

```
P_k(w | ctx) = (C(ctx, w) + k) / (C(ctx) + k·V)
```

`k = 1` is Laplace smoothing; any `k > 0` is allowed (`k = 0` is plain MLE, negative or
non-finite `k` raises `ValueError`).

**Normalisation.** `Σ_w (C(ctx, w) + k) = C(ctx) + k·V`, which is exactly the
denominator, so the V probabilities always sum to 1. The tests check this for every
context, several k values and unseen contexts.

**Worked example** (k = 1, V = 8, context `love`, C = 3):

| w | count | MLE | add-1 |
|---|---:|---:|---:|
| python | 1 | 0.3333 | (1+1)/(3+8) = 0.1818 |
| machine | 1 | 0.3333 | 0.1818 |
| ai | 1 | 0.3333 | 0.1818 |
| each of the other 5 tokens | 0 | 0 | (0+1)/(3+8) = 0.0909 |
| **Σ** | 3 | 1 | 3·(2/11) + 5·(1/11) = **1** |

**What changes, mathematically.**

* Seen words are discounted: `(c+k)/(C+kV) < c/C` whenever `c/C > 1/V`.
* Unseen words gain `k/(C+kV) > 0`.
* Total mass moved to unseen outcomes: `(V − d)·k / (C + kV)`, d = distinct followers.
  Here 5/11 = 45 % — large, because V is large relative to C.
* An unseen context (`C = 0`) becomes uniform: `k / (k·V) = 1/V`.
* Effective count: `c* = (c + k) · C / (C + kV)`.

As `k → ∞` every distribution tends to uniform `1/V`. With a big vocabulary and a small
corpus add-1 over-smooths, which is why the evaluation page sweeps k.

---

## 6. Backoff (fallback for unseen contexts)

If the full context was never seen, `NGramModel.backoff_context` drops the oldest word
until the remaining context has a non-zero count; the empty (unigram) context always
exists. The choice depends only on the history, never on the predicted word, so every
history still gets exactly one normalised distribution. This is a simple
"longest-seen-context" fallback, not Katz backoff: no probability mass is redistributed
between orders.

---

## 7. Temperature (sampling extension, not part of MLE)

```
p'(w) = p(w)^(1/T) / Σ_v p(v)^(1/T)          (generation.apply_temperature)
```

T = 1 leaves the model unchanged, T < 1 sharpens (T → 0 is greedy), T > 1 flattens. It is
computed in log space so very small T does not overflow. Example: p = (0.6, 0.4), T = 0.5
→ (0.36, 0.16)/0.52 = (0.692, 0.308).

Sampling is inverse-CDF: draw `u ~ U[0, 1)` from a seeded `random.Random` and return the
first word whose cumulative probability exceeds u. The same seed gives the same text.

---

## 8. Cross-entropy

For held-out sentences with N predicted tokens (each word plus one `</s>` per sentence):

```
H = −(1/N) Σᵢ ln P(wᵢ | contextᵢ)        nats       (evaluation.evaluate)
H_bits = H / ln 2
```

**Worked example**, bigram MLE on `i love ai`:

| context | word | P | −ln P |
|---|---|---:|---:|
| `<s>` | i | 3/3 = 1 | 0 |
| i | love | 3/3 = 1 | 0 |
| love | ai | 1/3 | 1.0986 |
| ai | `</s>` | 1/1 = 1 | 0 |

```
H = 1.0986 / 4 = 0.2747 nats = 0.3962 bits
```

With add-1 (V = 8): P = 4/11, 4/11, 2/11, 2/9, so
`H = −(ln 4/11 + ln 4/11 + ln 2/11 + ln 2/9)/4 = 1.3080 nats`. Smoothing lowers the
probability of the events that were seen, so on training-like text it *raises* H; on
unseen text it is what keeps H finite.

---

## 9. Perplexity

```
PP = exp(H) = 2^(H_bits) = P(w₁ … w_N)^(−1/N)
```

Example above: `PP = exp(0.2747) = 3^(1/4) = 1.316`.

Interpretation: the effective number of equally likely choices per token.

* A uniform model over V outcomes has `H = ln V` and `PP = V`. With V = 8, PP = 8.
* A model that always predicts correctly with certainty has PP = 1.
* If any token has P = 0, PP = ∞.

### Evaluation rules the code enforces

1. The test set is split off **before** counting and is never used for training.
2. The vocabulary comes from the training split; unknown test words are scored as `<unk>`.
   With `min_count = 1`, `<unk>` never occurs in training, so MLE gives it P = 0. That is
   why even the MLE unigram is infinite on held-out text. Raise `min_count` to give
   `<unk>` real mass.
3. The same tokenization and padding are used in training and evaluation
   (`NGramModel.sentence_events` is shared).
4. Perplexities are only comparable with the same vocabulary, tokenization and test set.
5. Training-set metrics are reported separately and never presented as quality.
