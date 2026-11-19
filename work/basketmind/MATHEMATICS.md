# Mathematics

Notation: *N* baskets; a basket is a set of product ids; for an itemset *X*, `count(X)` is the
number of baskets that contain every item of *X*.

## Support, confidence, lift

```
Support(X)         = count(X) / N
Confidence(A → B)  = Support(A ∪ B) / Support(A)          = P(B | A)
Lift(A → B)        = Confidence(A → B) / Support(B)        = P(A, B) / (P(A) P(B))
Leverage(A → B)    = Support(A ∪ B) − Support(A) Support(B)
Conviction(A → B)  = (1 − Support(B)) / (1 − Confidence(A → B))
```

**Worked example** (10 baskets; A is in 5, B in 4, both in 3; this is the unit test):
Support(A) = 0.5, Support(B) = 0.4, Support(A ∪ B) = 0.3, so Confidence(A → B) = 0.3 / 0.5 = 0.6,
Lift = 0.6 / 0.4 = 1.5, Leverage = 0.3 − 0.5 · 0.4 = 0.10, Conviction = 0.6 / 0.4 = 1.5. In the
other direction, Confidence(B → A) = 0.3 / 0.4 = 0.75. Lift is symmetric; confidence is not.

Lift = 1 means independence. A rule can have high confidence and lift ≈ 1 when B is simply
popular: "people who buy anything also buy bananas". That is why the Cross-Sell Studio shows
lift next to confidence.

## The Apriori property

If X ⊆ Y then count(X) ≥ count(Y): every basket holding Y also holds X. So an itemset can only be
frequent if all of its subsets are frequent. Apriori uses this level by level:

1. L₁ = frequent single items.
2. Candidates of size k+1 are joins of two frequent k-itemsets that agree on their first k−1
   (sorted) items.
3. A candidate is **pruned** without counting unless every one of its k-subsets is in L_k.
4. The survivors are counted; those with count ≥ ⌈min_support · N⌉ form L_{k+1}.

On the 100,000-basket dataset at 0.1% support, 928 frequent items would allow about 133 million
triples, but only 17,474 size-3 candidates survive the join and prune
(`artifacts/full/mining.json`).

## Vertical bitsets

Item *i* is a bit vector `bits[i]` of length N, packed into ⌈N/64⌉ unsigned 64-bit words, with bit
*b* set when basket *b* holds *i*. Then

```
count({a, b, c}) = popcount( bits[a] AND bits[b] AND bits[c] )
```

For N = 100,000 that is 1,563 words per item. A candidate costs about 1,563 ANDs and
popcounts (`np.bitwise_count`), and all candidates sharing a prefix are counted in one vectorised
operation. The tests compare these counts with brute-force enumeration of every sub-itemset of
every basket.

## Causal language-modelling loss

A basket becomes the token sequence x₀ … x_T = `<basket>`, p₁, …, p_k, `</basket>`, right-padded
to the batch width. With padding mask m_t (1 for real tokens):

```
L(θ) = − ( Σ_{t=0}^{T−1} m_{t+1} · log p_θ(x_{t+1} | x_0 … x_t) ) / Σ_t m_{t+1}
```

* **Causal mask:** position *t* attends only to positions ≤ *t*. A test checks that changing
  tokens 3–4 leaves the logits at positions 0–2 identical.
* **Label shift:** the prediction at position *t* is scored against the token at *t+1*.
* **Padding:** targets with m_{t+1} = 0 contribute nothing, and `attention_mask` stops real tokens
  attending to padding. A test checks that a padded row gives the same logits as the
  unpadded sequence.
* `<basket>` and `<pad>` can never be the next token, so their logits are set to −∞ before
  the softmax. The softmax covers the 1,000 products plus `</basket>`; a uniform guess costs
  ln 1001 = 6.91 nats.

## LoRA

A frozen weight W₀ (d_in × d_out) is adapted as `W₀ + (α/r) Aᵀ Bᵀ` with A ∈ ℝ^{r×d_in} (random)
and B ∈ ℝ^{d_out×r} (zero), so training starts exactly at the pretrained model. Here r = 16 and
α = 32. Each adapted `c_attn` (768 → 2304) adds 16 · (768 + 2304) = 49,152 parameters, and each
`c_fc` (768 → 3072) adds 61,440; × 12 blocks = 1.33M. With the 1,003 × 768 embedding (0.77M)
and the final LayerNorm, about 2.1M of 88M parameters are trained (2.4%).

## Next-item ranking metrics

For a case with visible set V and target set T, every product not in V is a candidate. The rank of
a target *t* is the number of candidates whose score is ≥ score(t), with *t* itself included, so
ties count against the target.

```
rank*      = min_{t ∈ T} rank(t)                       best-ranked target
MRR        = mean over cases of 1 / rank*
Hits@K     = mean over cases of 1[rank* ≤ K]
Recall@K   = mean over cases of |{t ∈ T : rank(t) ≤ K}| / |T|
NDCG@10    = Σ_{t: rank(t) ≤ 10} 1/log₂(rank(t)+1)  /  Σ_{i=1}^{min(10,|T|)} 1/log₂(i+1)
```

**Worked example** (also a unit test): scores for six items [0.9, 0.8, 0.7, 0.6, 0.5, 0.1], V =
{0}, T = {3, 5}. The candidates in order are 1, 2, 3, 4, 5, so rank(3) = 3 and rank(5) = 5.
rank* = 3, so RR = 1/3, Hits@1 = 0, Hits@3 = 1, Recall@3 = 1/2 and Recall@5 = 1.
NDCG@10 = (1/log₂4 + 1/log₂6) / (1/log₂2 + 1/log₂3) = 0.887 / 1.631 = 0.544.

The language model scores a visible set as the mean of log P(next = product | V written in order
σ) over three random orders σ, so no single arbitrary order decides the answer.

## Bootstrap confidence intervals

Given per-case values v₁ … v_n of a metric:

* **Percentile bootstrap:** draw B = 2,000 samples of n case indices with replacement, take the
  mean of each, and report the 2.5th and 97.5th percentiles. For a mean, this agrees with the
  normal interval ±1.96 · s/√n when n is large (tested).
* **Paired bootstrap** for models A and B on the same cases: per replicate, draw one set of
  indices *I* and compute mean(a_I) − mean(b_I). Report the interval for the difference and
  P(A > B), the share of replicates where it is positive. Because cases vary in difficulty much
  more than models differ, pairing removes most of the variance. The tests construct a 0.1 gap
  that two separate intervals cannot resolve and the paired interval can.
* **Recovery rates** (planted itemsets re-created by the model): the *generated baskets* are
  resampled and re-mined, B = 100.

All seeds are fixed and recorded next to each interval.
