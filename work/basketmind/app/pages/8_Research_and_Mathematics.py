import math

import numpy as np
import streamlit as st

from common import dataset, dataset_choice, page

page("Research & Mathematics", "📐")
ds = dataset_choice()
d = dataset(ds)
n = len(d.baskets)
names = [p.name for p in d.catalogue]


@st.cache_data
def counts(name):
    dd = dataset(name)
    c = np.zeros(dd.n_items, dtype=np.int64)
    for b in dd.baskets:
        c[b] += 1
    return c


cnt = counts(ds)
st.subheader("Support, confidence, lift: worked on this shop's data")
c1, c2 = st.columns(2)
a_name = c1.selectbox("A", names, index=next((i for i, x in enumerate(names) if x.startswith("Oven Lane Bread")), 0))
b_name = c2.selectbox("B", names, index=next((i for i, x in enumerate(names) if "Butter" in x), 1))
a, b = names.index(a_name), names.index(b_name)
ab = sum(a in x and b in x for x in d.baskets)
sa, sb, sab = cnt[a] / n, cnt[b] / n, ab / n
conf = sab / sa if sa else float("nan")
lift = conf / sb if sb else float("nan")
st.latex(r"\text{Support}(A)=\frac{\#\{\text{baskets} \ni A\}}{N}=\frac{%d}{%d}=%.4f" % (cnt[a], n, sa))
st.latex(r"\text{Support}(A\cup B)=\frac{%d}{%d}=%.4f" % (ab, n, sab))
st.latex(r"\text{Confidence}(A\to B)=\frac{\text{Support}(A\cup B)}{\text{Support}(A)}=\frac{%.4f}{%.4f}=%.3f" % (sab, sa, conf))
st.latex(r"\text{Lift}(A\to B)=\frac{\text{Confidence}(A\to B)}{\text{Support}(B)}=\frac{%.3f}{%.4f}=%.2f" % (conf, sb, lift))
st.caption("Lift 1 means A tells you nothing about B. Above 1, B is more likely when A is in the basket.")

st.subheader("Apriori and bitsets")
st.markdown(r"""
**Apriori property.** If an itemset is frequent, every subset of it is frequent. So the size-(k+1) candidates
are built only by joining frequent size-k itemsets that share their first k−1 items, and a candidate is dropped
unless *all* its size-k subsets are frequent.

**Vertical bitsets.** Item *i* is stored as a bit string with one bit per basket. The support count of {a, b, c} is
`popcount(bits[a] & bits[b] & bits[c])`. Here that is """ + f"{(n + 63) // 64:,}" + r""" 64-bit words per item: one AND and one
popcount per word, instead of a pass over """ + f"{n:,}" + r""" baskets.""")
small = d.baskets[:8]
ex_items = [x for x in (a, b) if any(x in bb for bb in small)] or [small[0][0], small[0][1]]
st.code("\n".join(f"{names[i][:28]:<28} " + "".join("1" if i in bb else "0" for bb in small) for i in ex_items) +
        "\n" + f"{'AND':<28} " + "".join("1" if all(i in bb for i in ex_items) else "0" for bb in small) +
        f"   → count {sum(all(i in bb for i in ex_items) for bb in small)} in the first 8 baskets")

st.subheader("Causal language-modelling loss")
st.latex(r"\mathcal{L} = -\frac{1}{\sum_t m_{t+1}} \sum_{t} m_{t+1}\,\log p_\theta(x_{t+1}\mid x_{\le t})")
st.markdown("Position *t* sees only tokens up to *t* (the causal mask). The target at *t* is the token at *t+1* "
            "(labels shifted by one), and $m_{t+1}=0$ for padding, so padded positions add nothing. For a 7-item basket "
            "the model is graded on 8 predictions: each product, then `</basket>`. A uniform guess over the 1,001 possible "
            f"next tokens costs ln 1001 = {math.log(1001):.2f} nats per token.")

st.subheader("Ranking metrics, worked")
ranks = [1, 4, 12, 2]
st.markdown(f"Four cases whose best-ranked correct item came at ranks {ranks}:")
st.latex(r"\text{MRR}=\tfrac14\left(\tfrac11+\tfrac14+\tfrac1{12}+\tfrac12\right)=%.3f" % np.mean([1 / r for r in ranks]))
st.latex(r"\text{Hits@10}=\tfrac14(1+1+0+1)=0.75\qquad \text{Recall@10}=\frac{|\text{targets}\cap\text{top-10}|}{|\text{targets}|}\ \text{(per case)}")

st.subheader("Bootstrap confidence intervals")
st.markdown(r"Resample the *n* cases with replacement *B* = 2,000 times and recompute the mean each time. The 2.5th "
            r"and 97.5th percentiles of those means are a 95% interval. For two models on the **same** cases, resample "
            r"case indices once per replicate and take the difference (**paired** bootstrap). Shared case difficulty cancels, "
            r"so real differences show up with far fewer cases.")
rng = np.random.default_rng(0)
x = rng.binomial(1, 0.3, 200).astype(float)
reps = x[rng.integers(0, 200, (2000, 200))].mean(1)
st.code(f"200 cases, Hits@10 = {x.mean():.3f};  bootstrap 95% CI [{np.percentile(reps, 2.5):.3f}, {np.percentile(reps, 97.5):.3f}]"
        f";  normal approximation ±{1.96 * x.std(ddof=1) / math.sqrt(200):.3f}")
