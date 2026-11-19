import time

import pandas as pd
import plotly.express as px
import streamlit as st

from common import artifact, dataset, dataset_choice, page, truth

page("Association Mining", "⛏️")
ds = dataset_choice()
d = dataset(ds)
planted, planted_rows, truth_sets, labels, cfg = truth(ds)
mining = artifact(ds, "mining.json")

st.markdown("The **exact** answer: Apriori on vertical bitsets counts every itemset's support exactly. "
            "It is the experiment's answer key, and the baseline every model is measured against.")
if mining:
    c = st.columns(4)
    c[0].metric("Frequent itemsets", f"{mining['n_frequent']:,}", f"min support {mining['min_support']:.1%}")
    c[1].metric("Mining time", f"{mining['seconds']:.1f} s", f"{mining['n_baskets']:,} baskets")
    c[2].metric("Planted itemsets found", f"{sum(r['recovered'] for r in mining['planted_on_full_data'])}/{len(planted)}")
    bf = mining["bruteforce_check"]
    c[3].metric("Matches brute force", "yes" if bf["identical"] else "NO", f"on {bf['baskets']:,} baskets, size ≤ {bf['max_size']}")
    st.dataframe(pd.DataFrame(mining["levels"]).rename(columns={"size": "itemset size", "candidates": "candidates counted",
                                                               "frequent": "frequent"}), hide_index=True)
    st.caption("Candidates counted at size k+1 are only joins of frequent size-k itemsets whose every subset is frequent "
               "(the Apriori property). That is why size 3 counts ~17k candidates instead of all ~133 million possible triples.")

st.subheader("Mine it yourself")
c1, c2, c3 = st.columns(3)
ms = c1.select_slider("Minimum support", [0.001, 0.002, 0.005, 0.01, 0.02, 0.05], 0.005)
size = c2.multiselect("Itemset sizes", [1, 2, 3, 4, 5], [2, 3, 4, 5])
show_labels = c3.toggle("Show answer-key labels", True, help="Why each itemset is frequent: planted, part of a planted "
                        "set, an extension of one, or chance. Uses the hidden ground truth; models never see it.")


@st.cache_data(show_spinner="Running Apriori…")
def mine(name, ms):
    from basketmind.mining.apriori import apriori
    dd = dataset(name)
    t = time.time()
    c = apriori(dd.baskets, dd.n_items, ms, 5)
    return c, time.time() - t


counts, secs = mine(ds, ms)
n = len(d.baskets)
rows = [{"itemset": " + ".join(d.names(sorted(k))), "size": len(k), "count": v, "support": v / n,
         "label": labels.get(k, "–")} for k, v in counts.items() if len(k) in size]
df = pd.DataFrame(rows).sort_values("count", ascending=False)
st.caption(f"{len(counts):,} frequent itemsets at {ms:.1%} in {secs:.2f} s (exact counts).")
if show_labels and len(df):
    st.plotly_chart(px.histogram(df, x="size", color="label", barmode="stack", title="Frequent itemsets by size and origin")
                    .update_layout(height=300), width="stretch")
st.dataframe(df if show_labels else df.drop(columns="label"), hide_index=True, width="stretch", height=360)
st.info("**A planted itemset is not the same as a frequent itemset.** Planting {Bread, Butter} at 3% makes that pair "
        "frequent, but popular products also meet by chance, and the planted baskets carry other items too. "
        "'part', 'extension' and 'chance' are all frequent; only 'planted' ones were put there on purpose.")

st.subheader("Association-rule explorer")
from basketmind.mining.rules import generate_rules  # noqa: E402
c1, c2, c3 = st.columns(3)
mc = c1.slider("Minimum confidence", 0.0, 1.0, 0.3, 0.05)
ml = c2.slider("Minimum lift", 1.0, 50.0, 2.0, 0.5)
item = c3.selectbox("Antecedent contains", ["(any)"] + sorted({p.name for p in d.catalogue}))
rules = generate_rules({k: v / n for k, v in counts.items()}, mc, ml)
if item != "(any)":
    pid = [p.pid for p in d.catalogue if p.name == item][0]
    rules = [r for r in rules if pid in r.antecedent]
st.dataframe(pd.DataFrame([{"if basket has": " + ".join(d.names(sorted(r.antecedent))),
                            "then also": " + ".join(d.names(sorted(r.consequent))),
                            "support": round(r.support, 5), "confidence": round(r.confidence, 3), "lift": round(r.lift, 2),
                            "leverage": round(r.leverage, 5), "conviction": round(r.conviction, 2)}
                           for r in rules[:500]]), hide_index=True, width="stretch", height=360)
st.caption(f"{len(rules):,} rules with one item on the right-hand side. Sorted by lift.")
