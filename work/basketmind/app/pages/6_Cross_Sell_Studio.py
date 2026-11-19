import numpy as np
import pandas as pd
import streamlit as st

from common import classical, dataset, dataset_choice, label, lm, lm_runs, page, shares

page("Cross-Sell Studio", "🛍️")
ds = dataset_choice()
d = dataset(ds)
names = [p.name for p in d.catalogue]
default = [n for n in names if n.startswith("Oven Lane Bread")][:1] or names[:1]
chosen = st.multiselect("Your basket", names, default=default)
k = st.slider("Suggestions per recommender", 3, 10, 5)
use_lm = st.multiselect("Language models to ask", lm_runs(ds), default=[r for r in lm_runs(ds) if r == "cpt-fresh"],
                        help="Each model is a copy of GPT-2 in memory (about 0.5 GB); add the off-the-shelf one to compare.",
                        format_func=label)
if not chosen:
    st.stop()

from basketmind.recommenders.crosssell import cross_sell, reason, upsell  # noqa: E402

vis = [names.index(x) for x in chosen]
recs = classical(ds)
co = recs["Co-occurrence (confidence)"]
pop = recs["Popularity"]
cols = st.columns(len(recs) + len(use_lm))
results = {}
for col, (name, rec) in zip(cols, recs.items()):
    results[name] = (rec.score(vis), rec)
for run in use_lm:
    from basketmind.language_model.inference import next_item_logprobs
    model = lm(ds, run)
    results[label(run)] = (next_item_logprobs(model, [vis], n_orders=5)[0], None)

for col, (name, (scores, rec)) in zip(cols, results.items()):
    with col:
        st.markdown(f"**{name}**")
        top = cross_sell(scores, vis, d.catalogue, k)
        for rank, c in enumerate(top, 1):
            ev = co.evidence(vis, c)
            score_txt = (f"P = {np.exp(scores[c]):.2%}" if rec is None else f"score {scores[c]:.3f}")
            with st.container(border=True):
                st.markdown(f"{rank}. **{names[c]}**  \n<small>{d.catalogue[c].category} · {score_txt}</small>",
                            unsafe_allow_html=True)
                st.caption(f"support {ev['pair_support']:.3%} · confidence {ev['confidence']:.1%} · lift {ev['lift']:.1f}")

st.divider()
st.subheader("Why? Evidence for every suggestion")
rows = []
for name, (scores, rec) in results.items():
    for c in cross_sell(scores, vis, d.catalogue, k):
        ev = co.evidence(vis, c)
        row = {"recommender": name, "product": names[c], "category": d.catalogue[c].category,
               "pair support": round(ev["pair_support"], 5), "confidence": round(ev["confidence"], 3), "lift": round(ev["lift"], 2),
               "reason": reason(ev, names)}
        if rec is None:
            row["LLM probability"] = round(float(np.exp(scores[c])), 4)
        if name == "Association rules":
            r = rec.explain(vis, c)
            row["rule"] = (" + ".join(d.names(r["antecedent"])) + f" → conf {r['confidence']:.2f}, lift {r['lift']:.1f}") if r else "no rule fired (popularity fallback)"
        rows.append(row)
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.caption("Support, confidence and lift come from exact counts over the training baskets, for the visible product that "
           "supports the suggestion most. A high-confidence suggestion with lift near 1 is just a best seller: it would be "
           "bought anyway, so it is a weak cross-sell.")

st.subheader("Upsell: a pricier version of what is already in the basket")
up = upsell(vis, d.catalogue, shares(ds))
if up:
    st.dataframe(pd.DataFrame([{"instead of": names[u["instead_of"]], "suggest": names[u["suggest"]],
                                "price": f"{u['price_from']:.2f} → {u['price_to']:.2f}", "uplift": f"+{u['uplift_pct']:.0f}%",
                                "reason": u["reason"]} for u in up]), hide_index=True, width="stretch")
else:
    st.write("Nothing in the basket has a pricier variant.")
st.caption("Kept separate from cross-sell. In this synthetic shop the choice of brand and variant ignores the rest of the "
           "basket, so there is no upsell signal for any model to learn. This is a transparent business rule, not a "
           "prediction, and it is not scored.")
