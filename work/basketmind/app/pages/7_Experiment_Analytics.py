import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import ART, artifact, dataset_choice, label, page

page("Experiment Analytics", "📈")
ds = dataset_choice()

st.subheader("1 · Ordering ablation: fixed vs frozen vs fresh")
st.markdown("Three runs, identical except for how the basket order is presented: the same 4,000 training baskets, 2 "
            "epochs, 250 steps of 32, the same seed. **fixed** = sorted by product id every time; **frozen** = one random "
            "order per basket, reused; **fresh** = a new random order at every presentation.")
abl = {r: json.loads((ART / ds / "runs" / f"abl-{r}" / "history.json").read_text())
       for r in ("fixed", "frozen", "fresh") if (ART / ds / "runs" / f"abl-{r}" / "history.json").exists()}
if abl:
    a, b = st.columns(2)
    fig = go.Figure()
    for r, h in abl.items():
        h = pd.DataFrame(h)
        fig.add_trace(go.Scatter(x=h.step, y=h.val_loss_random, name=f"{r}: random-order val", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=h.step, y=h.val_loss_fixed, name=f"{r}: sorted-order val", mode="lines", line=dict(dash="dot")))
    fig.update_layout(height=380, xaxis_title="step", yaxis_title="validation loss (nats)", margin=dict(t=10))
    a.plotly_chart(fig, width="stretch")
    rows = []
    for r in abl:
        e = artifact(ds, f"lm/eval-abl-{r}.json")
        if e:
            rows.append({"run": r, "MRR, random-order prefixes": e["all"]["mrr"]["estimate"],
                         "MRR, sorted prefixes": e["fixed_order_prefix"]["mrr"]["estimate"],
                         "val loss random": e["val_loss_random_order"], "val loss sorted": e["val_loss_fixed_order"]})
    if rows:
        df = pd.DataFrame(rows)
        f2 = go.Figure([go.Bar(name=c, x=df.run, y=df[c]) for c in ("MRR, random-order prefixes", "MRR, sorted prefixes")])
        f2.update_layout(barmode="group", height=380, yaxis_title="next-item MRR", margin=dict(t=10))
        b.plotly_chart(f2, width="stretch")
        st.dataframe(df.round(4), hide_index=True, width="stretch")
    st.caption("Sorted order plants a fake signal: the next id is always larger than the last. The fixed-order model learns "
               "it, so its loss on *sorted* baskets falls while its loss on *shuffled* baskets rises, and its "
               "recommendations depend on how the basket is written down. Frozen and fresh orders carry no such signal.")

st.subheader("2 · Classical recommenders vs language models")
base = artifact(ds, "baselines.json")
evals = {p.stem.replace("eval-", ""): artifact(ds, f"lm/{p.name}") for p in sorted((ART / ds / "lm").glob("eval-*.json"))} \
    if (ART / ds / "lm").exists() else {}
if base:
    rows = [{"model": label(k), "group": "classical", "all": v["all"]["mrr"], "planted": v["planted_cases"]["mrr"]}
            for k, v in base["models"].items()]
    rows += [{"model": label(k), "group": "language model", "all": v["all"]["mrr"], "planted": v["planted_cases"]["mrr"]}
             for k, v in evals.items()]
    fig = go.Figure()
    for col, nm in (("all", "all test cases"), ("planted", "cases splitting a planted itemset")):
        fig.add_trace(go.Bar(name=nm, y=[r["model"] for r in rows], x=[r[col]["estimate"] for r in rows], orientation="h",
                             error_x=dict(type="data", symmetric=False, array=[r[col]["hi"] - r[col]["estimate"] for r in rows],
                                          arrayminus=[r[col]["estimate"] - r[col]["lo"] for r in rows])))
    fig.update_layout(barmode="group", height=80 + 46 * len(rows), xaxis_title="MRR (95% bootstrap CI)", margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")

st.subheader("3 · Generated baskets vs the ground truth")
mining = artifact(ds, "mining.json")
probes = {p.stem.replace("probe-", ""): artifact(ds, f"lm/{p.name}") for p in sorted((ART / ds / "lm").glob("probe-*.json"))} \
    if (ART / ds / "lm").exists() else {}
if probes and mining:
    supports = list(next(iter(probes.values()))["planted_by_support"].keys())
    fig = go.Figure()
    fig.add_trace(go.Bar(name="exact miner, 10k real baskets (ceiling)", x=supports,
                         y=[mining["ceiling"]["10000"]["by_support"][s] for s in supports]))
    for run, p in probes.items():
        fig.add_trace(go.Bar(name=f"{label(run)} ({p['nonempty']:,} generated)", x=supports,
                             y=[p["planted_by_support"][s]["recovered"] / p["planted_by_support"][s]["of"] for s in supports]))
    fig.update_layout(barmode="group", height=360, yaxis_tickformat=".0%", xaxis_title="planted support",
                      yaxis_title="planted itemsets re-created", margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")
    for run, p in probes.items():
        st.write(f"**{label(run)}**: frequent itemsets in its baskets by origin: {p['labels_of_generated_itemsets']}; "
                 f"same-category share among frequent pairs {p['pair_same_category_share']['generated']:.0%} "
                 f"(real {p['pair_same_category_share']['real']:.0%}).")
