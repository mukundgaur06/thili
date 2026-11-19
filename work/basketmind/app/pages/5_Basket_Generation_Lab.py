import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import artifact, dataset, dataset_choice, label, lm, lm_runs, page, truth

page("Basket Generation Lab", "🧺")
ds = dataset_choice()
d = dataset(ds)
planted, planted_rows, truth_sets, labels, cfg = truth(ds)
runs = lm_runs(ds)
run = st.sidebar.selectbox("Model", runs, index=runs.index("cpt-fresh") if "cpt-fresh" in runs else 0, format_func=label)

probe = artifact(ds, f"lm/probe-{run}.json")
mining = artifact(ds, "mining.json")
if probe:
    st.subheader(f"Measured probe: {probe['nonempty']:,} baskets written by {label(run)}")
    c = st.columns(5)
    pr = probe["planted_recall"]
    c[0].metric("Planted itemsets re-created", f"{sum(r['recovered'] for r in probe['planted_detail'])}/{len(planted)}",
                f"95% CI {pr['bootstrap']['lo']:.0%}–{pr['bootstrap']['hi']:.0%}")
    if mining:
        ceil = mining["ceiling"].get(str(min(10000, probe["nonempty"])) if str(min(10000, probe["nonempty"])) in mining["ceiling"] else "10000")
        c[1].metric("Ceiling: exact miner on real sample", f"{ceil['planted_recall']['estimate']:.0%}", "same sample size")
    c[2].metric("Precision of its frequent itemsets", f"{probe['precision_vs_truth']:.0%}", "really frequent in the shop")
    c[3].metric("Mean basket size", f"{probe['mean_basket_size']['generated']:.2f}", f"real {probe['mean_basket_size']['real']:.2f}")
    c[4].metric("Item-frequency correlation", f"{probe['item_frequency_correlation']:.3f}", "generated vs real")
    bs = probe["planted_by_support"]
    rows = []
    for s, v in bs.items():
        row = {"target support": float(s), "re-created by the LM": v["recovered"] / v["of"]}
        if mining:
            row["exact miner on 10k real baskets"] = mining["ceiling"]["10000"]["by_support"][s]
        rows.append(row)
    df = pd.DataFrame(rows)
    fig = go.Figure([go.Bar(name=col, x=[f"{x:.2%}" for x in df["target support"]], y=df[col]) for col in df.columns[1:]])
    fig.update_layout(barmode="group", height=320, yaxis_tickformat=".0%", xaxis_title="planted support", margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(pd.DataFrame([{"planted itemset": " + ".join(d.names(r["items"])), "size": r["size"], "kind": r["kind"],
                                "target support": r["target_support"], "support in generated baskets": round(r["found_support"], 5),
                                "re-created": "yes" if r["recovered"] else "no"} for r in probe["planted_detail"]]),
                 hide_index=True, width="stretch", height=300)
    g = probe["generation_stats"]
    st.caption(f"Sampling at temperature {g['temperature']}: {g['share_repeated']:.1%} of written items were repeats "
               f"(dropped), {g['share_cut_off']:.1%} of baskets hit the length cap, {g['invalid_tokens']} invalid tokens. "
               "An itemset counts as re-created if its support among the generated baskets is at least the mining threshold "
               f"({probe['min_support']:.1%}).")

st.divider()
st.subheader("Generate baskets now")
c1, c2, c3 = st.columns(3)
n = c1.select_slider("How many", [5, 20, 100, 500], 20)
temp = c2.slider("Temperature", 0.5, 1.5, 1.0, 0.1)
seed = c3.number_input("Seed", 0, 9999, 0)
if st.button("Generate", type="primary"):
    from basketmind.language_model.generation import sample_baskets
    from basketmind.mining.apriori import apriori
    model = lm(ds, run)
    gen, stats = sample_baskets(model, n, temperature=temp, seed=int(seed), batch=min(n, 250))
    st.dataframe(pd.DataFrame([{"#": i, "size": len(b), "products": " ; ".join(d.names(b)),
                                "ids": " ".join(f"<p{x:04d}>" for x in b),
                                "categories": ", ".join(sorted({d.catalogue[x].category for x in b}))}
                               for i, b in enumerate(gen)]), hide_index=True, width="stretch", height=360)
    st.json(stats)
    hits = [p for p in planted if any(set(p.items) <= set(b) for b in gen)]
    st.write(f"Baskets containing a complete planted itemset: "
             f"{sum(any(set(p.items) <= set(b) for p in planted) for b in gen)} of {len(gen)}; "
             f"distinct planted itemsets seen: {len(hits)}.")
    for p in hits[:10]:
        st.write("• " + " + ".join(d.names(p.items)) + f"  (planted at {p.target_support:.2%})")
