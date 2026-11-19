import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import ART, artifact, dataset_choice, label, page

page("Prediction Benchmark", "🎯")
ds = dataset_choice()
base = artifact(ds, "baselines.json")
if not base:
    st.warning("Run `python -m basketmind.experiments.run baselines` first.")
    st.stop()
evals = {p.stem.replace("eval-", ""): artifact(ds, f"lm/{p.name}") for p in sorted((ART / ds / "lm").glob("eval-*.json"))} \
    if (ART / ds / "lm").exists() else {}

subset = st.radio("Cases", ["all", "planted_cases"], horizontal=True,
                  format_func=lambda s: "All test cases" if s == "all" else "Cases that split a planted itemset")
rows = {}
for name, m in base["models"].items():
    rows[label(name)] = m[subset]
for run, e in evals.items():
    if run.startswith("abl-"):
        continue
    rows[label(run)] = e[subset]
metrics = ["mrr", "hits@1", "hits@5", "hits@10", "recall@10", "ndcg@10"]
table = pd.DataFrame([{"model": k, **{m: f"{v[m]['estimate']:.3f} [{v[m]['lo']:.3f}, {v[m]['hi']:.3f}]" for m in metrics},
                       "median rank": v["median_rank"]} for k, v in rows.items()])
st.dataframe(table, hide_index=True, width="stretch")
n = base["n_cases"] if subset == "all" else base["n_planted_cases"]
st.caption(f"{n:,} cases from held-out test baskets: half of each basket visible, the rest are targets. MRR uses the best-ranked "
           "target; Recall@10 is the share of targets in the top 10. Brackets: 95% percentile bootstrap, 2,000 resamples, seed 1.")

metric = st.selectbox("Chart metric", metrics, index=0)
order = sorted(rows, key=lambda k: rows[k][metric]["estimate"])
fig = go.Figure(go.Bar(y=order, x=[rows[k][metric]["estimate"] for k in order], orientation="h",
                       error_x=dict(type="data", symmetric=False,
                                    array=[rows[k][metric]["hi"] - rows[k][metric]["estimate"] for k in order],
                                    arrayminus=[rows[k][metric]["estimate"] - rows[k][metric]["lo"] for k in order])))
fig.update_layout(height=60 + 42 * len(order), margin=dict(t=10), xaxis_title=metric)
st.plotly_chart(fig, width="stretch")

comp = artifact(ds, "comparisons.json")
if comp:
    st.subheader("Paired comparisons (same cases, paired bootstrap)")
    pm = st.selectbox("Metric", ["mrr", "hits@10", "recall@10"])
    rows = [r for r in comp["paired"] if r["metric"] == pm and not r["a"].startswith("llm-abl") and not r["b"].startswith("llm-abl")
            and not r["b"].endswith("fixedprefix")]
    st.dataframe(pd.DataFrame([{"A": label(r["a"]), "B": label(r["b"]), "A − B": round(r["difference"], 4),
                                "95% CI": f"[{r['lo']:+.4f}, {r['hi']:+.4f}]", "P(A > B)": round(r["p_a_better"], 3),
                                "verdict": "A better" if r["lo"] > 0 else "B better" if r["hi"] < 0 else "no clear difference"}
                               for r in rows]), hide_index=True, width="stretch")
    st.caption("Each bootstrap replicate resamples case indices once and applies them to both models, so case difficulty "
               "cancels out. 'No clear difference' means the interval contains 0.")
