"""BasketMind dashboard: Executive Overview.

    streamlit run app/app.py
"""

import streamlit as st

from common import artifact, ci_text, dataset, dataset_choice, label, lm_runs, page, truth

page("Executive Overview", "🛒")
ds = dataset_choice()
d = dataset(ds)
planted, planted_rows, truth_sets, labels, cfg = truth(ds)
mining = artifact(ds, "mining.json")
base = artifact(ds, "baselines.json")
runs = [r for r in lm_runs(ds) if r != "offshelf"]
model = st.sidebar.selectbox("Language model", ["cpt-fresh"] if "cpt-fresh" in runs else (runs or ["offshelf"]))
ev = artifact(ds, f"lm/eval-{model}.json")
probe = artifact(ds, f"lm/probe-{model}.json")

st.markdown("**Research question:** can a small pretrained language model learn which products sell together "
            "from transaction text alone, and is it a useful cross-sell engine compared with exact counting? "
            "Every number below was measured by the experiment pipeline; nothing is illustrative.")
c = st.columns(6)
c[0].metric("Products", f"{d.n_items:,}")
c[1].metric("Baskets", f"{len(d.baskets):,}", f"{d.meta['mean_basket_size']:.1f} items on average")
c[2].metric("Planted itemsets", len(planted), "sizes 2–5, 4 support levels")
c[3].metric("Frequent itemsets (exact)", f"{cfg['n_truth']:,}", f"min support {cfg['min_support']:.1%}")
if probe:
    c[4].metric("Planted itemsets the LM re-creates", f"{sum(r['recovered'] for r in probe['planted_detail'])}/{len(planted)}",
                f"from {probe['nonempty']:,} generated baskets")
c[5].metric("Selected model", label(model))

if base and ev:
    st.subheader("Next-item prediction on held-out baskets")
    m = st.columns(4)
    rules = base["models"]["association-rules"]["all"]
    pop = base["models"]["popularity"]["all"]
    m[0].metric("Popularity MRR", ci_text(pop["mrr"]))
    m[1].metric("Association rules MRR", ci_text(rules["mrr"]))
    off = artifact(ds, "lm/eval-offshelf.json")
    if off:
        m[2].metric("GPT-2 off-the-shelf MRR", ci_text(off["all"]["mrr"]))
    m[3].metric(f"{label(model)} MRR", ci_text(ev["all"]["mrr"]))
    st.caption(f"{ev['n_cases']:,} test baskets, half of each basket shown, the rest to be predicted; 95% bootstrap "
               "intervals in brackets. The Prediction Benchmark page has every metric and paired comparisons.")

st.subheader("Experiment pipeline")
st.graphviz_chart("""
digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef3f8", color="#9aaabb", fontname="Helvetica", fontsize=11];
  gen [label="Synthetic shop\\n1,000 products\\n100,000 baskets", fillcolor="#fff4e5"];
  gt [label="Hidden ground truth\\n32 planted itemsets", fillcolor="#fde9e4"];
  apr [label="Exact Apriori\\n(bitsets)\\n= answer key"];
  cls [label="Popularity · co-occurrence\\n· association rules"];
  verb [label="Verbalize baskets\\n<basket> <p0042> … </basket>\\nshuffled order"];
  lm0 [label="GPT-2 off-the-shelf"]; cpt [label="Continued pretraining\\n(LoRA, next-token loss)"];
  eval [label="Next-item ranking\\nMRR · Hits@K · Recall@K\\n+ bootstrap CIs", fillcolor="#e8f4ea"];
  probe [label="Probe: generate baskets\\n→ mine → compare", fillcolor="#e8f4ea"];
  gen -> gt [style=dashed]; gen -> apr; gen -> cls; gen -> verb; verb -> lm0 -> cpt;
  cls -> eval; lm0 -> eval; cpt -> eval; cpt -> probe; apr -> probe; gt -> probe [style=dashed]; gt -> apr [style=dashed, label=" label "];
}""")
st.caption("Dashed: the ground truth is used only to *score*. No model ever reads it.")
