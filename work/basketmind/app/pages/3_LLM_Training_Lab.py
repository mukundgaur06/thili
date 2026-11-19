import json
import platform

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import ART, artifact, dataset, dataset_choice, label, page

page("LLM Training Laboratory", "🧠")
ds = dataset_choice()
d = dataset(ds)
runs = sorted(p.name for p in (ART / ds / "runs").glob("*") if (p / "history.json").exists()) if (ART / ds / "runs").exists() else []

st.markdown("""
**Model:** OpenAI GPT-2 (124M), the pretrained weights from the Hugging Face hub, loaded from the local cache.

| | Off-the-shelf | Continued pretraining |
|---|---|---|
| Transformer blocks | pretrained, unchanged | pretrained, **frozen**, plus LoRA adapters (r = 16) on `c_attn` and `c_fc` |
| Vocabulary | 1,003 domain tokens: one per product, plus `<basket>`, `</basket>`, `<pad>` | same |
| Product embeddings | mean of GPT-2's embeddings of the product's *name* | trained |
| Objective | – | next-token cross-entropy on verbalized baskets (padding masked, labels shifted) |
""")


@st.cache_data(show_spinner="Reading the GPT-2 tokenizer…")
def vocab_stats(name):
    from basketmind.language_model.tokenizer import vocabulary_stats
    return vocabulary_stats(dataset(name).catalogue)


@st.cache_data
def seq_stats(name):
    from basketmind.data.verbalizer import length_stats
    return length_stats(dataset(name).baskets)


v, s = vocab_stats(ds), seq_stats(ds)
c = st.columns(4)
c[0].metric("GPT-2 pieces per product name", f"{v['gpt2_pieces_per_name_mean']:.1f}", f"max {v['gpt2_pieces_per_name_max']}")
c[1].metric("Domain tokens per product", "1", "ids preserved exactly")
c[2].metric("Domain vocabulary", f"{v['domain_vocab_size']:,}", f"GPT-2: {v['gpt2_vocab_size']:,}")
c[3].metric("Tokens per basket", f"{s['mean_tokens']:.1f}", f"95th pct {s['p95_tokens']:.0f}, max {s['max_tokens']}")

if not runs:
    st.info("No training runs yet: `python -m basketmind.experiments.run lm-train --dataset full --run cpt-fresh`.")
    st.stop()
run = st.selectbox("Training run", runs, index=runs.index("cpt-fresh") if "cpt-fresh" in runs else 0)
hist = pd.DataFrame(json.loads((ART / ds / "runs" / run / "history.json").read_text()))
meta = artifact(ds, f"runs/{run}/run.json") or {}
conf = meta.get("config", {})
fig = go.Figure()
fig.add_trace(go.Scatter(x=hist.step[1:], y=hist.train_loss[1:], name="train loss", mode="lines+markers"))
fig.add_trace(go.Scatter(x=hist.step, y=hist.val_loss_random, name="validation, random order", mode="lines+markers"))
fig.add_trace(go.Scatter(x=hist.step, y=hist.val_loss_fixed, name="validation, sorted order", mode="lines+markers", line=dict(dash="dot")))
fig.update_layout(height=380, xaxis_title="step", yaxis_title="next-token cross-entropy (nats)", margin=dict(t=10))
st.plotly_chart(fig, width="stretch")
st.caption("A model that learned co-occurrence predicts held-out baskets equally well in any order. A gap between "
           "the two validation curves means it learned the *presentation order*. Compare the runs on Experiment Analytics.")
a, b = st.columns(2)
with a:
    st.subheader("Configuration")
    st.json({**conf, "train baskets": meta.get("n_train_baskets"), "status": "finished" if meta.get("done") else f"paused at step {meta.get('step')}"})
with b:
    st.subheader("Checkpoint and hardware")
    p = meta.get("params", {})
    st.json({"trainable parameters": p.get("trainable"), "total parameters": p.get("total"),
             "trainable %": round(p.get("percent", 0), 2), "checkpoint": f"artifacts/{ds}/runs/{run}/adapter.pt",
             "wall-clock seconds": float(hist.seconds.iloc[-1]), "baskets presented": int(hist.presented.iloc[-1]),
             "epochs": round(float(hist.epoch.iloc[-1]), 2), "trained on": meta.get("machine"), "GPU": "none (CPU only)",
             "this viewer": platform.processor() or platform.machine()})
