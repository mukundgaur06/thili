"""QuantLab dashboard: Overview page.

    streamlit run app/app.py
"""

import plotly.graph_objects as go
import streamlit as st

from common import COLORS, LABELS, need_results, page, variants

page("Overview", "🧮")
r = need_results()
rows = variants(r)
by = {v["name"]: v for v in rows}
base = by["fp32"]
int8 = [v for v in rows if v["name"].endswith("int8")]
best = max(int8, key=lambda v: v["accuracy"])

st.markdown(
    "A 207k-parameter CNN is trained on **Fashion-MNIST**, then compressed four ways. Every number "
    "on these pages was **measured on this machine** by `python -m quantlab.pipeline`: the same 10,000 "
    "test images, the same inputs, the same thread count.")

c = st.columns(5)
c[0].metric("FP32 accuracy", f"{base['accuracy']:.2%}")
c[1].metric(f"Best INT8 ({LABELS[best['name']]})", f"{best['accuracy']:.2%}",
            f"{-best['accuracy_drop_pts']:+.2f} pts vs FP32")
c[2].metric("Size: FP32 → INT8", f"{base['size_bytes'] / 1024:.0f} → {best['size_bytes'] / 1024:.0f} KB",
            f"{best['compression']:.2f}× smaller")
c[3].metric("Speed-up at batch 64", f"{best['speedup_bs64']:.2f}×", f"{best['latency']['bs64']['median_ms']:.2f} ms")
c[4].metric("Quantized engine", r["environment"]["quantized_engine"], f"torch {r['environment']['torch']}")

left, right = st.columns([1.1, 1])
with left:
    st.subheader("Where the quantization boundaries sit")
    st.graphviz_chart("""
digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef2f7", color="#9aa8ba", fontname="Helvetica", fontsize=11];
  in [label="image\\nfp32 1×28×28", fillcolor="#ffffff"];
  q [label="QuantStub\\nfp32 → uint8", fillcolor="#fde9d9"];
  c1 [label="Conv3×3 16\\n+BN+ReLU (fused)"]; p1 [label="MaxPool"];
  c2 [label="Conv3×3 32\\n+BN+ReLU (fused)"]; p2 [label="MaxPool"];
  f1 [label="Linear 1568→128\\n+ReLU (fused)"]; f2 [label="Linear 128→10"];
  dq [label="DeQuantStub\\nuint8 → fp32", fillcolor="#fde9d9"];
  out [label="logits", fillcolor="#ffffff"];
  in -> q -> c1 -> p1 -> c2 -> p2 -> f1 -> f2 -> dq -> out; }
""")
    st.caption("In static INT8 and QAT INT8 everything between the two stubs runs on int8 kernels. "
               "Dynamic INT8 only swaps the two Linear layers; the convolutions stay fp32.")
with right:
    st.subheader("Size by precision")
    fig = go.Figure(go.Bar(x=[LABELS[v["name"]] for v in rows], y=[v["size_bytes"] / 1024 for v in rows],
                           marker_color=[COLORS[v["name"]] for v in rows],
                           text=[f"{v['size_bytes'] / 1024:.0f} KB" for v in rows], textposition="outside"))
    fig.update_layout(height=320, margin=dict(t=10, b=10, l=10, r=10), yaxis_title="serialized state_dict (KB)")
    st.plotly_chart(fig, width="stretch")

st.subheader("The techniques")
st.markdown("""
| Method | What becomes int8 / fp16 | Needs data? | Training? | Main limitation |
|---|---|---|---|---|
| **FP32 baseline** | nothing | – | yes | the reference |
| **FP16** | weights and activations in half precision | no | no | x86 CPUs have no native fp16 math: it is *slower* here |
| **Dynamic INT8** | Linear weights; activations quantized on the fly | no | no | PyTorch has no dynamic Conv2d, so convs stay fp32 |
| **Static INT8 (PTQ)** | all weights and activations | ~1k calibration images | no | ranges are fixed from calibration; outliers can hurt |
| **QAT INT8** | all weights and activations | full training set | a short fine-tune | costs training time; needs the training pipeline |
""")
unsupported = [v for v in r["variants"] if v["status"] != "ok"]
for v in unsupported:
    st.warning(f"{v['label']}: not supported on this machine. {v['note']}")
st.caption(f"Measured on {r['environment']['processor']}, {r['environment']['threads']} threads, "
           f"{r['environment']['platform']}. Pipeline run took {r['total_seconds']:.0f} s.")
