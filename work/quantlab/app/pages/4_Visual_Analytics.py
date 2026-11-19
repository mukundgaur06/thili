import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import COLORS, LABELS, need_results, page, variants

page("Visual Analytics", "📊")
r = need_results()
rows = variants(r)
names = [LABELS[v["name"]] for v in rows]
colors = [COLORS[v["name"]] for v in rows]


def bars(values, title, fmt, yaxis):
    fig = go.Figure(go.Bar(x=names, y=values, marker_color=colors, text=[fmt(v) for v in values], textposition="outside"))
    fig.update_layout(title=title, height=330, margin=dict(t=40, b=10, l=10, r=10), yaxis_title=yaxis)
    return fig


c1, c2, c3 = st.columns(3)
acc = [100 * v["accuracy"] for v in rows]
fig = bars(acc, "Test accuracy", lambda x: f"{x:.2f}%", "%")
fig.update_yaxes(range=[min(acc) - 1.5, max(acc) + 0.8])
c1.plotly_chart(fig, width="stretch")
c2.plotly_chart(bars([v["size_bytes"] / 1024 for v in rows], "Model size", lambda x: f"{x:.0f} KB", "KB"),
                width="stretch")
c3.plotly_chart(bars([v["latency"]["bs64"]["median_ms"] for v in rows], "Latency, batch 64 (median)",
                     lambda x: f"{x:.2f} ms", "ms"), width="stretch")

st.subheader("Compression versus accuracy")
fig = go.Figure()
for v in rows:
    fig.add_trace(go.Scatter(x=[v.get("compression", 1)], y=[100 * v["accuracy"]], mode="markers+text",
                             text=[LABELS[v["name"]]], textposition="top center", name=LABELS[v["name"]],
                             marker=dict(size=10 + 30 * v["latency"]["bs64"]["throughput_ips"] /
                                         max(x["latency"]["bs64"]["throughput_ips"] for x in rows),
                                         color=COLORS[v["name"]])))
fig.update_layout(height=380, showlegend=False, xaxis_title="compression (× smaller than FP32)",
                  yaxis_title="test accuracy (%)", margin=dict(t=10))
st.plotly_chart(fig, width="stretch")
st.caption("Marker size is throughput at batch 64. Top right is best: smaller and still accurate.")

st.subheader("Per-class performance")
pc = pd.DataFrame([{"model": LABELS[v["name"]], **row} for v in rows for row in v["per_class"]])
metric = st.radio("Metric", ["f1", "precision", "recall"], horizontal=True)
fig = px.bar(pc, x="class", y=metric, color="model", barmode="group",
             color_discrete_map={LABELS[k]: c for k, c in COLORS.items()})
fig.update_layout(height=380, yaxis_range=[max(0, pc[metric].min() - 0.05), 1], margin=dict(t=10))
st.plotly_chart(fig, width="stretch")

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    pick = st.selectbox("Model", names, index=names.index("Static INT8") if "Static INT8" in names else 0)
    v = rows[names.index(pick)]
    cm = np.array(v["confusion_matrix"])
    fig = px.imshow(cm, x=r["classes"], y=r["classes"], text_auto=True, color_continuous_scale="Blues",
                    labels=dict(x="predicted", y="true", color="images"))
    fig.update_layout(height=520, margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")
with right:
    st.subheader("Where INT8 changes the answer")
    base = np.array(rows[0]["confusion_matrix"])
    fig = px.imshow(cm - base, x=r["classes"], y=r["classes"], text_auto=True, color_continuous_scale="RdBu",
                    color_continuous_midpoint=0, labels=dict(x="predicted", y="true", color="Δ images"))
    fig.update_layout(height=520, margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"{pick} minus FP32. Most cells are 0: quantization moves only a handful of borderline images, "
               "mostly among the classes FP32 already confuses (shirt / T-shirt / pullover / coat).")

if r.get("static_ablation"):
    st.subheader("Static PTQ: observer, weight granularity, calibration size")
    ab = pd.DataFrame(r["static_ablation"])
    ab["setting"] = ab.observer + " · " + np.where(ab.per_channel, "per-channel", "per-tensor")
    fig = px.line(ab, x="calibration_images", y="accuracy", color="setting", markers=True, log_x=True)
    fig.add_hline(y=rows[0]["accuracy"], line_dash="dot", annotation_text="FP32")
    fig.update_layout(height=380, yaxis_tickformat=".2%", xaxis_title="calibration images (log scale)", margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")

qat = next((v for v in rows if v["name"] == "qat_int8"), None)
static = next((v for v in rows if v["name"] == "static_int8"), None)
if qat and static:
    st.subheader("How faithfully int8 weights represent the FP32 weights")
    w = pd.DataFrame([{"model": LABELS[v["name"]], **x} for v in (static, qat) for x in v["weight_sqnr"]])
    fig = px.bar(w, x="layer", y="sqnr_db", color="model", barmode="group",
                 color_discrete_map={LABELS[k]: c for k, c in COLORS.items()})
    fig.update_layout(height=320, yaxis_title="weight SQNR (dB, higher = closer)", margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("Each additional ~6 dB is one more bit of precision. QAT's weights are compared with the FP32 model "
               "they started from; they have also moved during fine-tuning, so lower SQNR there is expected.")
