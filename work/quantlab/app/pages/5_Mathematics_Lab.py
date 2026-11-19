import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

from common import page
from quantlab.quantization.utilities import calc_qparams, dequantize, error_report, fake_quantize, quantize

page("Mathematics Lab", "➗")
st.latex(r"q = \mathrm{clamp}\!\left(\mathrm{round}\!\left(\frac{x}{s}\right) + z,\; q_{\min},\; q_{\max}\right)"
         r"\qquad \hat{x} = s\,(q - z)")
st.markdown("Affine: $s = \\dfrac{\\max - \\min}{q_{\\max} - q_{\\min}}$, $z = q_{\\min} - \\mathrm{round}(\\min / s)$. "
            "Symmetric: $s = \\dfrac{\\max(|\\min|,|\\max|)}{(q_{\\max} - q_{\\min})/2}$, $z = 0$. "
            "The range is always widened to include 0, so 0.0 is stored exactly.")

st.subheader("1 · Choose a range and a bit width")
c = st.columns(4)
lo = c[0].number_input("min", value=-1.5, step=0.1)
hi = c[1].number_input("max", value=4.0, step=0.1)
bits = c[2].select_slider("bits", [2, 3, 4, 6, 8], 8)
symmetric = c[3].toggle("symmetric", False)
p = calc_qparams(lo, hi, bits, symmetric=symmetric)
m = st.columns(4)
m[0].metric("scale s", f"{p.scale:.6f}")
m[1].metric("zero point z", p.zero_point)
m[2].metric("integer range", f"[{p.q_min}, {p.q_max}]")
m[3].metric("float range covered", f"[{p.float_range[0]:.3f}, {p.float_range[1]:.3f}]")
if symmetric and lo >= 0:
    st.info("All values are ≥ 0, yet symmetric quantization reserves half the codes for negatives: the step is "
            "twice as coarse as affine would give. This is why activations after ReLU use affine uint8.")

st.subheader("2 · Quantize one number by hand")
x0 = st.slider("x", float(min(lo, 0) - 1), float(max(hi, 0) + 1), 1.2345, 0.0005)
q0 = int(quantize(torch.tensor([x0]), p)[0])
x_hat = float(dequantize(torch.tensor([q0]), p)[0])
raw = x0 / p.scale + p.zero_point
st.markdown(f"""
| step | value |
|---|---|
| x / s + z | {x0:.4f} / {p.scale:.6f} + {p.zero_point} = **{raw:.3f}** |
| round, then clamp to [{p.q_min}, {p.q_max}] | **{q0}**{"  ← clipped" if round(raw) != q0 else ""} |
| dequantize: s · (q − z) | {p.scale:.6f} · ({q0} − {p.zero_point}) = **{x_hat:.5f}** |
| error | {x_hat - x0:+.5f} (half a step is ±{p.scale / 2:.5f}) |
""")

st.subheader("3 · A whole tensor: rounding error and clipping")
c = st.columns(3)
dist = c[0].selectbox("Values", ["normal", "uniform", "normal + outliers", "ReLU outputs"])
n = c[1].select_slider("How many", [1000, 10000, 100000], 10000)
seed = c[2].number_input("seed", 0, 999, 0)
g = torch.Generator().manual_seed(int(seed))
x = {"normal": lambda: torch.randn(n, generator=g),
     "uniform": lambda: torch.rand(n, generator=g) * 4 - 1,
     "normal + outliers": lambda: torch.cat([torch.randn(n - 10, generator=g), torch.randn(10, generator=g) * 12]),
     "ReLU outputs": lambda: torch.relu(torch.randn(n, generator=g) * 1.5)}[dist]()
rep = error_report(x, p)
xq = fake_quantize(x, p)
cols = st.columns(4)
cols[0].metric("SQNR", f"{rep['sqnr_db']:.1f} dB")
cols[1].metric("mean squared error", f"{rep['mse']:.2e}")
cols[2].metric("max |error|", f"{rep['max_abs_error']:.4f}")
cols[3].metric("clipped", f"{rep['clipped_share']:.2%}")
fig = go.Figure()
fig.add_trace(go.Histogram(x=x.numpy(), nbinsx=150, name="fp32 values", opacity=0.55))
levels = dequantize(torch.arange(p.q_min, p.q_max + 1), p).numpy()
if len(levels) <= 64:
    for lv in levels:
        fig.add_vline(x=float(lv), line_width=1, line_color="rgba(200,80,40,.45)")
fig.add_vrect(x0=p.float_range[0], x1=p.float_range[1], fillcolor="rgba(60,130,200,.07)", line_width=0)
fig.update_layout(height=320, margin=dict(t=10), xaxis_title="value", showlegend=False)
st.plotly_chart(fig, width="stretch")
st.caption("Shaded: the representable range. Vertical lines (shown at ≤ 6 bits): the quantization levels. Try "
           "'normal + outliers' with min/max set to the data's full range to see why a few outliers waste most of the codes.")

st.subheader("4 · Calibration: where the range comes from")
st.markdown("Static quantization measures ranges on real data. Two observers on the same tensor:")
from torch.ao.quantization.observer import HistogramObserver, MinMaxObserver  # noqa: E402
data = torch.cat([torch.relu(torch.randn(20000, generator=torch.Generator().manual_seed(1))),
                  torch.tensor([25.0, 30.0])])
rows = []
for name, obs in (("MinMax", MinMaxObserver()), ("Histogram (minimise L2 error)", HistogramObserver())):
    obs(data)
    s, z = obs.calculate_qparams()
    pp = calc_qparams(0.0, float(s) * (255 - int(z)))
    rows.append({"observer": name, "range max": f"{float(s) * (255 - int(z)):.2f}", "scale": f"{float(s):.5f}",
                 "zero point": int(z), "SQNR on this tensor (dB)": f"{error_report(data, pp)['sqnr_db']:.1f}",
                 "clipped": f"{error_report(data, pp)['clipped_share']:.3%}"})
st.table(rows)
st.caption("20,000 ReLU-like values plus two outliers (25 and 30). MinMax stretches the range to 30, so the step is "
           "~0.12 and most values get only a few levels. The histogram observer clips the two outliers and spends "
           "the 256 levels on where the data actually is.")

st.subheader("5 · Fake quantization and the straight-through estimator")
w = torch.linspace(-1, 1, 400, requires_grad=True)
pf = calc_qparams(-1, 1, 3)
fq = torch.fake_quantize_per_tensor_affine(w, pf.scale, pf.zero_point, pf.q_min, pf.q_max)
fq.sum().backward()
fig = go.Figure([go.Scatter(x=w.detach().numpy(), y=w.detach().numpy(), name="identity", line=dict(dash="dot")),
                 go.Scatter(x=w.detach().numpy(), y=fq.detach().numpy(), name="fake-quantized (3 bits)"),
                 go.Scatter(x=w.detach().numpy(), y=w.grad.numpy(), name="gradient used in training", yaxis="y2")])
fig.update_layout(height=320, margin=dict(t=10), yaxis2=dict(overlaying="y", side="right", range=[-0.1, 1.5], title="gradient"))
st.plotly_chart(fig, width="stretch")
st.caption("Forward: a staircase (the value really is rounded). Backward: PyTorch passes the gradient through as 1 "
           "inside the range and 0 outside it (clipped values get no gradient). That is how QAT trains through rounding.")
