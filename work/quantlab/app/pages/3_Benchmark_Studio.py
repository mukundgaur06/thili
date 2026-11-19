import pandas as pd
import streamlit as st

from common import LABELS, MODELS, ORDER, need_results, page, variants

page("Benchmark Studio", "⏱️")
r = need_results()
rows = variants(r)

st.markdown(
    "Measured by the pipeline under identical conditions: the same random input tensor, "
    f"**{r['environment']['threads']} threads**, untimed warm-up runs, then **5 interleaved rounds** (every model timed "
    "in turn, each round) pooled into one **median**: 500 runs at batch 1, 60 at batch 64. Interleaving keeps a burst of "
    "background load from landing on one model only. Peak memory is measured in a fresh process per model.")

table = []
for v in rows:
    mem = v.get("memory", {})
    table.append({
        "Model": LABELS[v["name"]],
        "Size (KB)": round(v["size_bytes"] / 1024, 1),
        "Compression": round(v.get("compression", 1.0), 2),
        "Accuracy": round(100 * v["accuracy"], 2),
        "Drop (pts)": round(v.get("accuracy_drop_pts", 0.0), 2),
        "Agree w/ FP32": round(100 * v["agreement_with_fp32"], 2),
        "bs1 median (ms)": round(v["latency"]["bs1"]["median_ms"], 3),
        "bs1 p90 (ms)": round(v["latency"]["bs1"]["p90_ms"], 3),
        "bs64 median (ms)": round(v["latency"]["bs64"]["median_ms"], 2),
        "Throughput bs64 (img/s)": round(v["latency"]["bs64"]["throughput_ips"]),
        "Speed-up bs64": round(v.get("speedup_bs64", 1.0), 2),
        "bs64 round medians (ms)": "{:.2f} to {:.2f}".format(*v["latency"]["bs64"].get("round_median_range_ms", [float("nan")] * 2)),
        "Peak RSS growth (MB)": mem.get("peak_growth_mb"),
    })
st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
for v in r["variants"]:
    if v["status"] != "ok":
        st.warning(f"{v['label']}: {v['note']}")
st.caption("Peak RSS growth: how much the process's peak resident memory rose above an idle Python process that "
           "had already imported torch, while loading the model and running 20 batches of 64. With weights under "
           "1 MB, activations and kernel workspaces dominate, so the differences are small.")

st.divider()
st.subheader("Re-run the latency benchmark now")
st.caption("Uses the exported TorchScript files, so every variant runs through the same runtime. Results vary with "
           "whatever else this machine is doing, which is exactly why the pipeline reports medians.")
c = st.columns(3)
batch = c[0].select_slider("Batch size", [1, 8, 32, 64, 256], 64)
iters = c[1].slider("Timed iterations", 10, 200, 50)
threads = c[2].slider("Threads", 1, 8, r["environment"]["threads"])
if st.button("Benchmark", type="primary"):
    import torch
    from quantlab.benchmarks import latency
    torch.set_num_threads(threads)
    out = []
    bar = st.progress(0.0)
    names = [n for n in ORDER if (MODELS / f"{n}.ts").exists()]
    for i, n in enumerate(names):
        m = torch.jit.load(str(MODELS / f"{n}.ts"))
        res = latency(m, batch, warmup=10, iters=iters, dtype=torch.float16 if n == "fp16" else torch.float32)
        out.append({"Model": LABELS[n], "median (ms)": round(res["median_ms"], 3), "p90 (ms)": round(res["p90_ms"], 3),
                    "min (ms)": round(res["min_ms"], 3), "throughput (img/s)": round(res["throughput_ips"])})
        bar.progress((i + 1) / len(names))
    df = pd.DataFrame(out)
    base = df.loc[df.Model == "FP32", "median (ms)"].iloc[0]
    df["speed-up vs FP32"] = (base / df["median (ms)"]).round(2)
    st.dataframe(df, hide_index=True, width="stretch")
