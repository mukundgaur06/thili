"""LoRA from Scratch: demonstration dashboard.

    streamlit run dashboard.py

Tabs:
* Results: the measured numbers from results/*.json
* Try the adapters: classify your own sentences with the trained adapters, hot-swapping them
  on one frozen GPT-2
* LoRA maths: how rank and targets set the parameter count
* QLoRA / NF4: quantize a weight tensor to 4 bits live
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
ADAPTERS = ROOT / "adapters"

st.set_page_config(page_title="LoRA from Scratch", page_icon="🧩", layout="wide")
st.title("LoRA from Scratch")
st.caption("Parameter-efficient fine-tuning of real GPT-2 (124M) for SST-2 sentiment, written by hand: "
           "LoRALinear, injection, merge, adapter hot-swap and a 4-bit NF4 QLoRA base.")


def load_json(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


tab1, tab2, tab3, tab4 = st.tabs(["Results", "Try the adapters", "LoRA maths", "QLoRA / NF4"])

# ------------------------------------------------------------------------------------ results
with tab1:
    runs = load_json("results.json") or []
    if runs:
        df = pd.DataFrame([{"run": r["name"], "mode": r["mode"], "rank": r.get("rank"), "train sentences": r.get("n_train"),
                            "trainable params": r["trainable_params"], "% of GPT-2": round(r["trainable_percent"], 3),
                            "adapter (KB)": round(r["adapter_bytes"] / 1024), "test accuracy": r["test_acc"],
                            "peak RAM (MB)": r.get("peak_rss_mb")} for r in runs])
        c = st.columns(4)
        best = df.loc[df["test accuracy"].idxmax()]
        c[0].metric("Best test accuracy", f"{best['test accuracy']:.1%}", best["run"])
        c[1].metric("Head only (linear probe)", f"{df.loc[df['mode'] == 'head', 'test accuracy'].max():.1%}")
        r8 = df[df.run == "lora-r8-c_attn"]
        if len(r8):
            c[2].metric("LoRA r=8 trainable", f"{int(r8['trainable params'].iloc[0]):,}", f"{r8['% of GPT-2'].iloc[0]:.2f}% of GPT-2")
            c[3].metric("LoRA r=8 adapter", f"{int(r8['adapter (KB)'].iloc[0])} KB", "GPT-2 itself: 498 MB")
        st.dataframe(df, hide_index=True, width="stretch")
        fig = go.Figure(go.Bar(x=df["run"], y=df["test accuracy"], text=[f"{a:.1%}" for a in df["test accuracy"]],
                               textposition="outside"))
        fig.update_layout(height=320, yaxis_tickformat=".0%", yaxis_range=[0.7, 0.95], margin=dict(t=10),
                          yaxis_title="SST-2 test accuracy (872 sentences)")
        st.plotly_chart(fig, width="stretch")
        st.caption("One standard error on 872 test sentences is about ±1.2 points, so r = 1 and r = 8 are not clearly different.")
    ms = load_json("merge_and_swap.json")
    if ms:
        st.subheader("Merge and hot-swap, measured")
        m = ms["merge"]
        c = st.columns(4)
        c[0].metric("Max logit change after merging", f"{m['max_logit_diff']:.1e}", "float rounding")
        c[1].metric("Accuracy unmerged → merged", f"{m['acc_unmerged']:.2%} → {m['acc_merged']:.2%}")
        sw = next(iter(ms["swap"]["adapters"].values()))
        c[2].metric("Activate an adapter", f"{sw['swap_ms']:.1f} ms")
        c[3].metric("Reload the whole base model", f"{ms['swap']['full_reload_ms']:,} ms")
    fv = load_json("full_vs_lora_minigpt.json")
    if fv:
        st.subheader("Full fine-tuning vs LoRA, same optimizer budget (mini-GPT, 3 seeds)")
        d = pd.DataFrame(fv)
        d["method"] = d["mode"] + d["rank"].apply(lambda r: f" r={int(r)}" if pd.notna(r) else "")
        g = d.groupby("method").agg(trainable=("trainable", "first"), pct=("percent", "first"),
                                    test_mean=("test", "mean"), test_min=("test", "min"), test_max=("test", "max")).reset_index()
        st.dataframe(g.rename(columns={"pct": "% trainable", "test_mean": "test acc (mean)", "test_min": "min", "test_max": "max"})
                     .round(4), hide_index=True, width="stretch")

# ------------------------------------------------------------------------------------ live adapters
with tab2:
    # QLoRA adapters were trained against a 4-bit base; this tab loads the fp32 base, so they are left out
    files = sorted(f for f in ADAPTERS.glob("*.safetensors") if not f.stem.startswith("qlora"))
    if not files:
        st.info("No adapters in adapters/: run `python finetune.py --mode lora --rank 8` first.")
    else:
        from lora.gpt2 import GPT2Classifier, load_gpt2, load_tokenizer
        from lora.inject import inject_lora, load_adapter, read_adapter, set_lora_enabled

        @st.cache_resource(show_spinner="Loading GPT-2 (124M) once…")
        def base_for_rank(rank: int):
            torch.set_num_threads(4)
            m = GPT2Classifier(load_gpt2()).eval()
            inject_lora(m, ("c_attn",), r=rank, alpha=16.0)
            return m

        @st.cache_resource
        def tokenizer():
            return load_tokenizer()

        names = [f.stem for f in files]
        pick = st.selectbox("Adapter", names, index=names.index("lora-r1-c_attn") if "lora-r1-c_attn" in names else 0)
        sd, meta = read_adapter(ADAPTERS / f"{pick}.safetensors")
        rank = int(meta.get("rank", 8))
        model = base_for_rank(rank)
        t = time.perf_counter()
        set_lora_enabled(model, meta.get("mode") != "head")          # a head-only adapter = base GPT-2 + head
        load_adapter(model, {k: v.float() for k, v in sd.items()})
        swap_ms = (time.perf_counter() - t) * 1000
        c = st.columns(4)
        c[0].metric("Adapter file", f"{(ADAPTERS / f'{pick}.safetensors').stat().st_size / 1024:.0f} KB")
        c[1].metric("Numbers in it", f"{sum(v.numel() for v in sd.values()):,}")
        c[2].metric("Swap time", f"{swap_ms:.1f} ms", "base GPT-2 stays in memory")
        c[3].metric("Its measured test accuracy", f"{float(meta.get('test_acc', 'nan')):.1%}")
        text = st.text_area("Sentences, one per line", "a gripping, beautifully acted film\nthe plot is a mess and the jokes fall flat\n"
                            "not as bad as I feared, but hardly memorable")
        lines = [s.strip() for s in text.splitlines() if s.strip()][:20]
        if lines:
            tok = tokenizer()
            ids = [tok.encode(s).ids[:64] for s in lines]
            width = max(len(i) for i in ids)
            x = torch.full((len(ids), width), 50256, dtype=torch.long)
            for r, i in enumerate(ids):
                x[r, :len(i)] = torch.tensor(i)
            with torch.no_grad():
                p = model(x, torch.tensor([len(i) for i in ids])).softmax(-1)[:, 1].numpy()
            st.dataframe(pd.DataFrame({"sentence": lines, "P(positive)": np.round(p, 3),
                                       "label": ["positive" if v >= 0.5 else "negative" for v in p]}),
                         hide_index=True, width="stretch")
        st.caption("One frozen GPT-2 per LoRA rank is kept in memory; switching adapters copies a few hundred KB of "
                   "A/B matrices into it instead of reloading 500 MB of weights. This is how multi-adapter serving works.")

# ------------------------------------------------------------------------------------ LoRA maths
with tab3:
    st.latex(r"h = W_0x + \frac{\alpha}{r}\,BAx,\qquad A\in\mathbb{R}^{r\times d_\text{in}}\ (\text{random}),\ "
             r"B\in\mathbb{R}^{d_\text{out}\times r}\ (\text{zero})")
    st.markdown("B starts at zero, so training starts exactly at the pretrained model. ΔW = (α/r)·BA has rank ≤ r and "
                "r·(d_in + d_out) parameters instead of d_in·d_out.")
    c1, c2 = st.columns(2)
    r = c1.select_slider("Rank r", [1, 2, 4, 8, 16, 32, 64], 8)
    targets = c2.multiselect("Adapt these GPT-2 layers (×12 blocks)", ["c_attn (768→2304)", "attn c_proj (768→768)",
                                                                         "c_fc (768→3072)", "mlp c_proj (3072→768)"],
                             ["c_attn (768→2304)"])
    dims = {"c_attn (768→2304)": (768, 2304), "attn c_proj (768→768)": (768, 768), "c_fc (768→3072)": (768, 3072),
            "mlp c_proj (3072→768)": (3072, 768)}
    lora_params = 12 * sum(r * (dims[t][0] + dims[t][1]) for t in targets)
    full = 12 * sum(dims[t][0] * dims[t][1] for t in targets)
    c = st.columns(3)
    c[0].metric("LoRA parameters", f"{lora_params:,}", f"{100 * lora_params / 124_439_808:.3f}% of GPT-2")
    c[1].metric("Those layers in full", f"{full:,}", f"{full / max(1, lora_params):.0f}× more")
    c[2].metric("fp16 adapter size", f"{2 * lora_params / 1024:.0f} KB")
    st.markdown("| | Weights | Gradients | AdamW m + v | Total before activations |\n|---|---:|---:|---:|---:|\n"
                "| Full fine-tuning, fp32 | 498 MB | 498 MB | 996 MB | ≈ 2.0 GB |\n"
                f"| LoRA (this setting) | 498 MB (frozen) | {4 * lora_params / 2**20:.1f} MB | {8 * lora_params / 2**20:.1f} MB | "
                f"≈ {(498 + 12 * lora_params / 2**20) / 1024:.2f} GB |")

# ------------------------------------------------------------------------------------ QLoRA
with tab4:
    from lora.qlora import INT4_LEVELS, NF4_REFERENCE, dequantize_blockwise, nf4_levels, quantize_blockwise
    st.markdown("**NF4** puts the 16 levels at quantiles of N(0, 1), so normally distributed weights use every code "
                "equally often. Block-wise absmax (one scale per 64 weights) keeps an outlier from ruining the rest.")
    lv = nf4_levels()
    st.caption(f"Derived NF4 levels match the published QLoRA table to {float((lv - torch.tensor(NF4_REFERENCE)).abs().max()):.1e}.")
    c1, c2, c3 = st.columns(3)
    block = c1.select_slider("Block size", [16, 32, 64, 128, 256, 4096], 64)
    outliers = c2.slider("Outliers (×20 std)", 0, 20, 2)
    seed = c3.number_input("Seed", 0, 99, 0)
    g = torch.Generator().manual_seed(int(seed))
    w = torch.randn(4096, generator=g) * 0.02
    if outliers:
        w[torch.randperm(4096, generator=g)[:outliers]] *= 20
    rows = []
    for name, levels in (("NF4", lv), ("uniform int4", INT4_LEVELS)):
        codes, absmax, n = quantize_blockwise(w, levels, block)
        wq = dequantize_blockwise(codes, absmax, levels, n, w.shape)
        freq = torch.bincount(codes.flatten().long(), minlength=16).float() / codes.numel()
        ent = float(-(freq[freq > 0] * freq[freq > 0].log2()).sum())
        rows.append({"format": name, "MSE": float((wq - w).pow(2).mean()), "SQNR (dB)": float(10 * torch.log10(w.pow(2).mean() / (wq - w).pow(2).mean())),
                     "code entropy (bits, max 4)": ent})
    st.dataframe(pd.DataFrame(rows).round(6), hide_index=True, width="stretch")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=(w / w.abs().max()).numpy(), nbinsx=120, name="weights / absmax", opacity=0.6))
    for v in lv.tolist():
        fig.add_vline(x=v, line_color="rgba(200,70,40,.6)", line_width=1)
    fig.update_layout(height=300, margin=dict(t=10), showlegend=False, xaxis_title="normalised weight (red lines: NF4 levels)")
    st.plotly_chart(fig, width="stretch")
    st.caption("Storage: 4 bits per weight plus one scale per block. Double quantization stores those scales in int8, "
               "for about 4.13 bits per weight, 12.9% of fp32.")
