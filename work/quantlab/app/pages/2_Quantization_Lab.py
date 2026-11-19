import time

import pandas as pd
import streamlit as st

from common import MODELS, dataset, page

page("Quantization Lab", "🔬")
ckpts = sorted(p.name for p in MODELS.glob("*.pt"))
if not ckpts:
    st.warning("No trained model yet: run `python -m quantlab.pipeline` or use the Model Training page.")
    st.stop()

left, right = st.columns([1, 2])
with left:
    ck_name = st.selectbox("Trained model", ckpts, index=ckpts.index("fp32.pt") if "fp32.pt" in ckpts else 0)
    method = st.radio("Method", ["Static INT8 (PTQ)", "Dynamic INT8", "QAT INT8", "FP16"])
    if method == "Static INT8 (PTQ)":
        observer = st.selectbox("Activation observer", ["histogram", "minmax", "moving_average"],
                                help="How the activation range is estimated from calibration data.")
        per_channel = st.toggle("Per-channel weight scales", True)
        n_cal = st.select_slider("Calibration images (from the training split)", [16, 64, 256, 1024, 4096], 1024)
    elif method == "QAT INT8":
        qat_epochs = st.slider("QAT epochs", 1, 4, 1)
        qat_lr = st.select_slider("Learning rate", [5e-5, 1e-4, 2e-4, 5e-4], 2e-4)
    run = st.button("Run conversion", type="primary")

with right:
    st.markdown({
        "Static INT8 (PTQ)": """**Static post-training quantization**
1. *Fuse* Conv+BN+ReLU and Linear+ReLU; BatchNorm is folded into the conv weights.
2. *Prepare*: an observer is attached after every op. It records ranges, it changes nothing.
3. *Calibrate*: real training images flow through; each observer now holds a measured min/max (or histogram).
4. *Convert*: each range becomes `scale = (max − min) / 255`, `zero_point = −round(min / scale)`; every module is swapped for its int8 kernel.""",
        "Dynamic INT8": """**Dynamic quantization**
Linear weights become int8 once. At run time each Linear measures *its current input's* range,
quantizes it, multiplies in int8 and dequantizes. No calibration data needed. PyTorch has no
dynamic Conv2d, so the convolutions stay fp32.""",
        "QAT INT8": """**Quantization-aware training**
Fake-quantize modules (quantize → dequantize) are inserted on weights and activations, then the
model is fine-tuned. The forward pass already sees int8 rounding, so the weights adapt to it;
gradients pass through the rounding unchanged (straight-through estimator). BN statistics and
observer ranges are frozen for the last epoch, then the model is converted like static PTQ.""",
        "FP16": """**Half precision**
Every weight and activation is stored in 16-bit floating point. Whether it runs, and whether it is
faster, depends on the CPU: this is tested, not assumed.""",
    }[method])

if run:
    import torch
    from quantlab.benchmarks import latency, state_dict_bytes
    from quantlab.evaluate import agreement, evaluate
    from quantlab.train import load_checkpoint

    fp32, ck = load_checkpoint(MODELS / ck_name)
    d = dataset(ck.get("dataset", "fashion_mnist"))
    torch.set_num_threads(4)
    base_eval = evaluate(fp32, d)
    with st.status(f"{method} on {ck_name}", expanded=True) as status:
        t = time.time()
        dtype = torch.float32
        stats = None
        if method == "Static INT8 (PTQ)":
            from quantlab.quantization.static import (StaticConfig, calibrate, convert_static, observer_stats,
                                                      prepare_static)
            cfg = StaticConfig(observer=observer, per_channel=per_channel, calibration_images=n_cal)
            st.write("Fusing modules and attaching observers…")
            prepared = prepare_static(fp32, cfg)
            st.write(f"Calibrating on {n_cal} training images…")
            calibrate(prepared, d.calibration_set(n_cal), 128)
            stats = observer_stats(prepared)
            st.write("Computing scales and zero points, swapping in int8 kernels…")
            q = convert_static(prepared)
        elif method == "Dynamic INT8":
            from quantlab.quantization.dynamic import quantize_dynamic, quantized_layers
            q = quantize_dynamic(fp32)
            st.write(quantized_layers(q))
        elif method == "QAT INT8":
            from quantlab.quantization.qat import QATConfig, quantize_qat
            st.write(f"Fine-tuning with fake quantization for {qat_epochs} epoch(s)…")
            q, hist = quantize_qat(fp32, d, QATConfig(epochs=qat_epochs, lr=qat_lr, freeze_bn_epoch=qat_epochs,
                                                      freeze_observer_epoch=qat_epochs),
                                   on_epoch=lambda h: st.write(f"epoch {h['epoch']}: val accuracy {h['val_accuracy']:.2%}"))
        else:
            from quantlab.quantization.fp16 import check_support, to_half
            ok, reason = check_support(fp32)
            st.write(reason)
            if not ok:
                status.update(label="FP16 is not supported here", state="error")
                st.stop()
            q, dtype = to_half(fp32), torch.float16
        ev = evaluate(q, d, dtype=dtype)
        status.update(label=f"Done in {time.time() - t:.1f} s", state="complete")

    c = st.columns(4)
    c[0].metric("Test accuracy", f"{ev['accuracy']:.2%}", f"{100 * (ev['accuracy'] - base_eval['accuracy']):+.2f} pts")
    c[1].metric("Size", f"{state_dict_bytes(q) / 1024:.0f} KB", f"{state_dict_bytes(fp32) / state_dict_bytes(q):.2f}× smaller")
    lat_q, lat_f = latency(q, 64, iters=20, dtype=dtype), latency(fp32, 64, iters=20)
    c[2].metric("Latency, batch 64", f"{lat_q['median_ms']:.2f} ms", f"{lat_f['median_ms'] / lat_q['median_ms']:.2f}× vs FP32")
    c[3].metric("Same prediction as FP32", f"{agreement(ev['pred'], base_eval['pred']):.2%}")
    if stats:
        st.subheader("What calibration measured")
        st.dataframe(pd.DataFrame(stats), hide_index=True, width="stretch")
        st.caption("Activation rows: the observer's min/max over the calibration images and the resulting uint8 "
                   "scale and zero point. Weight rows: int8 symmetric (zero point 0); per-channel scale is the mean "
                   "over output channels. Note that every activation after a fused ReLU has min 0, so its zero point is 0.")
