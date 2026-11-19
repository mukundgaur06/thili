"""Run the whole study end to end and write ``artifacts/results.json``.

    python -m quantlab.pipeline                    # train 8 epochs, then every method
    python -m quantlab.pipeline --skip-train       # reuse artifacts/models/fp32.pt

Every variant is evaluated on the same 10,000 test images and benchmarked with the same
inputs, thread count and iteration counts.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from . import data as data_mod
from .benchmarks import compare, environment, interleaved_latency, peak_memory, state_dict_bytes
from .evaluate import agreement, evaluate
from .export import export_onnx, export_torchscript, write_manifest
from .model import count_parameters, layer_table
from .quantization.dynamic import quantize_dynamic, quantized_layers
from .quantization.fp16 import check_support, to_half
from .quantization.qat import QATConfig, quantize_qat
from .quantization.static import StaticConfig, quantize_static
from .quantization.utilities import pick_engine, sqnr
from .train import TrainConfig, load_checkpoint, save_checkpoint, train_baseline

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
MODELS = ART / "models"


def int8_weight_sqnr(fp32: nn.Module, qmodel: nn.Module) -> list:
    """Per layer: how faithfully the int8 weights represent the (BN-folded) fp32 weights, in dB."""
    import copy
    ref = copy.deepcopy(fp32).eval().fuse(qat=False)
    rows = []
    for name in ("conv1", "conv2", "fc1", "fc2"):
        q = getattr(qmodel, name, None)
        r = getattr(ref, name)
        w_ref = (r[0] if isinstance(r, nn.Sequential) else r).weight.detach()
        if q is None or not callable(getattr(q, "weight", None)):
            continue
        w_q = q.weight()
        rows.append({"layer": name, "sqnr_db": sqnr(w_ref, w_q.dequantize()),
                     "scheme": str(w_q.qscheme()).replace("torch.", ""), "dtype": str(w_q.dtype).replace("torch.", "")})
    return rows


def record(name: str, label: str, model: nn.Module, d, base_pred=None, dtype=torch.float32, note: str = "") -> dict:
    t = time.time()
    ev = evaluate(model, d, dtype=dtype)
    row = {"name": name, "label": label, "status": "ok", "note": note, "accuracy": ev["accuracy"],
           "test_loss": ev["loss"], "per_class": ev["per_class"], "confusion_matrix": ev["confusion_matrix"],
           "size_bytes": state_dict_bytes(model), "eval_seconds": round(time.time() - t, 2)}
    row["agreement_with_fp32"] = agreement(ev["pred"], base_pred) if base_pred is not None else 1.0
    row["_pred"] = ev["pred"]
    print(f"  {label:<22} acc {row['accuracy']:.4f}  size {row['size_bytes'] / 1024:7.1f} KB", flush=True)
    return row


def ablations(fp32: nn.Module, d) -> list:
    """Static PTQ under different observers, weight granularities and calibration sizes."""
    rows = []
    for obs in ("minmax", "histogram"):
        for per_channel in (False, True):
            for n in (32, 256, 2048):
                cfg = StaticConfig(observer=obs, per_channel=per_channel, calibration_images=n)
                q, _ = quantize_static(fp32, d.calibration_set(n, seed=1), cfg)
                rows.append({**cfg.to_dict(), "accuracy": evaluate(q, d)["accuracy"]})
                print(f"  ablation {obs:9s} per_channel={per_channel!s:5s} n={n:4d}: {rows[-1]['accuracy']:.4f}", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fashion_mnist", choices=["fashion_mnist", "mnist"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--qat-epochs", type=int, default=2)
    ap.add_argument("--calibration-images", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-ablations", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    engine = pick_engine()
    MODELS.mkdir(parents=True, exist_ok=True)
    d = data_mod.load(args.dataset)
    print(f"{args.dataset}: {len(d.train)} train / {len(d.val)} val / {len(d.test)} test | engine {engine}")
    t_start = time.time()

    ck_path = MODELS / "fp32.pt"
    if args.skip_train and ck_path.exists():
        fp32, ck = load_checkpoint(ck_path)
        history, train_seconds = ck["history"], None
    else:
        cfg = TrainConfig(epochs=args.epochs, threads=args.threads)
        t = time.time()
        fp32, history = train_baseline(d, cfg=cfg, on_epoch=lambda r: print(
            f"  epoch {r['epoch']}: train loss {r['train_loss']:.4f}  val loss {r['val_loss']:.4f}  "
            f"val acc {r['val_accuracy']:.4f}  ({r['seconds']}s)", flush=True))
        train_seconds = round(time.time() - t, 1)
        save_checkpoint(ck_path, fp32, history, cfg, args.dataset)

    example = d.normalise(d.test.x[:64])
    print("evaluating and benchmarking")
    rows, models = [], {}
    base = record("fp32", "FP32 baseline", fp32, d)
    rows.append(base)
    models["fp32"] = fp32

    ok, reason = check_support(fp32, torch.float16)
    if ok:
        half = to_half(fp32)
        r = record("fp16", "FP16 (CPU)", half, d, base["_pred"], dtype=torch.float16, note=reason)
        rows.append(r)
        models["fp16"] = half
    else:
        rows.append({"name": "fp16", "label": "FP16 (CPU)", "status": "unsupported", "note": reason})
        print(f"  FP16 unsupported: {reason}")

    dyn = quantize_dynamic(fp32)
    r = record("dynamic_int8", "Dynamic INT8", dyn, d, base["_pred"],
               note="int8 weights for Linear layers; convolutions stay fp32 (PyTorch has no dynamic Conv2d)")
    r["layers"] = quantized_layers(dyn)
    rows.append(r)
    models["dynamic_int8"] = dyn

    scfg = StaticConfig(calibration_images=args.calibration_images)
    t = time.time()
    stat, obs = quantize_static(fp32, d.calibration_set(scfg.calibration_images, scfg.seed), scfg)
    r = record("static_int8", "Static INT8 (PTQ)", stat, d, base["_pred"],
               note=f"{scfg.observer} observer, per-channel weights, {scfg.calibration_images} calibration images")
    r.update({"config": scfg.to_dict(), "observers": obs, "weight_sqnr": int8_weight_sqnr(fp32, stat),
              "convert_seconds": round(time.time() - t, 2)})
    rows.append(r)
    models["static_int8"] = stat

    qcfg = QATConfig(epochs=args.qat_epochs)
    t = time.time()
    qat, qat_hist = quantize_qat(fp32, d, qcfg, on_epoch=lambda h: print(
        f"  QAT epoch {h['epoch']}: val acc {h['val_accuracy']:.4f}", flush=True))
    r = record("qat_int8", "QAT INT8", qat, d, base["_pred"],
               note=f"{qcfg.epochs} fine-tuning epochs with fake quantization from the fp32 weights")
    r.update({"config": qcfg.to_dict(), "history": qat_hist, "weight_sqnr": int8_weight_sqnr(fp32, qat),
              "train_seconds": round(time.time() - t, 1)})
    rows.append(r)
    models["qat_int8"] = qat

    print("timing all variants, interleaved (5 rounds)")
    lat = interleaved_latency(models, {"fp16": torch.float16})
    for r in rows:
        if r["name"] in lat:
            r["latency"] = lat[r["name"]]
            print(f"  {r['label']:<22} bs1 {r['latency']['bs1']['median_ms']:.3f} ms  "
                  f"bs64 {r['latency']['bs64']['median_ms']:.2f} ms  (round medians "
                  f"{r['latency']['bs64']['round_median_range_ms'][0]:.2f} to {r['latency']['bs64']['round_median_range_ms'][1]:.2f})", flush=True)

    print("exporting and measuring memory in fresh processes")
    exports = {}
    for name, m in models.items():
        ex = example.half() if name == "fp16" else example
        ts = export_torchscript(m, MODELS / f"{name}.ts", ex)
        exports[name] = [ts]
        if name in ("fp32", "static_int8", "qat_int8"):
            exports[name].append(export_onnx(m, MODELS / f"{name}.onnx", ex, d.normalise(d.test.x),
                                             quantized=name != "fp32"))
        row = next(r for r in rows if r["name"] == name)
        row["torchscript_bytes"] = ts.get("bytes")
        if ts.get("ok"):
            row["memory"] = peak_memory(MODELS / f"{name}.ts", 64, "float16" if name == "fp16" else "float32", args.threads)
    for entries in exports.values():                 # store paths relative to the project, not this machine
        for e in entries:
            e["path"] = Path(e["path"]).relative_to(ROOT).as_posix()
    write_manifest(ART / "export_manifest.json", exports, {"dataset": args.dataset, "torch": torch.__version__})

    abl = [] if args.skip_ablations else ablations(fp32, d)

    for r in rows:
        r.pop("_pred", None)
    results = {
        "environment": environment(), "dataset": args.dataset, "classes": d.classes,
        "split_sizes": {"train": len(d.train), "val": len(d.val), "test": len(d.test)},
        "model": {"config": fp32.cfg.to_dict(), "parameters": count_parameters(fp32), "layers": layer_table(fp32)},
        "training": {"history": history, "seconds": train_seconds, "epochs": len(history)},
        "variants": compare([r for r in rows if r["status"] == "ok"]) + [r for r in rows if r["status"] != "ok"],
        "static_ablation": abl, "exports": exports, "total_seconds": round(time.time() - t_start, 1),
    }
    (ART / "results.json").write_text(json.dumps(results, indent=1, default=str))
    print(f"wrote artifacts/results.json in {results['total_seconds']}s")


if __name__ == "__main__":
    main()
