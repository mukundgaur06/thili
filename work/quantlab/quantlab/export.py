"""Export model variants, and test every export before advertising it.

* **TorchScript** (``torch.jit.trace``) for every variant, int8 ones included. The test:
  reload the file and compare its outputs with the in-memory model on real test images.
* **ONNX** for the fp32 model (and, where the exporter accepts them, quantized variants). The
  test: run the file with ONNX Runtime and compare logits and predictions.

A format is listed as available only if its export *and* its parity test passed; failures are
recorded with the error message instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn


def export_torchscript(model: nn.Module, path: Path, example: torch.Tensor) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with torch.no_grad():
            traced = torch.jit.trace(model.eval(), example, check_trace=False)
        torch.jit.save(traced, str(path))
        reloaded = torch.jit.load(str(path))
        with torch.no_grad():
            a, b = model(example).float(), reloaded(example).float()
        diff = float((a - b).abs().max())
        same = bool((a.argmax(1) == b.argmax(1)).all())
        return {"format": "torchscript", "path": str(path), "bytes": path.stat().st_size,
                "max_abs_diff": diff, "same_predictions": same, "ok": same and diff < 1e-3}
    except Exception as e:  # noqa: BLE001
        return {"format": "torchscript", "path": str(path), "ok": False, "error": f"{type(e).__name__}: {e}"[:300]}


def export_onnx(model: nn.Module, path: Path, example: torch.Tensor, check_images: torch.Tensor | None = None,
                quantized: bool = False) -> dict:
    """Export to ONNX and verify it with ONNX Runtime.

    fp32 must match PyTorch to 1e-3 with identical predictions. An int8 graph runs on ONNX
    Runtime's own integer kernels, which may round a value one step differently from PyTorch's,
    so for it we require >= 99.5% identical predictions over ``check_images`` (the whole test
    set) and report the largest logit difference in units of the output quantization step.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.onnx.export(model.eval(), (example,), str(path), input_names=["image"], output_names=["logits"],
                          dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17,
                          dynamo=False)
    except Exception as e:  # noqa: BLE001
        return {"format": "onnx", "path": str(path), "ok": False,
                "error": f"export failed: {type(e).__name__}: {str(e).splitlines()[0]}"[:300]}
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        x = check_images if check_images is not None else example
        got, ref = [], []
        with torch.no_grad():
            for i in range(0, len(x), 1000):
                got.append(torch.from_numpy(sess.run(None, {"image": x[i:i + 1000].numpy()})[0]))
                ref.append(model(x[i:i + 1000]).float())
        got, ref = torch.cat(got), torch.cat(ref)
        diff = float((got - ref).abs().max())
        agree = float((got.argmax(1) == ref.argmax(1)).float().mean())
        out = {"format": "onnx", "path": str(path), "bytes": path.stat().st_size, "max_abs_diff": diff,
               "prediction_agreement": agree, "checked_images": len(x), "runtime": f"onnxruntime {ort.__version__}"}
        if quantized:
            step = float(model.fc2.scale) if hasattr(model.fc2, "scale") else float("nan")
            out.update({"output_step": step, "max_diff_in_steps": diff / step, "ok": agree >= 0.995 and diff <= 2.01 * step})
        else:
            out["ok"] = agree == 1.0 and diff < 1e-3
        return out
    except Exception as e:  # noqa: BLE001
        return {"format": "onnx", "path": str(path), "ok": False,
                "error": f"onnxruntime check failed: {type(e).__name__}: {str(e).splitlines()[0]}"[:300]}


def write_manifest(path: Path, entries: Dict[str, list], meta: dict) -> None:
    path.write_text(json.dumps({"meta": meta, "exports": entries}, indent=1))
