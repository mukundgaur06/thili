"""Size, latency, throughput and memory, measured the same way for every model variant.

* **Size**: bytes of the serialized ``state_dict`` (what you would ship), and of the
  TorchScript file.
* **Latency**: the same input tensor, the same thread count, ``warmup`` untimed runs, then
  ``iters`` timed runs; we report the median (robust to OS hiccups) and the 90th percentile.
* **Throughput**: images per second at the median latency.
* **Peak memory**: each model runs in a *fresh* Python process (``quantlab.memprobe``); we
  report how much its peak resident memory grew over an idle process that had already imported
  torch. In-process numbers would be polluted by everything else loaded in this process.
"""

from __future__ import annotations

import io
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent


def state_dict_bytes(model: nn.Module) -> int:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes


def weight_bytes_fp32(model: nn.Module) -> int:
    return sum(p.numel() * 4 for p in model.parameters())


@torch.no_grad()
def latency(model: nn.Module, batch_size: int, warmup: int = 10, iters: int = 50,
            dtype: torch.dtype = torch.float32, seed: int = 0) -> Dict[str, float]:
    model.eval()
    x = torch.randn(batch_size, 1, 28, 28, generator=torch.Generator().manual_seed(seed)).to(dtype)
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(iters):
        t = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    median = statistics.median(times)
    return {"batch_size": batch_size, "median_ms": median, "p90_ms": times[int(0.9 * (len(times) - 1))],
            "min_ms": times[0], "throughput_ips": batch_size / (median / 1000), "iters": iters, "warmup": warmup}


def interleaved_latency(models: Dict[str, nn.Module], dtypes: Dict[str, torch.dtype], batch_sizes=(1, 64),
                        rounds: int = 5, iters: Dict[int, int] | None = None, warmup: int = 10) -> Dict[str, dict]:
    """Time every model in turn, ``rounds`` times over, and pool the timings per model.

    Timing models one after another lets a burst of background load land on one model only.
    Interleaving spreads any such burst across all of them, so the *comparison* stays fair. We
    report the median of all pooled runs, and the lowest and highest per-round median as a
    measure of run-to-run spread.
    """
    iters = iters or {1: 100, 64: 12}
    gen = torch.Generator().manual_seed(0)
    inputs = {b: torch.randn(b, 1, 28, 28, generator=gen) for b in batch_sizes}
    pooled = {n: {b: [] for b in batch_sizes} for n in models}
    rounds_med = {n: {b: [] for b in batch_sizes} for n in models}
    with torch.no_grad():
        for n, m in models.items():
            m.eval()
            for b in batch_sizes:
                for _ in range(warmup):
                    m(inputs[b].to(dtypes.get(n, torch.float32)))
        for _ in range(rounds):
            for n, m in models.items():
                for b in batch_sizes:
                    x = inputs[b].to(dtypes.get(n, torch.float32))
                    ts = []
                    for _ in range(iters.get(b, 20)):
                        t = time.perf_counter()
                        m(x)
                        ts.append((time.perf_counter() - t) * 1000)
                    pooled[n][b] += ts
                    rounds_med[n][b].append(statistics.median(ts))
    out = {}
    for n in models:
        out[n] = {}
        for b in batch_sizes:
            ts = sorted(pooled[n][b])
            med = statistics.median(ts)
            out[n][f"bs{b}"] = {"batch_size": b, "median_ms": med, "p90_ms": ts[int(0.9 * (len(ts) - 1))],
                                "min_ms": ts[0], "throughput_ips": b / (med / 1000), "iters": len(ts),
                                "rounds": rounds, "round_median_range_ms": [min(rounds_med[n][b]), max(rounds_med[n][b])],
                                "warmup": warmup}
    return out


def latency_suite(model: nn.Module, batch_sizes=(1, 64), iters: Dict[int, int] | None = None,
                  dtype: torch.dtype = torch.float32) -> Dict[str, dict]:
    iters = iters or {1: 300, 64: 40}
    return {f"bs{b}": latency(model, b, iters=iters.get(b, 50), dtype=dtype) for b in batch_sizes}


def peak_memory(script_path: Path, batch_size: int = 64, dtype: str = "float32", threads: int = 4) -> dict:
    """Run ``quantlab.memprobe`` in a fresh interpreter and parse its JSON line."""
    cmd = [sys.executable, "-m", "quantlab.memprobe", str(script_path), "--batch", str(batch_size),
           "--dtype", dtype, "--threads", str(threads)]
    env = {**os.environ, "PYTHONWARNINGS": "ignore"}
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300, env=env)
    for line in out.stdout.splitlines()[::-1]:
        if line.startswith("{"):
            return json.loads(line)
    return {"error": (out.stderr or out.stdout).strip().splitlines()[-1][:300] if (out.stderr or out.stdout) else "no output"}


def environment() -> dict:
    return {"torch": torch.__version__, "python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(), "cpu_count": os.cpu_count(),
            "threads": torch.get_num_threads(), "quantized_engine": torch.backends.quantized.engine,
            "supported_engines": list(torch.backends.quantized.supported_engines)}


def compare(rows: List[dict], baseline: str = "fp32") -> List[dict]:
    """Add compression ratio, accuracy drop and speed-up relative to the baseline row."""
    base = next(r for r in rows if r["name"] == baseline)
    out = []
    for r in rows:
        r = dict(r)
        if r.get("size_bytes"):
            r["compression"] = base["size_bytes"] / r["size_bytes"]
        if r.get("accuracy") is not None:
            r["accuracy_drop_pts"] = 100 * (base["accuracy"] - r["accuracy"])
        for key in ("bs1", "bs64"):
            if r.get("latency", {}).get(key) and base.get("latency", {}).get(key):
                r[f"speedup_{key}"] = base["latency"][key]["median_ms"] / r["latency"][key]["median_ms"]
        out.append(r)
    return out
