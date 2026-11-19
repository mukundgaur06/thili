"""Measure one model's peak memory in a fresh process.

    python -m quantlab.memprobe artifacts/models/static_int8.ts --batch 64

Prints one JSON line: resident memory after importing torch, after loading the model, and the
peak while running 20 batches, all in MB, plus the growth over the idle baseline.
"""

from __future__ import annotations

import argparse
import json
import sys


def peak_rss_mb() -> float:
    if sys.platform == "win32":
        import psutil
        return psutil.Process().memory_info().peak_wset / 2**20
    import resource
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / 2**20 if sys.platform == "darwin" else kb / 1024     # macOS reports bytes


def rss_mb() -> float:
    import psutil
    return psutil.Process().memory_info().rss / 2**20


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    import warnings
    warnings.filterwarnings("ignore")
    import torch
    torch.set_num_threads(a.threads)
    x = torch.randn(a.batch, 1, 28, 28).to(getattr(torch, a.dtype))
    idle = rss_mb()
    idle_peak = peak_rss_mb()
    model = torch.jit.load(a.path, map_location="cpu").eval()
    loaded = rss_mb()
    with torch.no_grad():
        for _ in range(20):
            model(x)
    peak = peak_rss_mb()
    print(json.dumps({"idle_mb": round(idle, 1), "after_load_mb": round(loaded, 1), "peak_mb": round(peak, 1),
                      "load_growth_mb": round(loaded - idle, 2), "peak_growth_mb": round(peak - max(idle, idle_peak), 2),
                      "batch": a.batch}))


if __name__ == "__main__":
    main()
