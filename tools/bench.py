# SPDX-License-Identifier: Apache-2.0
"""Compare lowlat_gemm with torch.nn.functional.linear (cuBLAS) on a shape list.

    python tools/bench.py --shapes shapes/qwen3.8-flash-next-tp1.json [--tokens 1,2,3,4,8,16] [--plans plan.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lowlat_gemm  # noqa: E402
from timing import ColdWeights, time_per_call  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", required=True)
    ap.add_argument("--tokens", default="1,2,3,4,6,8,16")
    ap.add_argument("--plans", help="plan file from tools/tune.py (default: built-in plans)")
    ap.add_argument("--peak-gbs", type=float, default=1792.0, help="DRAM bandwidth for the %% column")
    ap.add_argument("--json", help="write results here")
    args = ap.parse_args()

    if args.plans:
        lowlat_gemm.load_plans(args.plans)
    device = torch.device("cuda", torch.cuda.current_device())
    tokens = [int(t) for t in args.tokens.split(",")]
    shapes = json.load(open(args.shapes))["shapes"]
    rows = []
    print(f"{torch.cuda.get_device_name(device)}; times are us per call, cold weights, CUDA graph replay")
    print(f"{'shape':28s} {'N':>7s} {'K':>6s} {'M':>3s} {'cuBLAS':>8s} {'lowlat':>8s} {'speedup':>7s} {'%peak':>6s}")
    for s in shapes:
        n, k = s["n"], s["k"]
        cold = ColdWeights(n, k, device)
        for m in tokens:
            x = torch.randn(m, k, device=device).to(torch.bfloat16)
            out = torch.empty(m, n, device=device, dtype=torch.bfloat16)
            t_ref = time_per_call(lambda w: torch.nn.functional.linear(x, w), cold)
            t_llg = time_per_call(lambda w: lowlat_gemm.gemm(x, w, out=out), cold)
            pct = cold.nbytes / (t_llg * 1e-6) / (args.peak_gbs * 1e9) * 100
            print(f"{s['name'][:28]:28s} {n:7d} {k:6d} {m:3d} {t_ref:8.2f} {t_llg:8.2f} {t_ref / t_llg:7.2f} {pct:6.1f}")
            rows.append({"name": s["name"], "n": n, "k": k, "m": m, "cublas_us": t_ref, "lowlat_us": t_llg})
        del cold
        torch.cuda.empty_cache()
    if args.json:
        json.dump({"device": torch.cuda.get_device_name(device), "results": rows}, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
