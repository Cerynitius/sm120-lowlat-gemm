# SPDX-License-Identifier: Apache-2.0
"""Time lowlat_gemm (and a vendor baseline where one exists) on a shape list.

    python tools/bench.py --shapes shapes/qwen3.8-flash-next-tp1.json --kinds bf16,fp8,nvfp4 \
        [--tokens 1,2,4,8,16,64,256] [--plans plans.json] [--json out.json]

Weight-streaming cases are reported against --peak-gbs (DRAM bandwidth); large-M cases against --peak-tflops
for the precision (dense tensor throughput), whichever bound is closer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lowlat_gemm  # noqa: E402
from common import ColdWeights, baseline, make_input, producer, time_per_call  # noqa: E402

# nominal dense tensor throughput of the RTX PRO 6000 Blackwell (override with --peak-tflops)
PEAK_TFLOPS = {"bf16": 500, "f16": 500, "fp8": 1000, "int8": 1000, "nvfp4": 2000,
               "w8a16_bf16": 500, "w8a16_f16": 500, "w4a16_bf16": 500, "w4a16_f16": 500}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", required=True)
    ap.add_argument("--kinds", default="bf16")
    ap.add_argument("--tokens", default="1,2,4,8,16,64,256")
    ap.add_argument("--plans")
    ap.add_argument("--peak-gbs", type=float, default=1792.0)
    ap.add_argument("--peak-tflops", type=float, help="override the nominal per-precision tensor TFLOPS")
    ap.add_argument("--json")
    ap.add_argument("--pdl", action="store_true", help="programmatic dependent launch (lowlat_gemm.set_pdl)")
    ap.add_argument("--prefetch-next", action="store_true",
                    help="each lowlat_gemm call prefetches the next call's weight into L2 (gemm(prefetch=...))")
    ap.add_argument("--prefetch-max-mb", type=float, default=16)
    ap.add_argument("--between", choices=("none", "write"), default="none",
                    help="write: a small kernel rewrites x before every GEMM (its own time is subtracted)")
    args = ap.parse_args()

    if args.plans:
        lowlat_gemm.load_plans(args.plans)
    dev = torch.device("cuda", torch.cuda.current_device())
    lowlat_gemm.set_pdl(args.pdl)
    tokens = [int(t) for t in args.tokens.split(",")]
    shapes = json.load(open(args.shapes))["shapes"]
    rows = []
    print(f"{torch.cuda.get_device_name(dev)}; us per call, cold weights, CUDA graph replay; "
          f"pdl={args.pdl} between={args.between} prefetch_next={args.prefetch_next}")
    print(f"{'kind':11s} {'shape':26s} {'N':>7s} {'K':>6s} {'M':>5s} {'vendor':>8s} {'lowlat':>8s} {'x':>5s} {'%bound':>6s}")
    for kind in args.kinds.split(","):
        for s in shapes:
            n, k = s["n"], s["k"]
            if k % lowlat_gemm.KINDS[kind][2]:
                continue
            cold = ColdWeights(kind, n, k, dev)
            for m in tokens:
                x, x_kw = make_input(kind, m, k, dev)
                out = torch.empty(m, n, device=dev, dtype=torch.bfloat16 if kind != "f16" and not kind.endswith("_f16") else torch.float16)
                before = producer(x) if args.between == "write" else None
                pf_cap = int(args.prefetch_max_mb * (1 << 20))

                def call(w, kw, nxt):
                    if args.prefetch_next:
                        return lowlat_gemm.gemm(x, w, out=out, prefetch=nxt, prefetch_bytes=pf_cap, **kw, **x_kw)
                    return lowlat_gemm.gemm(x, w, out=out, **kw, **x_kw)

                t = time_per_call(call, cold, before=before)
                base = baseline(kind, x, x_kw)
                tb = time_per_call(base, cold, before=before) if base is not None else float("nan")
                bw_us = cold.nbytes / (args.peak_gbs * 1e9) * 1e6
                fl_us = 2 * m * n * k / ((args.peak_tflops or PEAK_TFLOPS[kind]) * 1e12) * 1e6
                pct = max(bw_us, fl_us) / t * 100
                print(f"{kind:11s} {s['name'][:26]:26s} {n:7d} {k:6d} {m:5d} {tb:8.2f} {t:8.2f} {tb / t:5.2f} {pct:6.1f}",
                      flush=True)
                rows.append({"kind": kind, "name": s["name"], "n": n, "k": k, "m": m, "vendor_us": tb, "lowlat_us": t})
            del cold
            torch.cuda.empty_cache()
    if args.json:
        json.dump({"device": torch.cuda.get_device_name(dev), "results": rows}, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
