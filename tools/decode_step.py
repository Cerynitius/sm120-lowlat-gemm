# SPDX-License-Identifier: Apache-2.0
"""Time the projections of a whole decode step in execution order, vendor GEMM vs lowlat_gemm variants.

    python tools/decode_step.py --step shapes/qwen3.8-flash-next-tp1-step.json [--kind bf16] [--tokens 1,2,4,8]
        [--plans plans.json] [--prefetch-max-mb 16]

Every layer gets its own weights (the step streams all of them from DRAM). Between the GEMMs, "act" is a small
kernel that rewrites the next GEMM's input (norms, activations) and "experts" reads a separate per-layer buffer of
experts_bytes (routed experts: DRAM traffic and L2 pollution). The time of a graph with only those ops is reported
too, so the GEMM share is the difference.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lowlat_gemm as L  # noqa: E402
from common import make_input, make_weight, weight_bytes  # noqa: E402


def build_sequence(spec: dict) -> list[tuple]:
    """[(op, name, layer)], op in {"gemm", "act", "experts"}."""
    seq = []
    for layer in range(spec["layers"]):
        mixer = spec["mixers"][layer % len(spec["mixers"])]
        for block in spec["layer"]:
            for op in spec["blocks"][mixer if block == "mixer" else block]:
                if op in ("act", "experts"):
                    seq.append((op, None, layer))
                else:
                    seq.append(("gemm", op, layer))
    for op in spec["tail"]:
        seq.append((op, None, spec["layers"]) if op in ("act", "experts") else ("gemm", op, spec["layers"]))
    return seq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True)
    ap.add_argument("--kind", default="bf16")
    ap.add_argument("--tokens", default="1,2,4,8")
    ap.add_argument("--plans")
    ap.add_argument("--prefetch-max-mb", type=float, default=16)
    ap.add_argument("--replays", type=int, default=20)
    ap.add_argument("--variants", default="vendor,lowlat,lowlat+pdl,lowlat+prefetch,lowlat+pdl+prefetch,"
                                          "routed+pdl+prefetch")
    args = ap.parse_args()

    spec_path = args.step
    spec = json.load(open(spec_path))
    shapes = {s["name"]: (s["n"], s["k"]) for s in
              json.load(open(os.path.join(os.path.dirname(spec_path), spec["shapes"])))["shapes"]}
    if args.plans:
        L.load_plans(args.plans)
    dev = torch.device("cuda", torch.cuda.current_device())
    kind = args.kind
    tokens = [int(t) for t in args.tokens.split(",")]
    seq = build_sequence(spec)
    gemms = [(name, layer) for op, name, layer in seq if op == "gemm"]

    need = sum(weight_bytes(kind, *shapes[name]) for name, _ in gemms)
    need += (spec["layers"] + 1) * spec["experts_bytes"]
    free, _ = torch.cuda.mem_get_info(dev)
    print(f"{torch.cuda.get_device_name(dev)}; kind={kind}; {len(gemms)} GEMMs per step, "
          f"{need / 2**30:.2f} GiB of weights + expert stand-ins; {free / 2**30:.1f} GiB free")
    if need + (4 << 30) > free:
        raise SystemExit("not enough free device memory (need the step's weights plus 4 GiB)")

    g = torch.Generator(device=dev).manual_seed(0)
    weights = {}
    for name, layer in gemms:
        weights[(name, layer)] = make_weight(kind, *shapes[name], g, dev)
    experts = [torch.empty(spec["experts_bytes"] // 2, device=dev, dtype=torch.bfloat16).normal_(generator=g)
               for _ in range(spec["layers"] + 1)]
    sink = torch.empty((), device=dev, dtype=torch.float32)
    max_m = max(tokens)
    elems = counters = 0
    for name in {n for n, _ in gemms}:
        e, c = L.workspace_need(kind, *shapes[name], max_m, dev)
        elems, counters = max(elems, e), max(counters, c)
    L.reserve_workspace(dev, max(elems, 1 << 20), max(counters, 4096))

    # the GEMM that runs after each GEMM (prefetch target)
    nxt = {gemms[i]: weights[gemms[(i + 1) % len(gemms)]][0] for i in range(len(gemms))}
    pf_cap = int(args.prefetch_max_mb * (1 << 20))
    out_dt = torch.float16 if kind == "f16" or kind.endswith("_f16") else torch.bfloat16

    for m in tokens:
        xs, srcs, outs = {}, {}, {}
        for name in {n for n, _ in gemms}:
            n, k = shapes[name]
            x, x_kw = make_input(kind, m, k, dev)
            xs[name] = (x, x_kw)
            srcs[name] = x.clone()
            outs[name] = torch.empty(m, n, device=dev, dtype=out_dt)

        def run(variant: str) -> None:
            pending_act = False
            for op, name, layer in seq:
                if op == "act":
                    pending_act = True
                    continue
                if op == "experts":
                    torch.sum(experts[layer], dim=0, dtype=torch.float32, out=sink)
                    continue
                x, x_kw = xs[name]
                if pending_act:  # the producer of this GEMM's input
                    (x.view(torch.uint8) if x.element_size() == 1 else x).copy_(
                        srcs[name].view(torch.uint8) if x.element_size() == 1 else srcs[name])
                    pending_act = False
                if variant == "none":
                    continue
                w, w_kw = weights[(name, layer)]
                n, k = shapes[name]
                vendor = variant == "vendor" or (variant.startswith("routed") and not L.prefer_lowlat(kind, n, k, m))
                if vendor:
                    if kind not in ("bf16", "f16"):
                        raise SystemExit("vendor variant only for bf16 / f16 here")
                    torch.matmul(x, w.t(), out=outs[name])
                elif "prefetch" in variant:
                    nw = nxt[(name, layer)]
                    L.gemm(x, w, out=outs[name], prefetch=nw,
                           prefetch_bytes=min(pf_cap, nw.numel() * nw.element_size()), **w_kw, **x_kw)
                else:
                    L.gemm(x, w, out=outs[name], **w_kw, **x_kw)

        def timed(variant: str) -> float:
            L.set_pdl("pdl" in variant)
            try:
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    run(variant)
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run(variant)
            finally:
                L.set_pdl(False)
            graph.replay()
            torch.cuda.synchronize()
            ts = []
            for _ in range(args.replays):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                graph.replay()
                b.record()
                b.synchronize()
                ts.append(a.elapsed_time(b))
            del graph
            ts.sort()
            return ts[len(ts) // 2]

        base = timed("none")
        print(f"M={m}: step without GEMMs {base:.3f} ms")
        ref = None
        for v in args.variants.split(","):
            if kind not in ("bf16", "f16") and (v == "vendor" or v.startswith("routed")):
                continue
            t = timed(v)
            ref = t - base if v == "vendor" else ref
            rel = f"  vendor / this = {ref / (t - base):.3f}x" if ref and v != "vendor" else ""
            print(f"  {v:22s} step {t:7.3f} ms   GEMMs {t - base:7.3f} ms{rel}", flush=True)
        del xs, srcs, outs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
