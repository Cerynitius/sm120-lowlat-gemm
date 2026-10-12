# SPDX-License-Identifier: Apache-2.0
"""Measure plans for a shape list and write a plan file (load with LLG_PLAN_FILE or lowlat_gemm.load_plans).

    python tools/tune.py --shapes shapes/qwen3.8-flash-next-tp1.json --kinds bf16,nvfp4 --out plans.json \
        [--tokens 1,2,3,4,8,16,32,64,128,256]

Tensor-core: every (config, orientation, stages, K split) is timed per token count and the fastest kept.
Skinny (W16A16): the summation order (lanes_per_row, K split) is chosen once per (N, K) -- so results stay
bit-identical across token counts -- and the tile per token count; skinny serves the token counts where it beats
the tensor-core choice, up to the first count where it does not.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lowlat_gemm as L  # noqa: E402
from common import ColdWeights, make_input, time_per_call  # noqa: E402
from lowlat_gemm import Plan, SkinnyOrder, SkinnyTile, TcChoice  # noqa: E402

SKINNY_TILES = [SkinnyTile(r, u, w) for (r, u), w in itertools.product([(1, 2), (1, 4), (1, 8), (2, 2), (2, 4), (4, 2)], (2, 4, 8))]
KSPLITS = (1, 2, 4, 8, 16)


def tc_candidates(kind: str, m: int, n: int, k: int):
    nk = -(-k // L.KINDS[kind][1])
    for cfg in range(L.NUM_CFGS):
        top = L.max_stages(kind, cfg)
        if top < 2:
            continue
        for swap in ((True,) if kind in L._MIXED else (True, False)):
            bp, bq, _, _ = L.tc_info(kind, cfg)
            p, q = (n, m) if swap else (m, n)
            if bq > 2 * q and bq > 8:  # tile far wider than the problem: skip
                continue
            for stages in sorted({2, min(3, top), min(4, top), top}):
                for ks in KSPLITS:
                    if ks > 1 and nk // ks < 2:
                        continue
                    yield TcChoice(cfg=cfg, swap=swap, stages=stages, ksplit=ks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", required=True)
    ap.add_argument("--kinds", default="bf16")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", default="1,2,3,4,8,16,32,64,128,256")
    ap.add_argument("--replays", type=int, default=7)
    args = ap.parse_args()

    dev = torch.device("cuda", torch.cuda.current_device())
    tokens = [int(t) for t in args.tokens.split(",")]
    shapes = json.load(open(args.shapes))["shapes"]
    max_n = max(s["n"] for s in shapes)
    L.reserve_workspace(dev, max(KSPLITS) * max(tokens) * max_n, max_n)
    plans: dict[tuple[str, int, int], Plan] = {}
    for kind in args.kinds.split(","):
        for s in shapes:
            n, k = s["n"], s["k"]
            if (kind, n, k) in plans or k % L.KINDS[kind][2]:
                continue
            cold = ColdWeights(kind, n, k, dev)
            plan = Plan()
            tc_us = {}
            for m in tokens:
                x, x_kw = make_input(kind, m, k, dev)
                out = torch.empty(m, n, device=dev, dtype=torch.float16 if kind == "f16" or kind.endswith("_f16") else torch.bfloat16)
                best = None
                for c in tc_candidates(kind, m, n, k):
                    t = time_per_call(lambda w, kw: L.gemm(x, w, out=out, plan=Plan(tc={1: c}), **kw, **x_kw), cold,
                                      replays=args.replays)
                    if best is None or t < best[0]:
                        best = (t, c)
                tc_us[m], plan.tc[m] = best
            if kind in ("bf16", "f16"):
                skinny_ms = [m for m in tokens if m <= L.SKINNY_MAX_TOKENS]
                best_order = None
                kv = k // 8
                for lanes, ks in itertools.product((32, 16, 8), (1, 2, 4, 8, 16)):
                    if ks > 1 and kv // ks < lanes:
                        continue
                    tiles, total = {}, 0.0
                    for m in skinny_ms:
                        x, _ = make_input(kind, m, k, dev)
                        out = torch.empty(m, n, device=dev, dtype=x.dtype)
                        bt = None
                        for tile in SKINNY_TILES:
                            order = SkinnyOrder(lanes_per_row=lanes, ksplit=ks, default_tile=tile)
                            pl = Plan(skinny=order, skinny_max_m=L.SKINNY_MAX_TOKENS)
                            t = time_per_call(lambda w, kw: L.gemm(x, w, out=out, plan=pl), cold, replays=args.replays)
                            if bt is None or t < bt[0]:
                                bt = (t, tile)
                        tiles[m] = bt
                        total += bt[0]
                    if best_order is None or total < best_order[0]:
                        best_order = (total, lanes, ks, tiles)
                _, lanes, ks, tiles = best_order
                plan.skinny = SkinnyOrder(lanes_per_row=lanes, ksplit=ks, tiles={m: t for m, (_, t) in tiles.items()})
                plan.skinny_max_m = 0
                for m in skinny_ms:
                    if tiles[m][0] < tc_us[m]:
                        plan.skinny_max_m = m
                    else:
                        break
            plans[(kind, n, k)] = plan
            summary = " ".join(f"M{m}:{tc_us[m]:.1f}" for m in tokens)
            print(f"{kind:11s} {s['name'][:26]:26s} {n}x{k}  tc[{summary}]  skinny<= {plan.skinny_max_m}", flush=True)
            L.save_plans(args.out, plans, torch.cuda.get_device_name(dev))
            del cold
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
