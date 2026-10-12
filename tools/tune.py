# SPDX-License-Identifier: Apache-2.0
"""Search plans for a shape list and write a plan file (load with LLG_PLAN_FILE or lowlat_gemm.load_plans).

    python tools/tune.py --shapes shapes/qwen3.8-flash-next-tp1.json --out plans.json [--tokens 1,2,3,4,6,8,16]

Per (N, K) the summation order (lanes_per_row, ksplit) is chosen once for all M, so results stay bit-identical
across token counts; only the tile (rows, unroll, warps) is chosen per M.
Stage 1 ranks every (lanes_per_row, ksplit) by its best tile at the --rank-tokens counts; stage 2 searches all
tiles for every token count on the --finalists best orders and keeps the order with the lowest summed time.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lowlat_gemm  # noqa: E402
from lowlat_gemm import Plan, Tile  # noqa: E402
from timing import ColdWeights, time_per_call  # noqa: E402

TILES = [Tile(r, u, wp) for (r, u), wp in itertools.product(sorted(lowlat_gemm._RU_PAIRS), (2, 4, 8))]
KSPLITS = (1, 2, 3, 4, 6, 8, 12, 16)


def orders(k: int):
    kv = k // 8
    for lanes in (32, 16, 8):
        if kv < lanes and lanes != 8:
            continue
        for ks in KSPLITS:
            if ks == 1 or kv // ks >= lanes:
                yield lanes, ks


def best_tile(x, out, cold, lanes, ks, m):
    best = (float("inf"), None)
    for tile in TILES:
        plan = Plan(lanes_per_row=lanes, ksplit=ks, default_tile=tile)
        t = time_per_call(lambda w: lowlat_gemm.gemm(x, w, out=out, plan=plan), cold, replays=7)
        best = min(best, (t, tile), key=lambda b: b[0])
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", default="1,2,3,4,6,8,16")
    ap.add_argument("--rank-tokens", default="1,4")
    ap.add_argument("--finalists", type=int, default=3)
    args = ap.parse_args()

    device = torch.device("cuda", torch.cuda.current_device())
    tokens = [int(t) for t in args.tokens.split(",")]
    rank_tokens = [int(t) for t in args.rank_tokens.split(",")]
    shapes = json.load(open(args.shapes))["shapes"]
    max_n = max(s["n"] for s in shapes)
    lowlat_gemm.reserve_workspace(device, max(KSPLITS) * max(tokens + rank_tokens) * max_n, max_n)
    result = {"device": torch.cuda.get_device_name(device), "shapes": {}}
    for s in shapes:
        n, k = s["n"], s["k"]
        key = f"{n}x{k}"
        if key in result["shapes"]:
            continue
        cold = ColdWeights(n, k, device)
        xs = {m: torch.randn(m, k, device=device).to(torch.bfloat16) for m in tokens + rank_tokens}
        outs = {m: torch.empty(m, n, device=device, dtype=torch.bfloat16) for m in xs}
        ranked = []
        for lanes, ks in orders(k):
            total = sum(best_tile(xs[m], outs[m], cold, lanes, ks, m)[0] for m in rank_tokens)
            ranked.append((total, lanes, ks))
        ranked.sort()
        best = None
        for _, lanes, ks in ranked[: args.finalists]:
            tiles, total = {}, 0.0
            for m in tokens:
                t, tile = best_tile(xs[m], outs[m], cold, lanes, ks, m)
                tiles[m], total = tile, total + t
            if best is None or total < best[0]:
                best = (total, Plan(lanes_per_row=lanes, ksplit=ks, tiles=tiles, default_tile=tiles[max(tokens)]))
        plan = best[1]
        cublas = {}
        for m in tokens:
            x = xs[m]
            cublas[m] = time_per_call(lambda w: torch.nn.functional.linear(x, w), cold, replays=7)
        entry = plan.to_json()
        entry["us"] = {
            str(m): round(time_per_call(lambda w: lowlat_gemm.gemm(xs[m], w, out=outs[m], plan=plan), cold), 3)
            for m in tokens
        }
        entry["cublas_us"] = {str(m): round(t, 3) for m, t in cublas.items()}
        result["shapes"][key] = entry
        print(f"{s['name']:28s} {key:>13s} lanes={plan.lanes_per_row:2d} ksplit={plan.ksplit:2d} "
              + " ".join(f"M{m}:{entry['us'][str(m)]:.1f}/{cublas[m]:.1f}" for m in tokens), flush=True)
        del cold, xs, outs
        torch.cuda.empty_cache()
        json.dump(result, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
