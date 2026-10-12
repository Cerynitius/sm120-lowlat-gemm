# SPDX-License-Identifier: Apache-2.0
"""Mark, in a plan file, the token counts where bench.py measured the vendor library as faster.

    python tools/prefer_vendor.py --plans plans.json --bench bench.json [--margin 0.02]

lowlat_gemm.prefer_lowlat() (and the vLLM plugin) then leaves those token counts to the vendor library.
"""

from __future__ import annotations

import argparse
import json
import math


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--margin", type=float, default=0.02, help="vendor must be faster by this fraction")
    args = ap.parse_args()

    data = json.load(open(args.plans))
    rows = json.load(open(args.bench))["results"]
    marked = 0
    for key, plan in data["plans"].items():
        kind, shape = key.split(":")
        n, k = (int(v) for v in shape.split("x"))
        vendor = set()
        for r in rows:
            if (r["kind"], r["n"], r["k"]) != (kind, n, k) or math.isnan(r["vendor_us"]):
                continue
            if r["vendor_us"] * (1 + args.margin) < r["lowlat_us"]:
                vendor.add(r["m"])
        if vendor:
            plan["prefer_vendor"] = sorted(vendor)
            marked += len(vendor)
        else:
            plan.pop("prefer_vendor", None)
    json.dump(data, open(args.plans, "w"), indent=1)
    print(f"{marked} (plan, M) entries left to the vendor library")


if __name__ == "__main__":
    main()
