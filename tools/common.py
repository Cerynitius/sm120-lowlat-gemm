# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for tools/: operands per precision and cold-weight CUDA-graph timing.

Every timed call reads a different weight copy and the copies together exceed the L2 cache several times, so each
call streams its weight from DRAM, as in a decode step."""

from __future__ import annotations

import torch

FP8 = torch.float8_e4m3fn
FP4 = getattr(torch, "float4_e2m1fn_x2", torch.uint8)
MIN_ROTATION_BYTES = 512 << 20  # 4x the 128 MB L2 of GB202


def make_weight(kind: str, n: int, k: int, g: torch.Generator, device) -> tuple[torch.Tensor, dict]:
    """One weight of this precision plus its gemm() keyword arguments."""
    if kind in ("bf16", "f16"):
        dt = torch.bfloat16 if kind == "bf16" else torch.float16
        return (torch.randn(n, k, device=device, generator=g) / k**0.5).to(dt), {}
    if kind == "fp8" or kind.startswith("w8a16"):
        return (torch.randn(n, k, device=device, generator=g) * 4).to(FP8), {"w_scale": torch.ones(n, device=device)}
    if kind == "int8":
        return torch.randint(-128, 128, (n, k), device=device, generator=g, dtype=torch.int8), {
            "w_scale": torch.ones(n, device=device)}
    if kind == "nvfp4" or kind.startswith("w4a16"):
        w = torch.randint(0, 256, (n, k // 2), device=device, generator=g, dtype=torch.uint8).view(FP4)
        sf = torch.randint(0x18, 0x48, (n, k // 16), device=device, generator=g, dtype=torch.uint8).view(FP8)
        return w, {"w_block_scale": sf}
    raise ValueError(kind)


def make_input(kind: str, m: int, k: int, device) -> tuple[torch.Tensor, dict]:
    g = torch.Generator(device=device).manual_seed(1234)
    if kind == "bf16" or kind.endswith("_bf16"):
        return torch.randn(m, k, device=device, generator=g).to(torch.bfloat16), {}
    if kind == "f16" or kind.endswith("_f16"):
        return torch.randn(m, k, device=device, generator=g).to(torch.float16), {}
    if kind == "fp8":
        return (torch.randn(m, k, device=device, generator=g) * 4).to(FP8), {"x_scale": torch.ones(m, device=device)}
    if kind == "int8":
        return torch.randint(-128, 128, (m, k), device=device, generator=g, dtype=torch.int8), {
            "x_scale": torch.ones(m, device=device)}
    if kind == "nvfp4":
        x = torch.randint(0, 256, (m, k // 2), device=device, generator=g, dtype=torch.uint8).view(FP4)
        sf = torch.randint(0x18, 0x48, (m, k // 16), device=device, generator=g, dtype=torch.uint8).view(FP8)
        return x, {"x_block_scale": sf}
    raise ValueError(kind)


def weight_bytes(kind: str, n: int, k: int) -> int:
    bits = {"bf16": 16, "f16": 16, "fp8": 8, "int8": 8, "nvfp4": 4}.get(kind, 8 if kind.startswith("w8") else 4)
    scale = n * (k // 16) if (kind == "nvfp4" or kind.startswith("w4a16")) else 0
    return n * k * bits // 8 + scale


class ColdWeights:
    def __init__(self, kind: str, n: int, k: int, device, max_copies: int = 4096):
        self.nbytes = weight_bytes(kind, n, k)
        copies = max(1, min(max_copies, -(-MIN_ROTATION_BYTES // self.nbytes)))
        g = torch.Generator(device=device).manual_seed(0)
        self.items = [make_weight(kind, n, k, g, device) for _ in range(copies)]

    def __len__(self) -> int:
        return len(self.items)


def producer(x: torch.Tensor):
    """A small kernel that rewrites x before each GEMM (as a norm or activation would in a decode step), so the
    timing includes the dependency on a preceding non-GEMM kernel. Returns a callable, or None for back-to-back."""
    xb = x.view(torch.uint8) if x.element_size() == 1 else x
    src = xb.clone()
    return lambda: xb.copy_(src)


def time_per_call(fn, weights: ColdWeights, calls: int | None = None, replays: int = 20, before=None) -> float:
    """Median microseconds per fn(w, w_kwargs, next_w) call, replaying a CUDA graph over the rotating weights
    (next_w: the weight of the following call).

    With `before`, every call is preceded by before() and the time of a graph of before() calls alone is
    subtracted, so the result is the GEMM's share of a producer -> GEMM pair."""
    calls = calls or max(len(weights), 64)
    t = _time_graph(fn, weights, calls, replays, before)
    if before is None:
        return t
    return t - _time_graph(None, weights, calls, replays, before)


def _time_graph(fn, weights: ColdWeights, calls: int, replays: int, before) -> float:
    seq = [weights.items[i % len(weights)] for i in range(calls)]
    nxt = [weights.items[(i + 1) % len(weights)][0] for i in range(calls)]
    pos = [0]

    def step(w, kw):
        if before is not None:
            before()
        if fn is not None:
            fn(w, kw, nxt[pos[0] % calls])
        pos[0] += 1

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for w, kw in seq[: min(len(seq), 8)]:
            step(w, kw)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    pos[0] = 0
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for w, kw in seq:
            step(w, kw)
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(replays):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000.0 / calls)
    del graph
    times.sort()
    return times[len(times) // 2]


def baseline(kind: str, x: torch.Tensor, x_kw: dict):
    """A vendor-library reference for the same product, or None (W8A16 / W4A16 / NVFP4 have none here)."""
    if kind in ("bf16", "f16"):
        return lambda w, kw, nxt=None: torch.nn.functional.linear(x, w)
    if kind == "fp8" and hasattr(torch, "_scaled_mm"):
        one = torch.ones(1, device=x.device)
        return lambda w, kw, nxt=None: torch._scaled_mm(x, w.t(), one, one, out_dtype=torch.bfloat16)
    if kind == "int8" and hasattr(torch, "_int_mm") and x.shape[0] > 16:
        return lambda w, kw, nxt=None: torch._int_mm(x, w.t())
    return None
