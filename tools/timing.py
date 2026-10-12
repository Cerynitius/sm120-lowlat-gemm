# SPDX-License-Identifier: Apache-2.0
"""CUDA-graph timing with cold weights: every call in the graph reads a different weight copy, and the copies
together exceed the L2 cache several times, so each call streams its weight from DRAM (as in a decode step)."""

from __future__ import annotations

import torch

MIN_ROTATION_BYTES = 512 << 20  # 4x the 128 MB L2 of GB202


class ColdWeights:
    def __init__(self, n: int, k: int, device: torch.device, seed: int = 0, max_copies: int = 4096):
        nbytes = n * k * 2
        copies = max(1, min(max_copies, -(-MIN_ROTATION_BYTES // nbytes)))
        g = torch.Generator(device=device).manual_seed(seed)
        self.w = [(torch.randn(n, k, device=device, generator=g) / k**0.5).to(torch.bfloat16) for _ in range(copies)]
        self.nbytes = nbytes

    def __len__(self) -> int:
        return len(self.w)


def time_per_call(fn, weights: ColdWeights, calls: int | None = None, replays: int = 20) -> float:
    """Median microseconds per fn(w) call, measured as CUDA graph replays over the rotating weights."""
    calls = calls or max(len(weights), 64)
    seq = [weights.w[i % len(weights)] for i in range(calls)]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for w in seq[: min(len(seq), 8)]:
            fn(w)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for w in seq:
            fn(w)
    g.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(replays):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        g.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000.0 / calls)
    del g
    times.sort()
    return times[len(times) // 2]
