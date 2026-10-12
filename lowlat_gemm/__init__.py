# SPDX-License-Identifier: Apache-2.0
"""Low-latency skinny BF16 GEMM for SM120 (RTX PRO 6000 / RTX 50 series).

    y = lowlat_gemm.gemm(x, w)        # x [M, K] bf16, w [N, K] bf16 (nn.Linear layout), 1 <= M <= 16

Summation order depends only on (K, lanes_per_row, ksplit), which a plan fixes per (N, K). Rows of x therefore
get bit-identical results for every M and every per-M tile choice.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from . import _C  # noqa: F401  (registers torch.ops.lowlat_gemm)

__all__ = [
    "MAX_TOKENS",
    "Plan",
    "Tile",
    "default_plan",
    "eligible",
    "gemm",
    "get_plan",
    "load_plans",
    "reserve_workspace",
    "set_plan",
]

MAX_TOKENS = 16
_RU_PAIRS = {(1, 2), (1, 4), (1, 8), (2, 2), (2, 4), (4, 2)}
_LANES = (32, 16, 8)


@dataclass(frozen=True)
class Tile:
    """Per-M tiling. Does not change numerics."""

    rows: int = 1     # rows per row group
    unroll: int = 4   # 16-byte weight vectors in flight per row and lane
    warps: int = 4    # warps per block

    def __post_init__(self):
        if (self.rows, self.unroll) not in _RU_PAIRS:
            raise ValueError(f"unsupported (rows, unroll) {(self.rows, self.unroll)}; one of {sorted(_RU_PAIRS)}")
        if not 1 <= self.warps <= 8:
            raise ValueError(f"warps must be in [1, 8], got {self.warps}")


@dataclass
class Plan:
    """Per-(N, K) plan. lanes_per_row and ksplit fix the summation order; tiles map M -> Tile."""

    lanes_per_row: int = 32
    ksplit: int = 1
    tiles: dict[int, Tile] = field(default_factory=dict)
    default_tile: Tile = field(default_factory=Tile)

    def __post_init__(self):
        if self.lanes_per_row not in _LANES:
            raise ValueError(f"lanes_per_row must be one of {_LANES}, got {self.lanes_per_row}")
        if not 1 <= self.ksplit <= 64:
            raise ValueError(f"ksplit must be in [1, 64], got {self.ksplit}")

    def tile(self, m: int) -> Tile:
        t = self.tiles.get(m)
        if t is not None:
            return t
        smaller = [k for k in self.tiles if k <= m]
        return self.tiles[max(smaller)] if smaller else self.default_tile

    def to_json(self) -> dict:
        return {
            "lanes_per_row": self.lanes_per_row,
            "ksplit": self.ksplit,
            "default_tile": asdict(self.default_tile),
            "tiles": {str(m): asdict(t) for m, t in sorted(self.tiles.items())},
        }

    @classmethod
    def from_json(cls, d: dict) -> "Plan":
        return cls(
            lanes_per_row=int(d["lanes_per_row"]),
            ksplit=int(d["ksplit"]),
            default_tile=Tile(**d.get("default_tile", {})),
            tiles={int(m): Tile(**t) for m, t in d.get("tiles", {}).items()},
        )


def default_plan(n: int, k: int, sm_count: int) -> Plan:
    """Untuned starting point: enough independent row groups (and K splits) to keep every SM streaming."""
    kv = k // 8  # 16-byte vectors per row
    lanes = next((l for l in _LANES if kv >= 4 * l), _LANES[-1])
    rows = 2 if n >= 8192 else 1
    groups = -(-n // rows)
    want = 8 * sm_count
    ksplit = 1
    while groups * ksplit < want and kv // (ksplit * 2) >= 2 * lanes and ksplit < 16:
        ksplit *= 2
    tile = Tile(rows=rows, unroll=4 if rows <= 2 else 2, warps=4)
    return Plan(lanes_per_row=lanes, ksplit=ksplit, default_tile=tile)


# ---------------------------------------------------------------------------------------------------------------
# Plan registry: explicit set_plan() > plan files (LLG_PLAN_FILE, packaged plans/<device>.json) > default_plan().

_lock = threading.Lock()
_plans: dict[tuple[str, int, int], Plan] = {}
_loaded_devices: set[str] = set()
_sm_counts: dict[int, int] = {}


def _device_name(device: torch.device) -> str:
    return torch.cuda.get_device_name(device)


def _sm_count(device: torch.device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _sm_counts:
        _sm_counts[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _sm_counts[idx]


def _plan_key(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()


def load_plans(path: str | os.PathLike, device_name: str | None = None) -> int:
    """Load a plan file ({"device": ..., "shapes": {"NxK": plan}}). Returns the number of shapes loaded."""
    data = json.loads(Path(path).read_text())
    name = device_name or data.get("device")
    if not name:
        raise ValueError(f"{path}: no device name in the file and none given")
    with _lock:
        for key, d in data.get("shapes", {}).items():
            n, k = (int(v) for v in key.lower().split("x"))
            _plans[(name, n, k)] = Plan.from_json(d)
    return len(data.get("shapes", {}))


def _ensure_loaded(name: str) -> None:
    if name in _loaded_devices:
        return
    with _lock:
        if name in _loaded_devices:
            return
        _loaded_devices.add(name)
    packaged = Path(__file__).with_name("plans") / f"{_plan_key(name)}.json"
    if packaged.is_file():
        load_plans(packaged, name)
    env = os.environ.get("LLG_PLAN_FILE")
    if env:
        load_plans(env)


def set_plan(n: int, k: int, plan: Plan, device: torch.device | None = None) -> None:
    device = device or torch.device("cuda", torch.cuda.current_device())
    name = _device_name(device)
    _ensure_loaded(name)
    with _lock:
        _plans[(name, n, k)] = plan


def get_plan(n: int, k: int, device: torch.device | None = None) -> Plan:
    device = device or torch.device("cuda", torch.cuda.current_device())
    name = _device_name(device)
    _ensure_loaded(name)
    plan = _plans.get((name, n, k))
    if plan is None:
        plan = default_plan(n, k, _sm_count(device))
        with _lock:
            plan = _plans.setdefault((name, n, k), plan)
    return plan


# ---------------------------------------------------------------------------------------------------------------
# Split-K workspace: one fp32 partial-sum buffer and one zeroed counter array per device. Kernels on one stream
# reuse them back to back (each launch leaves its counters at zero). Grow them before CUDA graph capture.

_ws: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
_EMPTY: dict[int, torch.Tensor] = {}


def _columns(n: int, plan: Plan, tile: Tile) -> int:
    per_block = tile.warps * (32 // plan.lanes_per_row) * tile.rows
    return -(-n // per_block)


def reserve_workspace(device: torch.device, floats: int, counters: int) -> None:
    """Make sure the split-K buffers of `device` hold at least `floats` partials and `counters` counters."""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    ws, cnt = _ws.get(idx, (None, None))
    if ws is not None and ws.numel() >= floats and cnt.numel() >= counters and idx in _EMPTY:
        return
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "lowlat_gemm: split-K workspace must grow during CUDA graph capture; call reserve_workspace() "
            f"before capture (need {floats} floats, {counters} counters)"
        )
    if idx not in _EMPTY:
        _EMPTY[idx] = torch.empty(0, dtype=torch.float32, device=torch.device("cuda", idx))
    if ws is not None and ws.numel() >= floats and cnt.numel() >= counters:
        return
    floats = max(floats, ws.numel() if ws is not None else 0, 1 << 20)
    counters = max(counters, cnt.numel() if cnt is not None else 0, 1 << 12)
    dev = torch.device("cuda", idx)
    _ws[idx] = (
        torch.empty(floats, dtype=torch.float32, device=dev),
        torch.zeros(counters, dtype=torch.int32, device=dev),
    )


def _workspace(device: torch.device, plan: Plan, tile: Tile, m: int, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    idx = device.index
    if plan.ksplit == 1:
        if idx not in _EMPTY:
            _EMPTY[idx] = torch.empty(0, dtype=torch.float32, device=device)
        return _EMPTY[idx], _EMPTY[idx]
    reserve_workspace(device, plan.ksplit * m * n, _columns(n, plan, tile))
    return _ws[idx]


# ---------------------------------------------------------------------------------------------------------------


def eligible(x: torch.Tensor, w: torch.Tensor) -> bool:
    """True when gemm(x, w) can run (otherwise use torch.nn.functional.linear)."""
    if not (x.is_cuda and w.is_cuda and x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16):
        return False
    if w.dim() != 2 or x.dim() < 1 or x.shape[-1] != w.shape[1]:
        return False
    k = w.shape[1]
    m = x.numel() // k if k else 0
    if not (1 <= m <= MAX_TOKENS) or k % 8:
        return False
    if w.stride(1) != 1 or w.stride(0) % 8 or w.data_ptr() % 16:
        return False
    if x.dim() == 2:
        return x.stride(1) == 1 and x.stride(0) % 8 == 0 and x.data_ptr() % 16 == 0
    return x.is_contiguous() and x.data_ptr() % 16 == 0


def gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    plan: Plan | None = None,
) -> torch.Tensor:
    """y = x @ w.T for bf16 x [..., K] with 1..16 rows and bf16 w [N, K]. Output bf16 (or out_dtype=float32)."""
    lead = x.shape[:-1]
    x2 = x if x.dim() == 2 else x.reshape(-1, x.shape[-1])
    m, n, k = x2.shape[0], w.shape[0], w.shape[1]
    plan = plan or get_plan(n, k, x2.device)
    tile = plan.tile(m)
    ws, cnt = _workspace(x2.device, plan, tile, m, n)
    args = (tile.rows, tile.unroll, plan.lanes_per_row, tile.warps, plan.ksplit)
    if out is not None:
        y2 = out if out.dim() == 2 else out.view(-1, n)
        torch.ops.lowlat_gemm.gemm_out(y2, x2, w, ws, cnt, *args)
        return out
    fp32 = out_dtype == torch.float32
    if out_dtype not in (None, torch.bfloat16, torch.float32):
        raise ValueError(f"out_dtype must be bfloat16 or float32, got {out_dtype}")
    y = torch.ops.lowlat_gemm.gemm(x2, w, ws, cnt, *args, fp32)
    return y if x.dim() == 2 else y.view(*lead, n)
