# SPDX-License-Identifier: Apache-2.0
"""Low-latency GEMM for NVIDIA Blackwell SM120 (RTX PRO 6000 Blackwell, RTX 50 series).

    y = lowlat_gemm.gemm(x, w, ...)      # y[..., N] = x[..., K] @ w[N, K]^T (dequantized)

Precision is taken from the dtypes of x and w:

    x bf16/f16, w same          W16A16
    x bf16/f16, w float8_e4m3fn W8A16   (w_scale: per-tensor or per-channel)
    x bf16/f16, w fp4           W4A16   (w_block_scale [N, K/16] e4m3; w_scale: global)
    x float8_e4m3fn, w same     W8A8 FP8 (x_scale per-tensor or per-token, w_scale per-tensor or per-channel)
    x int8, w int8              W8A8 INT8
    x fp4, w fp4                W4A4 NVFP4 (x_block_scale, w_block_scale; x_scale / w_scale / alpha: global)

fp4 tensors are torch.float4_e2m1fn_x2 (or uint8 together with a block scale), two values per byte, [rows, K/2].
Dequantized operands are x_deq = x * x_scale (* x_block_scale), w_deq = w * w_scale (* w_block_scale), and
y = alpha * x_deq @ w_deq^T + bias.

Two engines: a CUDA-core streaming kernel for W16A16 with at most 16 rows ("skinny", batch invariant) and a
tensor-core kernel for every precision and any M. A plan per (precision, N, K) picks the engine and tiling for each
M; tools/tune.py measures plans, otherwise built-in heuristics are used.
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
    "KINDS",
    "Plan",
    "SkinnyOrder",
    "SkinnyTile",
    "TcChoice",
    "default_plan",
    "gemm",
    "get_plan",
    "kind_of",
    "prefer_lowlat",
    "load_plans",
    "quantize_fp8",
    "quantize_int8",
    "quantize_nvfp4",
    "reserve_workspace",
    "save_plans",
    "set_pdl",
    "set_plan",
    "supports",
    "workspace_need",
]

SKINNY_MAX_TOKENS = 16
_FP4 = getattr(torch, "float4_e2m1fn_x2", None)
_FP8 = torch.float8_e4m3fn

# kind name -> (id, K per pipeline stage, K multiple required)
KINDS = {
    "bf16": (0, 64, 8),
    "f16": (1, 64, 8),
    "fp8": (2, 128, 16),
    "int8": (3, 128, 16),
    "nvfp4": (4, 256, 64),
    "w8a16_bf16": (5, 128, 16),
    "w8a16_f16": (6, 128, 16),
    "w4a16_bf16": (7, 128, 64),
    "w4a16_f16": (8, 128, 64),
}
_MIXED = {"w8a16_bf16", "w8a16_f16", "w4a16_bf16", "w4a16_f16"}
NUM_CFGS = 8


# ---------------------------------------------------------------------------------------------------------------
# Plans


@dataclass(frozen=True)
class SkinnyTile:
    """Per-M tiling of the skinny kernel; does not change numerics."""

    rows: int = 1
    unroll: int = 4
    warps: int = 4

    def __post_init__(self):
        if (self.rows, self.unroll) not in {(1, 2), (1, 4), (1, 8), (2, 2), (2, 4), (4, 2)}:
            raise ValueError(f"unsupported skinny (rows, unroll) {(self.rows, self.unroll)}")
        if not 1 <= self.warps <= 8:
            raise ValueError(f"warps must be in [1, 8], got {self.warps}")


@dataclass
class SkinnyOrder:
    """Skinny summation order (fixed per N, K) plus the per-M tiles."""

    lanes_per_row: int = 32
    ksplit: int = 1
    tiles: dict[int, SkinnyTile] = field(default_factory=dict)
    default_tile: SkinnyTile = field(default_factory=SkinnyTile)
    cluster: bool = False  # split-K reduced in a thread block cluster (ksplit <= 8); same summation order

    def __post_init__(self):
        if self.lanes_per_row not in (8, 16, 32):
            raise ValueError(f"lanes_per_row must be 8, 16 or 32, got {self.lanes_per_row}")
        if not 1 <= self.ksplit <= 64:
            raise ValueError(f"ksplit must be in [1, 64], got {self.ksplit}")
        if self.cluster and self.ksplit > 8:
            raise ValueError(f"cluster split-K needs ksplit <= 8, got {self.ksplit}")

    def tile(self, m: int) -> SkinnyTile:
        return _nearest(self.tiles, m, self.default_tile)


@dataclass(frozen=True)
class TcChoice:
    cfg: int
    swap: bool       # weight is the MMA A operand (few tokens)
    stages: int = 4
    ksplit: int = 1
    cluster: bool = False  # split-K reduced in a thread block cluster (ksplit <= 8, no workspace)

    def __post_init__(self):
        if not 0 <= self.cfg < NUM_CFGS:
            raise ValueError(f"cfg must be in [0, {NUM_CFGS}), got {self.cfg}")
        if not 1 <= self.ksplit <= 64 or not 2 <= self.stages <= 8:
            raise ValueError(f"bad ksplit/stages {self.ksplit}/{self.stages}")
        if self.cluster and self.ksplit > 8:
            raise ValueError(f"cluster split-K needs ksplit <= 8, got {self.ksplit}")


@dataclass
class Plan:
    """Per (kind, N, K): which engine and tiling to use for each M (nearest lower M listed wins)."""

    skinny: SkinnyOrder | None = None
    skinny_max_m: int = 0              # skinny for M <= this (W16A16 only)
    tc: dict[int, TcChoice] = field(default_factory=dict)
    prefer_vendor: list[int] = field(default_factory=list)  # measured M where the vendor library was faster

    def to_json(self) -> dict:
        d: dict = {"skinny_max_m": self.skinny_max_m, "tc": {str(m): asdict(c) for m, c in sorted(self.tc.items())}}
        if self.prefer_vendor:
            d["prefer_vendor"] = sorted(self.prefer_vendor)
        if self.skinny is not None:
            d["skinny"] = {
                "lanes_per_row": self.skinny.lanes_per_row,
                "ksplit": self.skinny.ksplit,
                "cluster": self.skinny.cluster,
                "default_tile": asdict(self.skinny.default_tile),
                "tiles": {str(m): asdict(t) for m, t in sorted(self.skinny.tiles.items())},
            }
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Plan":
        sk = d.get("skinny")
        skinny = None
        if sk is not None:
            skinny = SkinnyOrder(
                lanes_per_row=int(sk["lanes_per_row"]),
                ksplit=int(sk["ksplit"]),
                cluster=bool(sk.get("cluster", False)),
                default_tile=SkinnyTile(**sk.get("default_tile", {})),
                tiles={int(m): SkinnyTile(**t) for m, t in sk.get("tiles", {}).items()},
            )
        tc = {int(m): TcChoice(**c) for m, c in d.get("tc", {}).items()}
        return cls(skinny=skinny, skinny_max_m=int(d.get("skinny_max_m", 0)), tc=tc,
                   prefer_vendor=[int(m) for m in d.get("prefer_vendor", [])])


def _tc_lookup(table: dict, m: int):
    """Tuned choice for m: the entry of the largest tuned M <= m, used up to twice the largest tuned M."""
    if m in table:
        return table[m]
    if not table or m > 2 * max(table):
        return None
    lower = [k for k in table if k <= m]
    return table[max(lower)] if lower else table[min(table)]


def _nearest(table: dict, m: int, default):
    if m in table:
        return table[m]
    lower = [k for k in table if k <= m]
    if lower:
        return table[max(lower)]
    return table[min(table)] if table else default


_info_cache: dict[tuple[int, int], tuple[int, int, int, int]] = {}


def tc_info(kind: str, cfg: int) -> tuple[int, int, int, int]:
    """(BP, BQ, threads, shared bytes per pipeline stage) of a tensor-core config."""
    key = (KINDS[kind][0], cfg)
    if key not in _info_cache:
        _info_cache[key] = tuple(torch.ops.lowlat_gemm.tc_info(*key))
    return _info_cache[key]


def max_stages(kind: str, cfg: int) -> int:
    return min(8, (99 * 1024 - 128) // tc_info(kind, cfg)[3])


def default_skinny_order(n: int, k: int, sm_count: int) -> SkinnyOrder:
    kv = k // 8
    lanes = next((l for l in (32, 16, 8) if kv >= 4 * l), 8)
    rows = 2 if n >= 8192 else 1
    groups = -(-n // rows)
    ksplit = 1
    while groups * ksplit < 8 * sm_count and kv // (ksplit * 2) >= 2 * lanes and ksplit < 16:
        ksplit *= 2
    return SkinnyOrder(lanes_per_row=lanes, ksplit=ksplit, default_tile=SkinnyTile(rows=rows, unroll=4, warps=4))


# next smaller tile per config (BPxBQ): 256x128 -> 128x128 -> 64x64 -> 64x8; 128x64 -> 128x32 -> 128x16 -> 64x8
_SMALLER = {7: 6, 6: 4, 5: 4, 4: 0, 3: 2, 2: 1, 1: 0, 0: None}


def default_tc_choice(kind: str, m: int, n: int, k: int, sm_count: int) -> TcChoice:
    """Heuristic: weight-as-A ("swap") tiles for few tokens, square tiles for many; split K to fill the GPU."""
    if kind in _MIXED or m <= 64:
        swap = True
        cfg = 0 if m <= 8 else 1 if m <= 16 else 2 if m <= 32 else 3 if m <= 64 else 6
    else:
        swap = False
        cfg = 4 if m * n <= (1 << 20) else 6 if m * n <= (1 << 24) else 7
    while max_stages(kind, cfg) < 2:  # e.g. cfg 7 with fp4 / mixed operands
        cfg = 6 if cfg == 7 else 4
    p, q = (n, m) if swap else (m, n)

    def tiles_of(c):
        bp, bq, _, _ = tc_info(kind, c)
        return -(-p // bp) * -(-q // bq)

    # a grid far smaller than the GPU leaves one SM doing all the MMA work: step down to smaller tiles
    while tiles_of(cfg) * 8 < sm_count and _SMALLER[cfg] is not None:
        cfg = _SMALLER[cfg]
    tiles = tiles_of(cfg)
    nk = -(-k // KINDS[kind][1])
    ksplit = 1
    while tiles * ksplit < sm_count and nk // (ksplit * 2) >= 2 and ksplit < 16:
        ksplit *= 2
    stages = min(max_stages(kind, cfg), 6 if cfg <= 2 else 4)
    return TcChoice(cfg=cfg, swap=swap, stages=max(stages, 2), ksplit=ksplit)


def default_plan(kind: str, n: int, k: int, sm_count: int) -> Plan:
    plan = Plan()
    if kind in ("bf16", "f16"):
        plan.skinny = default_skinny_order(n, k, sm_count)
        plan.skinny_max_m = 8
    for m in (1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096):
        plan.tc[m] = default_tc_choice(kind, m, n, k, sm_count)
    return plan


_lock = threading.Lock()
_plans: dict[tuple[str, str, int, int], Plan] = {}
_loaded: set[str] = set()
_sm_counts: dict[int, int] = {}


def _dev(device: torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device if device.index is not None else torch.device("cuda", torch.cuda.current_device())


def _sm_count(device: torch.device) -> int:
    if device.index not in _sm_counts:
        _sm_counts[device.index] = torch.cuda.get_device_properties(device.index).multi_processor_count
    return _sm_counts[device.index]


def _file_key(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()


def load_plans(path: str | os.PathLike, device_name: str | None = None) -> int:
    """Load a plan file {"device": ..., "plans": {"<kind>:<N>x<K>": plan}}. Returns the number of plans."""
    data = json.loads(Path(path).read_text())
    name = device_name or data.get("device")
    if not name:
        raise ValueError(f"{path}: no device name in the file and none given")
    with _lock:
        for key, d in data.get("plans", {}).items():
            kind, nk = key.split(":")
            n, k = (int(v) for v in nk.lower().split("x"))
            _plans[(name, kind, n, k)] = Plan.from_json(d)
    return len(data.get("plans", {}))


def save_plans(path: str | os.PathLike, plans: dict[tuple[str, int, int], Plan], device_name: str) -> None:
    data = {"device": device_name, "plans": {f"{k}:{n}x{kk}": p.to_json() for (k, n, kk), p in plans.items()}}
    Path(path).write_text(json.dumps(data, indent=1))


def _ensure_loaded(name: str) -> None:
    if name in _loaded:
        return
    with _lock:
        if name in _loaded:
            return
        _loaded.add(name)
    packaged = Path(__file__).with_name("plans") / f"{_file_key(name)}.json"
    if packaged.is_file():
        load_plans(packaged, name)
    env = os.environ.get("LLG_PLAN_FILE")
    if env:
        load_plans(env)


def set_plan(kind: str, n: int, k: int, plan: Plan, device: torch.device | None = None) -> None:
    device = _dev(device)
    name = torch.cuda.get_device_name(device)
    _ensure_loaded(name)
    with _lock:
        _plans[(name, kind, n, k)] = plan


def get_plan(kind: str, n: int, k: int, device: torch.device | None = None) -> Plan:
    device = _dev(device)
    name = torch.cuda.get_device_name(device)
    _ensure_loaded(name)
    plan = _plans.get((name, kind, n, k))
    if plan is None:
        plan = default_plan(kind, n, k, _sm_count(device))
        with _lock:
            plan = _plans.setdefault((name, kind, n, k), plan)
    return plan


# ---------------------------------------------------------------------------------------------------------------
# Split-K workspace (one per device; launches on one stream reuse it, each leaves its counters at zero)

_ws: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def set_pdl(on: bool = True) -> None:
    """Programmatic dependent launch for all later gemm() calls (process-wide).

    A GEMM may then start while the previous kernel is still running and stream its weight early; activations,
    outputs and split-K buffers are touched only after the previous kernel has finished. Enable it only when no kernel
    still in flight writes the weights (true for inference with fixed weights). Graphs keep the setting they were
    captured with."""
    torch.ops.lowlat_gemm.set_pdl(bool(on))


def pdl_enabled() -> bool:
    return bool(torch.ops.lowlat_gemm.pdl_enabled())


def reserve_workspace(device: torch.device, elems: int, counters: int) -> None:
    """Grow the split-K buffers of `device` to at least `elems` 4-byte partials and `counters` counters.
    Call before CUDA graph capture; growing during capture raises."""
    device = _dev(device)
    ws, cnt = _ws.get(device.index, (None, None))
    if ws is not None and ws.numel() >= elems and cnt.numel() >= counters:
        return
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"lowlat_gemm: split-K workspace too small during CUDA graph capture (need {elems} partials, "
            f"{counters} counters); call reserve_workspace() before capture"
        )
    elems = max(elems, ws.numel() if ws is not None else 0, 1 << 20)
    counters = max(counters, cnt.numel() if cnt is not None else 0, 1 << 12)
    _ws[device.index] = (
        torch.empty(elems, dtype=torch.float32, device=device),
        torch.zeros(counters, dtype=torch.int32, device=device),
    )


def _workspace(device: torch.device, elems: int, counters: int) -> tuple[torch.Tensor, torch.Tensor]:
    reserve_workspace(device, elems, counters)
    return _ws[device.index]


# ---------------------------------------------------------------------------------------------------------------
# Kind detection


def _is_fp4(t: torch.Tensor, block_scale) -> bool:
    return (_FP4 is not None and t.dtype == _FP4) or (t.dtype == torch.uint8 and block_scale is not None)


def kind_of(x: torch.Tensor, w: torch.Tensor, x_block_scale=None, w_block_scale=None) -> str:
    xt, wt = x.dtype, w.dtype
    if xt in (torch.bfloat16, torch.float16):
        tag = "bf16" if xt == torch.bfloat16 else "f16"
        if wt == xt:
            return tag
        if wt == _FP8:
            return f"w8a16_{tag}"
        if _is_fp4(w, w_block_scale):
            return f"w4a16_{tag}"
    elif xt == _FP8 and wt == _FP8:
        return "fp8"
    elif xt == torch.int8 and wt == torch.int8:
        return "int8"
    elif _is_fp4(x, x_block_scale) and _is_fp4(w, w_block_scale):
        return "nvfp4"
    raise TypeError(f"lowlat_gemm: unsupported dtype pair x={xt}, w={wt}")


def _as_bytes(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8) if t.dtype != torch.uint8 and t.element_size() == 1 else t


def _vec(v, device) -> torch.Tensor | None:
    if v is None:
        return None
    v = v.reshape(-1)
    return v if v.dtype == torch.float32 and v.is_contiguous() else v.float().contiguous()


def prefer_lowlat(kind: str, n: int, k: int, m: int, device: torch.device | None = None) -> bool:
    """False when the plan measured the vendor library as faster at the nearest tuned M (<= m, within 2x)."""
    plan = get_plan(kind, n, k, device)
    if not plan.prefer_vendor or not plan.tc:
        return True
    tuned = [t for t in plan.tc if t <= m]
    if not tuned or m > 2 * max(plan.tc):
        return True
    return max(tuned) not in plan.prefer_vendor


def supports(x: torch.Tensor, w: torch.Tensor, x_block_scale=None, w_block_scale=None) -> bool:
    """True when gemm(x, w, ...) can run on these operands (dtype pair, K alignment, 16-byte rows)."""
    if not (x.is_cuda and w.is_cuda and w.dim() == 2 and x.dim() >= 1):
        return False
    try:
        kind = kind_of(x, w, x_block_scale, w_block_scale)
    except TypeError:
        return False
    fp4w = kind in ("nvfp4", "w4a16_bf16", "w4a16_f16")
    k = w.shape[1] * (2 if fp4w else 1)
    if k == 0 or k % KINDS[kind][2] or x.shape[-1] * (2 if kind == "nvfp4" else 1) != k:
        return False
    for t in (x.reshape(-1, x.shape[-1]) if x.dim() != 2 and x.is_contiguous() else x, w):
        if t.dim() != 2 or t.stride(1) != 1 or (t.stride(0) * t.element_size()) % 16 or t.data_ptr() % 16:
            return False
    return True


def workspace_need(kind: str, n: int, k: int, max_m: int, device: torch.device | None = None) -> tuple[int, int]:
    """(partials, counters) the split-K buffers need for any M <= max_m of this (kind, N, K)."""
    device = _dev(device)
    plan = get_plan(kind, n, k, device)
    elems = counters = 0
    for m in range(1, max_m + 1):
        if plan.skinny is not None and m <= min(plan.skinny_max_m, SKINNY_MAX_TOKENS):
            o = plan.skinny
            t = o.tile(m)
            if o.ksplit > 1 and not o.cluster:
                elems = max(elems, o.ksplit * m * n)
                counters = max(counters, -(-n // (t.warps * (32 // o.lanes_per_row) * t.rows)))
            continue
        c = _tc_lookup(plan.tc, m) or default_tc_choice(kind, m, n, k, _sm_count(device))
        if c.ksplit > 1 and not c.cluster:
            bp, bq, _, _ = tc_info(kind, c.cfg)
            p, q = (n, m) if (c.swap or kind in _MIXED) else (m, n)
            elems = max(elems, c.ksplit * p * q)
            counters = max(counters, -(-p // bp) * -(-q // bq))
    return elems, counters


# ---------------------------------------------------------------------------------------------------------------
# GEMM


def gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    x_scale=None,
    w_scale=None,
    x_block_scale: torch.Tensor | None = None,
    w_block_scale: torch.Tensor | None = None,
    alpha=None,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    out: torch.Tensor | None = None,
    plan: Plan | None = None,
    prefetch: torch.Tensor | None = None,
    prefetch_bytes: int = 0,
) -> torch.Tensor:
    """y = alpha * (x * x_scale) @ (w * w_scale)^T + bias. See the module docstring for the precisions.

    x_scale / w_scale: None, a float, or a float32 tensor with 1 element or one per row (M for x, N for w).
    alpha: None, a float or a 1-element float32 tensor (device scalars stay valid inside CUDA graphs).
    prefetch: a contiguous tensor (typically the weight of the GEMM that runs next) whose first prefetch_bytes
    (0 = all) the kernel pulls into L2 once its own weight is streaming.
    """
    kind = kind_of(x, w, x_block_scale, w_block_scale)
    fp4 = kind in ("nvfp4", "w4a16_bf16", "w4a16_f16")
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    m = x2.shape[0]
    n = w.shape[0]
    k = w.shape[1] * (2 if fp4 else 1)
    if x2.shape[1] * (2 if kind == "nvfp4" else 1) != k:
        raise ValueError(f"lowlat_gemm: x has K={x2.shape[1]} but w {tuple(w.shape)} implies K={k}")
    device = _dev(x2.device)
    if out_dtype is None:
        out_dtype = out.dtype if out is not None else (x.dtype if x.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16)
    y2 = out.view(-1, n) if out is not None else torch.empty(m, n, dtype=out_dtype, device=device)
    if m == 0:
        return out if out is not None else y2.view(*lead, n)
    plan = plan or get_plan(kind, n, k, device)

    if (
        plan.skinny is not None
        and m <= min(plan.skinny_max_m, SKINNY_MAX_TOKENS)
        and x_scale is None and w_scale is None and alpha is None and bias is None
        and y2.dtype in (x.dtype, torch.float32)
    ):
        order = plan.skinny
        t = order.tile(m)
        if order.ksplit > 1 and not order.cluster:
            per_block = t.warps * (32 // order.lanes_per_row) * t.rows
            ws, cnt = _workspace(device, order.ksplit * m * n, -(-n // per_block))
        else:
            ws, cnt = _workspace(device, 0, 0)
        torch.ops.lowlat_gemm.skinny_out(y2, x2, w, ws, cnt, t.rows, t.unroll, order.lanes_per_row, t.warps, order.ksplit,
                                         order.cluster, prefetch, prefetch_bytes)
        return out if out is not None else y2.view(*lead, n)

    c = _tc_lookup(plan.tc, m) or default_tc_choice(kind, m, n, k, _sm_count(device))
    swap = c.swap or kind in _MIXED
    # python floats fold into the host-side alpha (no tensor creation, so CUDA-graph capture stays legal)
    alpha_f = 1.0
    for v in (x_scale, w_scale):
        if v is not None and not torch.is_tensor(v):
            alpha_f *= float(v)
    xs = _vec(x_scale, device) if torch.is_tensor(x_scale) else None
    ws_ = _vec(w_scale, device) if torch.is_tensor(w_scale) else None
    alpha_t = None
    if torch.is_tensor(alpha):
        alpha_t = alpha.reshape(1)
        alpha_t = alpha_t if alpha_t.dtype == torch.float32 else alpha_t.float()
    elif alpha is not None:
        alpha_f *= float(alpha)
    xa, wa = _as_bytes(x2) if kind == "nvfp4" or x2.element_size() == 1 else x2, _as_bytes(w) if w.element_size() == 1 else w
    xsf = _as_bytes(x_block_scale) if x_block_scale is not None else None
    wsf = _as_bytes(w_block_scale) if w_block_scale is not None else None
    if swap:
        a, b, a_sf, b_sf, row_s, col_s, p, q = wa, xa, wsf, xsf, ws_, xs, n, m
    else:
        a, b, a_sf, b_sf, row_s, col_s, p, q = xa, wa, xsf, wsf, xs, ws_, m, n
    if kind == "w4a16_bf16" or kind == "w4a16_f16":
        a_sf, b_sf = wsf, None
    bp, bq, _, _ = tc_info(kind, c.cfg)
    tiles = -(-p // bp) * -(-q // bq)
    use_ws = c.ksplit > 1 and not c.cluster
    wsp, cnt = _workspace(device, c.ksplit * p * q if use_ws else 0, tiles if use_ws else 0)
    torch.ops.lowlat_gemm.tc_gemm(
        y2, a, b, a_sf, b_sf, row_s, col_s, bias, alpha_t, alpha_f, KINDS[kind][0], swap, swap,
        c.cfg, c.stages, c.ksplit, wsp, cnt, c.cluster, prefetch, prefetch_bytes,
    )
    return out if out is not None else y2.view(*lead, n)


# ---------------------------------------------------------------------------------------------------------------
# Activation quantization


def quantize_fp8(x: torch.Tensor, scale: torch.Tensor | None = None, min_scale: float = 1e-12):
    """x [M, K] bf16/f16/f32 -> (q float8_e4m3fn [M, K], scale). Static if `scale` (1-element float32 tensor) is
    given, otherwise dynamic per token: scale[m] = amax(x[m]) / 448."""
    x2 = x.reshape(-1, x.shape[-1])
    q = torch.empty(x2.shape, dtype=_FP8, device=x.device)
    if scale is not None:
        s = scale.reshape(1).float()
        torch.ops.lowlat_gemm.quant(x2, q.view(torch.uint8), None, s, None, 0, min_scale)
        return q.view(*x.shape), s
    s = torch.empty(x2.shape[0], dtype=torch.float32, device=x.device)
    torch.ops.lowlat_gemm.quant(x2, q.view(torch.uint8), None, None, s, 1, min_scale)
    return q.view(*x.shape), s


def quantize_int8(x: torch.Tensor, min_scale: float = 1e-12):
    """x [M, K] -> (q int8 [M, K], scale float32 [M]) with scale[m] = amax(x[m]) / 127, round-to-nearest-even."""
    x2 = x.reshape(-1, x.shape[-1])
    q = torch.empty(x2.shape, dtype=torch.int8, device=x.device)
    s = torch.empty(x2.shape[0], dtype=torch.float32, device=x.device)
    torch.ops.lowlat_gemm.quant(x2, q, None, None, s, 2, min_scale)
    return q.view(*x.shape), s


def quantize_nvfp4(x: torch.Tensor, global_scale: torch.Tensor):
    """x [M, K] -> (q fp4 [M, K/2], block_scale float8_e4m3fn [M, K/16]).

    Per 16 elements: sf = e4m3(amax / 6 * g), q = e2m1(x * g / sf); x ~= q * sf / g. Pass x_scale = 1 / g to gemm.
    """
    x2 = x.reshape(-1, x.shape[-1])
    mm, kk = x2.shape
    q = torch.empty(mm, kk // 2, dtype=torch.uint8, device=x.device)
    sf = torch.empty(mm, kk // 16, dtype=torch.uint8, device=x.device)
    torch.ops.lowlat_gemm.quant(x2, q, sf, global_scale.reshape(1).float(), None, 3, 0.0)
    q = q.view(_FP4) if _FP4 is not None else q
    return q, sf.view(_FP8)
