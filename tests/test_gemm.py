# SPDX-License-Identifier: Apache-2.0
import itertools

import pytest
import torch

import lowlat_gemm
from lowlat_gemm import Plan, Tile

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

SHAPES = [  # (N, K): odd sizes plus the Qwen3.8-Flash-Next TP=1 projections
    (1, 8), (7, 24), (33, 136), (96, 2560), (320, 10240), (336, 10240), (512, 2560), (640, 2560),
    (1280, 2560), (2560, 640), (2560, 2560), (2560, 6144), (10240, 320), (13312, 2560), (16384, 2560),
]
TOKENS = [1, 2, 3, 4, 5, 8, 9, 16]


def _data(m, n, k, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, device="cuda", generator=g).to(torch.bfloat16)
    w = (torch.randn(n, k, device="cuda", generator=g) / k**0.5).to(torch.bfloat16)
    return x, w


def _ref(x, w):
    return x.double() @ w.double().T


def _plans(n, k):
    base = lowlat_gemm.default_plan(n, k, torch.cuda.get_device_properties(0).multi_processor_count)
    out = [base]
    for lanes, ksplit in [(32, 1), (16, 1), (8, 1), (32, 4), (16, 3)]:
        if k // 8 >= ksplit:
            out.append(Plan(lanes_per_row=lanes, ksplit=ksplit))
    return out


@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("m", TOKENS)
def test_matches_fp64_reference(m, n, k):
    x, w = _data(m, n, k)
    ref = _ref(x, w)
    cublas_err = (torch.nn.functional.linear(x, w).double() - ref).abs().max().item()
    for plan in _plans(n, k):
        y = lowlat_gemm.gemm(x, w, plan=plan)
        assert y.shape == (m, n) and y.dtype == torch.bfloat16
        err = (y.double() - ref).abs().max().item()
        # bf16 output rounding dominates; allow one extra bf16 ulp of the largest output over cuBLAS
        tol = cublas_err + ref.abs().max().item() * 2**-8 + 1e-6
        assert err <= tol, f"plan {plan}: max err {err} > {tol} (cuBLAS {cublas_err})"


@pytest.mark.parametrize("n,k", [(336, 10240), (2560, 6144), (10240, 320), (16384, 2560)])
def test_fp32_output_close_to_reference(n, k):
    x, w = _data(4, n, k)
    ref = _ref(x, w)
    for plan in _plans(n, k):
        y = lowlat_gemm.gemm(x, w, out_dtype=torch.float32, plan=plan)
        assert y.dtype == torch.float32
        rel = ((y.double() - ref).abs().max() / ref.abs().max()).item()
        assert rel < 1e-4, f"plan {plan}: relative error {rel}"


@pytest.mark.parametrize("n,k", [(336, 10240), (2560, 6144), (10240, 320), (16384, 2560), (33, 136)])
def test_batch_invariance(n, k):
    """Row i of an M-row call equals the 1-row call on x[i], for every M and every tile (same lanes/ksplit)."""
    x, w = _data(16, n, k, seed=1)
    for plan in _plans(n, k):
        single = torch.cat([lowlat_gemm.gemm(x[i : i + 1], w, plan=plan) for i in range(16)])
        tiles = [Tile(r, u, wp) for (r, u), wp in itertools.product(sorted(lowlat_gemm._RU_PAIRS), (1, 4, 8))]
        for m in TOKENS:
            for tile in tiles:
                p = Plan(lanes_per_row=plan.lanes_per_row, ksplit=plan.ksplit, default_tile=tile)
                y = lowlat_gemm.gemm(x[:m], w, plan=p)
                assert torch.equal(y, single[:m]), f"M={m} plan={p} differs from the single-row result"


def test_split_k_deterministic_and_counters_reset():
    x, w = _data(8, 336, 10240)
    plan = Plan(lanes_per_row=32, ksplit=8)
    first = lowlat_gemm.gemm(x, w, plan=plan)
    for _ in range(50):
        assert torch.equal(lowlat_gemm.gemm(x, w, plan=plan), first)
    _, cnt = lowlat_gemm._ws[torch.cuda.current_device()]
    assert int(cnt.abs().sum()) == 0


def test_strided_x_and_out():
    x_full, w = _data(4, 2560, 2560 + 64)
    x = x_full[:, :2560]  # row stride 2624
    w = w[:, :2560].contiguous()
    out = torch.empty(4, 2560, device="cuda", dtype=torch.bfloat16)
    lowlat_gemm.gemm(x, w, out=out)
    assert torch.equal(out, lowlat_gemm.gemm(x.contiguous(), w))


def test_3d_input():
    x, w = _data(6, 640, 2560)
    y = lowlat_gemm.gemm(x.view(2, 3, 2560), w)
    assert y.shape == (2, 3, 640)
    assert torch.equal(y.view(6, 640), lowlat_gemm.gemm(x, w))


def test_cuda_graph_replay():
    x, w = _data(4, 336, 10240)
    plan = Plan(lanes_per_row=32, ksplit=8)
    lowlat_gemm.reserve_workspace(x.device, plan.ksplit * 16 * 336, 4096)
    eager = lowlat_gemm.gemm(x, w, plan=plan)
    out = torch.empty_like(eager)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        lowlat_gemm.gemm(x, w, out=out, plan=plan)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        lowlat_gemm.gemm(x, w, out=out, plan=plan)
    for _ in range(10):
        out.zero_()
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, eager)


def test_eligible_rejects():
    x, w = _data(4, 64, 64)
    assert lowlat_gemm.eligible(x, w)
    assert not lowlat_gemm.eligible(x.float(), w)
    assert not lowlat_gemm.eligible(torch.cat([x] * 5), w)  # 20 rows
    x12, w12 = _data(2, 64, 12)
    assert not lowlat_gemm.eligible(x12, w12)  # K % 8 != 0
