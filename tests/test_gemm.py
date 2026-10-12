# SPDX-License-Identifier: Apache-2.0
import itertools

import pytest
import torch

import lowlat_gemm as L
from lowlat_gemm import Plan, SkinnyOrder, SkinnyTile, TcChoice

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

DEV = "cuda"
FP8 = torch.float8_e4m3fn
FP4 = getattr(torch, "float4_e2m1fn_x2", torch.uint8)
E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float64)


def gen(seed=0):
    return torch.Generator(device=DEV).manual_seed(seed)


def unpack_fp4(q: torch.Tensor) -> torch.Tensor:
    b = q.view(torch.uint8).long()
    lut = E2M1.to(q.device)
    out = torch.stack([lut[b & 0xF], lut[b >> 4]], dim=-1)
    return out.reshape(*b.shape[:-1], b.shape[-1] * 2)


def expand_blocks(sf: torch.Tensor) -> torch.Tensor:
    return sf.view(FP8).double().repeat_interleave(16, dim=-1)


def random_fp8(rows, k, g):
    return (torch.randn(rows, k, device=DEV, generator=g) * 4).to(FP8)


def random_fp4(rows, k, g):
    return torch.randint(0, 256, (rows, k // 2), device=DEV, generator=g, dtype=torch.uint8).view(FP4)


def random_sf(rows, k, g):  # moderate positive e4m3 block scales
    return torch.randint(0x18, 0x48, (rows, k // 16), device=DEV, generator=g, dtype=torch.uint8).view(FP8)


def operands(kind, m, n, k, seed=0):
    """Returns (x, w, kwargs, x_deq, w_deq) with float64 dequantized references."""
    g = gen(seed)
    kw = {}
    if kind in ("bf16", "f16"):
        dt = torch.bfloat16 if kind == "bf16" else torch.float16
        x = torch.randn(m, k, device=DEV, generator=g).to(dt)
        w = (torch.randn(n, k, device=DEV, generator=g) / k**0.5).to(dt)
        return x, w, kw, x.double(), w.double()
    if kind.startswith("w8a16"):
        dt = torch.bfloat16 if kind.endswith("bf16") else torch.float16
        x = torch.randn(m, k, device=DEV, generator=g).to(dt)
        w = random_fp8(n, k, g)
        ws = torch.rand(n, device=DEV, generator=g) + 0.5
        kw["w_scale"] = ws
        return x, w, kw, x.double(), w.double() * ws.double()[:, None]
    if kind.startswith("w4a16"):
        dt = torch.bfloat16 if kind.endswith("bf16") else torch.float16
        x = torch.randn(m, k, device=DEV, generator=g).to(dt)
        w, wsf = random_fp4(n, k, g), random_sf(n, k, g)
        kw.update(w_block_scale=wsf, w_scale=torch.tensor([0.01], device=DEV))
        return x, w, kw, x.double(), unpack_fp4(w) * expand_blocks(wsf) * 0.01
    if kind == "fp8":
        x, w = random_fp8(m, k, g), random_fp8(n, k, g)
        xs, ws = torch.rand(m, device=DEV, generator=g) + 0.5, torch.rand(n, device=DEV, generator=g) + 0.5
        kw.update(x_scale=xs, w_scale=ws)
        return x, w, kw, x.double() * xs.double()[:, None], w.double() * ws.double()[:, None]
    if kind == "int8":
        x = torch.randint(-128, 128, (m, k), device=DEV, generator=g, dtype=torch.int8)
        w = torch.randint(-128, 128, (n, k), device=DEV, generator=g, dtype=torch.int8)
        xs, ws = torch.rand(m, device=DEV, generator=g) + 0.5, torch.rand(n, device=DEV, generator=g) + 0.5
        kw.update(x_scale=xs, w_scale=ws)
        return x, w, kw, x.double() * xs.double()[:, None], w.double() * ws.double()[:, None]
    if kind == "nvfp4":
        x, w = random_fp4(m, k, g), random_fp4(n, k, g)
        xsf, wsf = random_sf(m, k, g), random_sf(n, k, g)
        kw.update(x_block_scale=xsf, w_block_scale=wsf, alpha=torch.tensor([1e-3], device=DEV))
        return x, w, kw, unpack_fp4(x) * expand_blocks(xsf), unpack_fp4(w) * expand_blocks(wsf) * 1e-3
    raise ValueError(kind)


def rel_err(y, ref):
    return ((y.double() - ref).abs().max() / ref.abs().max().clamp_min(1e-30)).item()


KIND_K = {"bf16": 8, "f16": 8, "fp8": 16, "int8": 16, "nvfp4": 64, "w8a16_bf16": 16, "w8a16_f16": 16,
          "w4a16_bf16": 64, "w4a16_f16": 64}


@pytest.mark.parametrize("kind", list(L.KINDS))
@pytest.mark.parametrize("m", [1, 3, 8, 17, 64, 200])
def test_default_plan_matches_reference(kind, m):
    n, k = 333, 4 * 256 + KIND_K[kind]  # odd N, partial last K tile
    x, w, kw, xd, wd = operands(kind, m, n, k)
    ref = xd @ wd.T
    y = L.gemm(x, w, out_dtype=torch.float32, **kw)
    assert y.shape == (m, n) and y.dtype == torch.float32
    assert rel_err(y, ref) < 2e-5, kind


@pytest.mark.parametrize("kind", list(L.KINDS))
def test_every_config_orientation_and_split(kind):
    n, k = 200, 2 * 256 + KIND_K[kind]
    for m in (5, 70):
        x, w, kw, xd, wd = operands(kind, m, n, k, seed=m)
        ref = xd @ wd.T
        for cfg, swap, ksplit in itertools.product(range(L.NUM_CFGS), (False, True), (1, 3)):
            if L.max_stages(kind, cfg) < 2:
                continue
            c = TcChoice(cfg=cfg, swap=swap, stages=min(3, L.max_stages(kind, cfg)), ksplit=ksplit)
            y = L.gemm(x, w, out_dtype=torch.float32, plan=Plan(tc={1: c}), **kw)
            e = rel_err(y, ref)
            assert e < 2e-5, f"{kind} m={m} {c}: rel err {e}"


@pytest.mark.parametrize("kind", ["bf16", "fp8", "nvfp4", "w4a16_bf16"])
def test_epilogue_bias_alpha_and_16bit_out(kind):
    m, n, k = 7, 130, 512
    x, w, kw, xd, wd = operands(kind, m, n, k)
    bias = torch.randn(n, device=DEV).to(torch.bfloat16)
    base_alpha = kw.pop("alpha", None)  # nvfp4: already folded into wd
    alpha = 0.5 * (base_alpha.item() if base_alpha is not None else 1.0)
    ref = 0.5 * (xd @ wd.T) + bias.double()
    y = L.gemm(x, w, alpha=alpha, bias=bias, out_dtype=torch.bfloat16, **kw)
    assert y.dtype == torch.bfloat16
    assert rel_err(y, ref) < 1e-2


def test_skinny_batch_invariance():
    """Row i of an M-row skinny call equals the 1-row call on x[i] for every M and tile."""
    for n, k in [(336, 10240), (2560, 6144), (10240, 320), (33, 136)]:
        x, w, _, _, _ = operands("bf16", 16, n, k, seed=1)
        base = L.default_skinny_order(n, k, torch.cuda.get_device_properties(0).multi_processor_count)
        single = torch.cat([L.gemm(x[i : i + 1], w, plan=Plan(skinny=base, skinny_max_m=16)) for i in range(16)])
        for (r, u), wp in itertools.product([(1, 2), (1, 4), (1, 8), (2, 2), (2, 4), (4, 2)], (1, 4, 8)):
            order = SkinnyOrder(lanes_per_row=base.lanes_per_row, ksplit=base.ksplit, default_tile=SkinnyTile(r, u, wp))
            for m in (1, 2, 3, 5, 9, 16):
                y = L.gemm(x[:m], w, plan=Plan(skinny=order, skinny_max_m=16))
                assert torch.equal(y, single[:m]), (n, k, r, u, wp, m)


@pytest.mark.parametrize("dt", [torch.bfloat16, torch.float16])
def test_skinny_matches_reference(dt):
    for n, k in [(1, 8), (96, 2560), (336, 10240), (10240, 320)]:
        x = torch.randn(9, k, device=DEV).to(dt)
        w = (torch.randn(n, k, device=DEV) / k**0.5).to(dt)
        ref = x.double() @ w.double().T
        y = L.gemm(x, w, out_dtype=torch.float32, plan=Plan(skinny=L.default_skinny_order(n, k, 188), skinny_max_m=16))
        assert rel_err(y, ref) < 2e-5


def test_split_k_deterministic_and_counters_reset():
    x, w, kw, _, _ = operands("fp8", 4, 336, 4096)
    plan = Plan(tc={1: TcChoice(cfg=0, swap=True, stages=4, ksplit=8)})
    first = L.gemm(x, w, plan=plan, **kw)
    for _ in range(20):
        assert torch.equal(L.gemm(x, w, plan=plan, **kw), first)
    _, cnt = L._ws[torch.cuda.current_device()]
    assert int(cnt.abs().sum()) == 0


def test_cuda_graph_replay():
    x, w, kw, _, _ = operands("nvfp4", 4, 512, 2048)
    plan = Plan(tc={1: TcChoice(cfg=0, swap=True, stages=4, ksplit=4)})
    eager = L.gemm(x, w, plan=plan, **kw)
    out = torch.empty_like(eager)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        L.gemm(x, w, out=out, plan=plan, **kw)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        L.gemm(x, w, out=out, plan=plan, **kw)
    for _ in range(5):
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, eager)


def test_quantize_fp8():
    x = torch.randn(5, 384, device=DEV).to(torch.bfloat16)
    q, s = L.quantize_fp8(x)
    ref_s = x.float().abs().amax(dim=1) / 448
    assert torch.allclose(s, ref_s)
    # the kernel multiplies by the reciprocal of the scale
    assert torch.equal(q, (x.float() * (1 / ref_s)[:, None]).clamp(-448, 448).to(FP8))
    st = torch.tensor([0.02], device=DEV)
    q2, _ = L.quantize_fp8(x, st)
    assert torch.equal(q2, (x.float() * (1 / st)).clamp(-448, 448).to(FP8))


def test_quantize_int8():
    x = torch.randn(5, 384, device=DEV, dtype=torch.float32)
    q, s = L.quantize_int8(x)
    ref_s = x.abs().amax(dim=1) / 127
    assert torch.allclose(s, ref_s)
    assert torch.equal(q, torch.round(x * (1 / ref_s)[:, None]).clamp(-127, 127).to(torch.int8))


def test_quantize_nvfp4_roundtrip_and_gemm():
    x = torch.randn(6, 512, device=DEV).to(torch.bfloat16)
    g = torch.tensor([448 * 6 / x.float().abs().max().item()], device=DEV)
    q, sf = L.quantize_nvfp4(x, g)
    deq = unpack_fp4(q) * expand_blocks(sf) / g.double()
    # NVFP4 has ~1 significant bit after the scale: within half a step of the largest block value
    blk_max = x.double().abs().reshape(6, -1, 16).amax(-1).repeat_interleave(16, -1)
    assert ((deq - x.double()).abs() <= blk_max * 0.26 + 1e-6).all()
    w, wsf = random_fp4(64, 512, gen(3)), random_sf(64, 512, gen(4))
    y = L.gemm(q, w, x_block_scale=sf, w_block_scale=wsf, x_scale=1.0 / g.item(), out_dtype=torch.float32)
    ref = deq @ (unpack_fp4(w) * expand_blocks(wsf)).T
    assert rel_err(y, ref) < 2e-5


def test_supports():
    x, w, _, _, _ = operands("bf16", 4, 64, 64)
    assert L.supports(x, w)
    assert not L.supports(x.float(), w)
    x12 = torch.randn(2, 12, device=DEV).to(torch.bfloat16)
    assert not L.supports(x12, torch.randn(3, 12, device=DEV).to(torch.bfloat16))
