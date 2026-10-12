# SPDX-License-Identifier: Apache-2.0
"""CPU emulation of csrc/tc_gemm.cuh at thread level (no GPU needed).

It replays the kernel's own index arithmetic -- cp.async tile placement with the XOR swizzle, ldmatrix lane
addressing, the per-precision MMA fragment layouts (as defined by the PTX ISA / CUTLASS SM120 traits), the
mixed-precision K permutation, the NVFP4 scale-factor registers, the epilogue mapping and split-K -- and compares
the result with a float64 reference. It checks the indexing logic, not the hardware.

    python tests/emulate_tc.py            # all kinds x configs on small odd shapes
"""

from __future__ import annotations

import itertools
import sys

import numpy as np

# ---- kernel constants (keep in sync with tc_gemm.cuh) ---------------------------------------------------------

KINDS = {  # name: (a_bits, b_bits, BK, mixed, block_scaled, weight_block_scaled, int_acc, b16 type of mixed/16-bit)
    "bf16": (16, 16, 64, False, False, False, False, "bf16"),
    "f16": (16, 16, 64, False, False, False, False, "f16"),
    "fp8": (8, 8, 128, False, False, False, False, None),
    "int8": (8, 8, 128, False, False, False, True, None),
    "nvfp4": (4, 4, 256, False, True, False, False, None),
    "w8a16_bf16": (8, 16, 128, True, False, False, False, "bf16"),
    "w8a16_f16": (8, 16, 128, True, False, False, False, "f16"),
    "w4a16_bf16": (4, 16, 128, True, False, True, False, "bf16"),
    "w4a16_f16": (4, 16, 128, True, False, True, False, "f16"),
}
CFGS = {0: (64, 8, 16, 8), 1: (128, 16, 32, 16), 2: (128, 32, 32, 32), 3: (128, 64, 64, 32), 4: (64, 64, 32, 32),
        5: (128, 128, 64, 64), 6: (128, 128, 64, 32), 7: (256, 128, 64, 64)}

# ---- number formats -------------------------------------------------------------------------------------------

E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float64)


def e4m3_val(b: int) -> float:
    s = -1.0 if b & 0x80 else 1.0
    e, m = (b >> 3) & 0xF, b & 7
    if e == 0:
        return s * m / 8 * 2.0**-6
    return s * (1 + m / 8) * 2.0 ** (e - 7)


E4M3 = np.array([e4m3_val(b) for b in range(256)])
E4M3_FINITE = np.array([b for b in range(256) if (b & 0x7F) != 0x7F], dtype=np.uint8)


def bf16_val(h: int) -> float:
    return float(np.array([h << 16], dtype=np.uint32).view(np.float32)[0])


def f16_val(h: int) -> float:
    return float(np.array([h], dtype=np.uint16).view(np.float16)[0])


def bf16_bits(v: float) -> int:  # v must be exactly representable
    b = int(np.array([v], dtype=np.float32).view(np.uint32)[0])
    assert b & 0xFFFF == 0, v
    return b >> 16


def f16_bits(v: float) -> int:
    h = np.array([v], dtype=np.float16)
    assert float(h[0]) == v, v
    return int(h.view(np.uint16)[0])


def b16_val(h, kind16):
    return bf16_val(h) if kind16 == "bf16" else f16_val(h)


# ---- shared memory model --------------------------------------------------------------------------------------


def swz(row_bytes: int, row: int, chunk: int) -> int:
    c = row_bytes // 16
    if c >= 8:
        return chunk ^ (row & 7)
    if c == 4:
        return chunk ^ ((row >> 1) & 3)
    assert c == 2
    return chunk ^ ((row >> 2) & 1)


def tile_addr(base: int, row_bytes: int, row: int, byte: int) -> int:
    return base + row * row_bytes + (swz(row_bytes, row, byte >> 4) << 4) + (byte & 15)


class Smem:
    def __init__(self, n):
        self.b = np.zeros(n, dtype=np.uint8)

    def u32(self, addr):
        assert addr % 4 == 0
        return int(self.b[addr : addr + 4].view(np.uint32)[0])

    def ldsm(self, lane_addrs, nmat):
        """ldmatrix .x{nmat} .b16: matrix j rows come from lanes 8j..8j+7; lane l gets bytes [4(l%4), +4) of
        row l/4 of every matrix."""
        regs = [[0] * nmat for _ in range(32)]
        for j in range(nmat):
            rows = [lane_addrs[8 * j + r] for r in range(8)]
            for r in rows:
                assert r % 16 == 0, "ldmatrix row address must be 16-byte aligned"
            for l in range(32):
                regs[l][j] = self.u32(rows[l // 4] + 4 * (l % 4))
        return regs


# ---- MMA models (D = A[16 x K] * B[K x 8]); fragments per PTX ----------------------------------------------------


def frag_a_rows_bytes(regs):
    """16 x 32-byte A block from the 4 regs of all lanes (byte-level layout shared by all precisions)."""
    a = np.zeros((16, 32), dtype=np.uint8)
    for l in range(32):
        g, t = l // 4, l % 4
        for ri, (row, half) in enumerate([(g, 0), (g + 8, 0), (g, 1), (g + 8, 1)]):
            a[row, half * 16 + 4 * t : half * 16 + 4 * t + 4] = np.array([regs[l][ri]], dtype=np.uint32).view(np.uint8)
    return a


def frag_b_rows_bytes(regs):
    b = np.zeros((8, 32), dtype=np.uint8)
    for l in range(32):
        g, t = l // 4, l % 4
        for ri in range(2):
            b[g, ri * 16 + 4 * t : ri * 16 + 4 * t + 4] = np.array([regs[l][ri]], dtype=np.uint32).view(np.uint8)
    return b


def decode_rows(byts, fmt):
    """[rows, 32 bytes] -> [rows, K] values in K order."""
    if fmt in ("bf16", "f16"):
        h = byts.view(np.uint16)
        f = bf16_val if fmt == "bf16" else f16_val
        return np.vectorize(f)(h).astype(np.float64)
    if fmt == "e4m3":
        return E4M3[byts]
    if fmt == "s8":
        return byts.view(np.int8).astype(np.float64)
    if fmt == "e2m1":
        lo, hi = byts & 0xF, byts >> 4
        out = np.empty((byts.shape[0], byts.shape[1] * 2))
        out[:, 0::2], out[:, 1::2] = E2M1[lo], E2M1[hi]
        return out
    raise ValueError(fmt)


def mma(acc, a_regs, b_regs, fmt, sfa=None, sfb=None):
    """acc: [32 lanes][4]; a_regs [32][4]; b_regs [32][2]."""
    A = decode_rows(frag_a_rows_bytes(a_regs), fmt)  # [16, K]
    B = decode_rows(frag_b_rows_bytes(b_regs), fmt)  # [8, K]
    if sfa is not None:  # NVFP4: row r scale bytes from lane 4*(r%8) + r//8, column n from lane 4n
        for r in range(16):
            reg = sfa[4 * (r % 8) + r // 8]
            for j in range(4):
                A[r, 16 * j : 16 * j + 16] *= E4M3[(reg >> (8 * j)) & 0xFF]
        for n in range(8):
            reg = sfb[4 * n]
            for j in range(4):
                B[n, 16 * j : 16 * j + 16] *= E4M3[(reg >> (8 * j)) & 0xFF]
    D = A @ B.T  # [16, 8]
    for l in range(32):
        g, t = l // 4, l % 4
        for e in range(4):
            acc[l][e] += D[g + (e >> 1) * 8, 2 * t + (e & 1)]


# ---- conversions used by the mixed path ----------------------------------------------------------------------


def cvt_pair(two_bytes_or_byte, src, dst):
    """Packed 2-element conversion -> 32-bit word with the low element in the low half."""
    if src == "e4m3":
        vals = [E4M3[two_bytes_or_byte & 0xFF], E4M3[(two_bytes_or_byte >> 8) & 0xFF]]
    else:
        vals = [E2M1[two_bytes_or_byte & 0xF], E2M1[(two_bytes_or_byte >> 4) & 0xF]]
    bits = bf16_bits if dst == "bf16" else f16_bits
    return bits(vals[0]) | (bits(vals[1]) << 16)


def mul_pair(a, b, dst):
    val = bf16_val if dst == "bf16" else f16_val
    bits = bf16_bits if dst == "bf16" else f16_bits
    lo = val(a & 0xFFFF) * val(b & 0xFFFF)
    hi = val(a >> 16) * val(b >> 16)
    return bits(lo) | (bits(hi) << 16)


# ---- the kernel ------------------------------------------------------------------------------------------------


def run_kernel(kind, cfg, P, Q, K, A, B, ASF, BSF, ksplit, stages, swap, rs, cs, alpha):
    a_bits, b_bits, BK, mixed, bscaled, wbscaled, int_acc, t16 = KINDS[kind]
    BP, BQ, WP, WQ = CFGS[cfg]
    MP, NQ, WARPS_P = WP // 16, WQ // 8, BP // WP
    warps = (BP // WP) * (BQ // WQ)
    threads = warps * 32
    a_row, b_row = BK * a_bits // 8, BK * b_bits // 8
    a_sf = BK // 16 if (bscaled or wbscaled) else 0
    b_sf = BK // 16 if bscaled else 0
    a_off, b_off = 0, BP * a_row
    asf_off = b_off + BQ * b_row
    bsf_off = asf_off + BP * a_sf
    stage_bytes = (bsf_off + BQ * b_sf + 127) // 128 * 128
    a_kbytes, b_kbytes, sf_kbytes = K * a_bits // 8, K * b_bits // 8, K // 16
    nk = -(-K // BK)
    fmt = {"bf16": "bf16", "f16": "f16", "fp8": "e4m3", "int8": "s8", "nvfp4": "e2m1"}.get(kind)

    Y = np.zeros((Q, P) if swap else (P, Q))
    WS = np.zeros((ksplit, P, Q))
    for bx, by in itertools.product(range(-(-P // BP)), range(-(-Q // BQ))):
        p0, q0 = bx * BP, by * BQ
        partial_sum = np.zeros((BP, BQ))
        for split in range(ksplit):
            kt_beg, kt_end = nk * split // ksplit, nk * (split + 1) // ksplit
            sm = Smem(stages * stage_bytes)

            def load_stage(slot, kt):
                st = slot * stage_bytes
                for (rows, row_b, off, src, r0, lim, kbytes) in [
                    (BP, a_row, a_off, A, p0, P, a_kbytes),
                    (BQ, b_row, b_off, B, q0, Q, b_kbytes),
                ]:
                    ch = row_b // 16
                    for i in range(rows * ch):
                        r, c = i // ch, i % ch
                        gr, kb = r0 + r, kt * row_b + c * 16
                        dst = tile_addr(st + off, row_b, r, c * 16)
                        ok = gr < lim and kb < kbytes
                        sm.b[dst : dst + 16] = src[gr, kb : kb + 16] if ok else 0
                for (rows, sfb_, off, src, r0, lim) in [(BP, a_sf, asf_off, ASF, p0, P), (BQ, b_sf, bsf_off, BSF, q0, Q)]:
                    if sfb_ == 0:
                        continue
                    ch = sfb_ // 4
                    for i in range(rows * ch):
                        r, c = i // ch, i % ch
                        gr, kb = r0 + r, kt * sfb_ + c * 4
                        dst = st + off + r * sfb_ + c * 4
                        ok = gr < lim and kb < sf_kbytes
                        sm.b[dst : dst + 4] = src[gr, kb : kb + 4] if ok else 0

            acc = [[[[0.0] * 4 for _ in range(32)] for _ in range(NQ)] for _ in range(MP)]
            acc = {w: [[[[0.0] * 4 for _ in range(32)] for _ in range(NQ)] for _ in range(MP)] for w in range(warps)}

            def compute_stage(slot):
                st = slot * stage_bytes
                sa, sb = st + a_off, st + b_off
                for w in range(warps):
                    wp0, wq0 = (w % WARPS_P) * WP, (w // WARPS_P) * WQ
                    lanes = range(32)
                    a_lrow = [(l & 7) + ((l >> 3) & 1) * 8 for l in lanes]
                    a_lch = [l >> 4 for l in lanes]
                    b_lrow = [(l & 7) + (l >> 4) * 8 for l in lanes]
                    b_lch = [(l >> 3) & 1 for l in lanes]
                    if not mixed:
                        for ks in range(a_row // 32):
                            af = [sm.ldsm([tile_addr(sa, a_row, wp0 + i * 16 + a_lrow[l], ks * 32 + a_lch[l] * 16)
                                           for l in lanes], 4) for i in range(MP)]
                            bf = [None] * NQ
                            if NQ == 1:
                                r = sm.ldsm([tile_addr(sb, b_row, wq0 + (l & 7), ks * 32 + ((l >> 3) & 1) * 16)
                                             for l in lanes], 2)
                                bf[0] = r
                            else:
                                for j in range(0, NQ, 2):
                                    r = sm.ldsm([tile_addr(sb, b_row, wq0 + j * 8 + b_lrow[l], ks * 32 + b_lch[l] * 16)
                                                 for l in lanes], 4)
                                    bf[j] = [[r[l][0], r[l][1]] for l in lanes]
                                    bf[j + 1] = [[r[l][2], r[l][3]] for l in lanes]
                            for i in range(MP):
                                for j in range(NQ):
                                    sfa = sfb = None
                                    if bscaled:
                                        sfa = [sm.u32(st + asf_off + (wp0 + i * 16 + l // 4 + 8 * (l & 1)) * a_sf + ks * 4)
                                               for l in lanes]
                                        sfb = [sm.u32(st + bsf_off + (wq0 + j * 8 + l // 4) * b_sf + ks * 4) for l in lanes]
                                    mma(acc[w][i][j], af[i], bf[j], fmt, sfa, sfb)
                    else:
                        fp4 = a_bits == 4
                        groups, jn = a_row // 32, (4 if fp4 else 2)
                        gk = 16 * jn
                        for gi in range(groups):
                            aw = [sm.ldsm([tile_addr(sa, a_row, wp0 + i * 16 + a_lrow[l], gi * 32 + a_lch[l] * 16)
                                           for l in lanes], 4) for i in range(MP)]
                            sc = None
                            if wbscaled:
                                sc = {}
                                for i in range(MP):
                                    for l in lanes:
                                        g, t = l // 4, l % 4
                                        for h in range(2):
                                            s4 = sm.u32(st + asf_off + (wp0 + i * 16 + g + 8 * h) * a_sf + gi * 4)
                                            b0 = (s4 >> (8 * (t >> 1))) & 0xFF
                                            b1 = (s4 >> (8 * (2 + (t >> 1)))) & 0xFF
                                            sc[(i, l, h, 0)] = cvt_pair(b0 | (b0 << 8), "e4m3", t16)
                                            sc[(i, l, h, 1)] = cvt_pair(b1 | (b1 << 8), "e4m3", t16)
                            for jm in range(jn):
                                bfr = []
                                for j in range(NQ):
                                    regs = []
                                    for l in lanes:
                                        g, t = l // 4, l % 4
                                        kel = ((jm >> 1) * 32 + 8 * t + 4 * (jm & 1)) if fp4 else (16 * jm + 4 * t)
                                        ad = tile_addr(sb, b_row, wq0 + j * 8 + g, (gi * gk + kel) * 2)
                                        assert ad % 8 == 0
                                        regs.append([sm.u32(ad), sm.u32(ad + 4)])
                                    bfr.append(regs)
                                for i in range(MP):
                                    afr = []
                                    for l in lanes:
                                        if not fp4:
                                            lo, hi = aw[i][l][jm * 2], aw[i][l][jm * 2 + 1]
                                            afr.append([cvt_pair(lo & 0xFFFF, "e4m3", t16), cvt_pair(hi & 0xFFFF, "e4m3", t16),
                                                        cvt_pair(lo >> 16, "e4m3", t16), cvt_pair(hi >> 16, "e4m3", t16)])
                                        else:
                                            half, sub = jm >> 1, jm & 1
                                            lo, hi = aw[i][l][half * 2], aw[i][l][half * 2 + 1]
                                            sh = 16 * sub
                                            f = [cvt_pair((lo >> sh) & 0xFF, "e2m1", t16), cvt_pair((hi >> sh) & 0xFF, "e2m1", t16),
                                                 cvt_pair((lo >> (sh + 8)) & 0xFF, "e2m1", t16),
                                                 cvt_pair((hi >> (sh + 8)) & 0xFF, "e2m1", t16)]
                                            sg, sg8 = sc[(i, l, 0, half)], sc[(i, l, 1, half)]
                                            f = [mul_pair(f[0], sg, t16), mul_pair(f[1], sg8, t16),
                                                 mul_pair(f[2], sg, t16), mul_pair(f[3], sg8, t16)]
                                            afr.append(f)
                                    for j in range(NQ):
                                        mma(acc[w][i][j], afr, bfr[j], t16)

            ntiles = kt_end - kt_beg
            # the emulator runs the pipeline in program order; slot reuse is the kernel's (it % stages)
            for s in range(stages - 1):
                if s < ntiles:
                    load_stage(s, kt_beg + s)
            for it in range(ntiles):
                pf = it + stages - 1
                if pf < ntiles:
                    load_stage(pf % stages, kt_beg + pf)
                compute_stage(it % stages)
            # epilogue / partials
            for w in range(warps):
                wp0, wq0 = (w % WARPS_P) * WP, (w // WARPS_P) * WQ
                for i, j, l, e in itertools.product(range(MP), range(NQ), range(32), range(4)):
                    g, t = l // 4, l % 4
                    p = p0 + wp0 + i * 16 + g + (e >> 1) * 8
                    q = q0 + wq0 + j * 8 + 2 * t + (e & 1)
                    if p < P and q < Q:
                        WS[split, p, q] = acc[w][i][j][l][e]
        for p in range(p0, min(P, p0 + BP)):
            for q in range(q0, min(Q, q0 + BQ)):
                s = sum(WS[k, p, q] for k in range(ksplit))
                v = s * alpha * rs[p] * cs[q]
                if swap:
                    Y[q, p] = v
                else:
                    Y[p, q] = v
    return Y


# ---- random operands + reference ------------------------------------------------------------------------------


def make_operand(rng, rows, K, fmt):
    """Returns (bytes [rows, K*bits/8] padded to 16-byte rows, values [rows, K])."""
    if fmt in ("bf16", "f16"):
        v = rng.standard_normal((rows, K)).astype(np.float32)
        if fmt == "bf16":
            bits = (v.view(np.uint32) >> 16).astype(np.uint16)
            vals = np.vectorize(bf16_val)(bits)
        else:
            h = v.astype(np.float16)
            bits, vals = h.view(np.uint16), h.astype(np.float64)
        return bits.view(np.uint8).reshape(rows, -1), vals
    if fmt == "e4m3":
        b = rng.choice(E4M3_FINITE, size=(rows, K))
        return b.astype(np.uint8), E4M3[b]
    if fmt == "s8":
        b = rng.integers(-128, 128, size=(rows, K)).astype(np.int8)
        return b.view(np.uint8), b.astype(np.float64)
    if fmt == "e2m1":
        n = rng.integers(0, 16, size=(rows, K)).astype(np.uint8)
        packed = (n[:, 0::2] | (n[:, 1::2] << 4)).astype(np.uint8)
        return packed, E2M1[n]
    raise ValueError(fmt)


def make_scales(rng, rows, K):
    b = rng.integers(0x18, 0x48, size=(rows, K // 16)).astype(np.uint8)  # moderate positive e4m3 scales
    return b, E4M3[b]


def check(kind, cfg, P, Q, K, ksplit, stages, swap, seed=0):
    a_bits, b_bits, BK, mixed, bscaled, wbscaled, int_acc, t16 = KINDS[kind]
    rng = np.random.default_rng(seed)
    fa = {"bf16": "bf16", "f16": "f16", "fp8": "e4m3", "int8": "s8", "nvfp4": "e2m1",
          "w8a16_bf16": "e4m3", "w8a16_f16": "e4m3", "w4a16_bf16": "e2m1", "w4a16_f16": "e2m1"}[kind]
    fb = t16 if mixed else fa
    A, av = make_operand(rng, P, K, fa)
    B, bv = make_operand(rng, Q, K, fb)
    ASF = BSF = None
    if bscaled or wbscaled:
        ASF, asv = make_scales(rng, P, K)
        av = av * np.repeat(asv, 16, axis=1)
    if bscaled:
        BSF, bsv = make_scales(rng, Q, K)
        bv = bv * np.repeat(bsv, 16, axis=1)
    rs = rng.uniform(0.5, 2, P)
    cs = rng.uniform(0.5, 2, Q)
    alpha = 0.75
    ref = (av @ bv.T) * alpha * rs[:, None] * cs[None, :]
    Y = run_kernel(kind, cfg, P, Q, K, A, B, ASF, BSF, ksplit, stages, swap, rs, cs, alpha)
    got = Y.T if swap else Y
    err = np.abs(got - ref).max() / max(1e-30, np.abs(ref).max())
    ok = err < 1e-12
    print(f"{'ok ' if ok else 'BAD'} {kind:11s} cfg{cfg} P={P} Q={Q} K={K} ksplit={ksplit} stages={stages} "
          f"swap={int(swap)} rel_err={err:.1e}", flush=True)
    return ok


def main() -> int:
    bad = 0
    cases = []
    for kind in KINDS:
        a_bits, b_bits, BK, *_ = KINDS[kind]
        kal = 64 if kind in ("nvfp4", "w4a16_bf16", "w4a16_f16") else 16
        K = 2 * BK + kal  # a partial last K tile
        cases.append((kind, 0, 70, 5, K, 1, 3, True))
        cases.append((kind, 1, 130, 13, K, 2, 2, True))
        cases.append((kind, 4, 67, 70, K, 3, 4, False))
    for kind in ("bf16", "fp8", "nvfp4", "w4a16_bf16"):
        cases.append((kind, 2, 40, 33, 3 * KINDS[kind][2], 1, 2, True))
        cases.append((kind, 3, 129, 65, KINDS[kind][2], 1, 2, False))
        cases.append((kind, 6, 130, 129, KINDS[kind][2] + 64, 2, 2, False))
    cases.append(("bf16", 5, 129, 130, 64, 1, 2, False))
    cases.append(("int8", 7, 257, 129, 128, 1, 2, False))
    for c in cases:
        bad += not check(*c)
    print("all ok" if bad == 0 else f"{bad} FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
