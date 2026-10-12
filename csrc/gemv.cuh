// SPDX-License-Identifier: Apache-2.0
//
// Skinny BF16/FP16 GEMM for decode-sized batches: y[M, N] = x[M, K] @ w[N, K]^T, M <= 16.
//
// The weight is streamed once from DRAM in 16-byte vectors (L1 bypass, 256B L2 prefetch); x is small and
// re-read through L1. Accumulation is fp32 FMA in a fixed order, so the result for one row of x does not
// depend on M or on the tiling knobs (rows per thread group, unroll, warps per block):
//   * lane `sl` of a row group of LPR lanes accumulates the 16B vectors v = vbeg + sl + LPR*j, j = 0, 1, ...
//     in increasing j, elements 0..7 of each vector in order;
//   * the LPR lane sums are combined by an xor butterfly over LPR lanes;
//   * with split-K, the KSPLIT partial sums are added in split order 0..KSPLIT-1 by the last block to arrive.
// The order is therefore a function of (K, LPR, KSPLIT) only.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include "params.h"

namespace llg {

__device__ __forceinline__ uint4 ld_weight(const __nv_bfloat16* p) {
  uint4 r;
  asm("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0, %1, %2, %3}, [%4];"
      : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
      : "l"(p));
  return r;
}

// Exact bf16 -> fp32 widening of the two halves of a 32-bit word (low half = lower address).
__device__ __forceinline__ void widen(uint32_t u, float& lo, float& hi) {
  lo = __uint_as_float(u << 16);
  hi = __uint_as_float(u & 0xffff0000u);
}

// Exact f16 -> fp32 widening of the two halves of a 32-bit word.
__device__ __forceinline__ void widen_h(uint32_t u, float& lo, float& hi) {
  const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&u));
  lo = f.x;
  hi = f.y;
}

template <bool F16>
__device__ __forceinline__ void widen8(const uint4& v, float (&f)[8]) {
  if constexpr (F16) {
    widen_h(v.x, f[0], f[1]);
    widen_h(v.y, f[2], f[3]);
    widen_h(v.z, f[4], f[5]);
    widen_h(v.w, f[6], f[7]);
  } else {
    widen(v.x, f[0], f[1]);
    widen(v.y, f[2], f[3]);
    widen(v.z, f[4], f[5]);
    widen(v.w, f[6], f[7]);
  }
}

template <bool F16>
__device__ __forceinline__ void store_out(const Params& p, int m, int n, float v) {
  if (p.out_fp32) {
    static_cast<float*>(p.y)[(int64_t)m * p.ldy + n] = v;
  } else if constexpr (F16) {
    static_cast<__half*>(p.y)[(int64_t)m * p.ldy + n] = __float2half_rn(v);
  } else {
    static_cast<__nv_bfloat16*>(p.y)[(int64_t)m * p.ldy + n] = __float2bfloat16_rn(v);
  }
}

// MT: token capacity (runtime M <= MT). R: rows per row group. U: vectors in flight per row and lane.
// LPR: lanes per row group (8, 16 or 32); a warp holds 32 / LPR row groups.
template <int MT, int R, int U, int LPR, bool F16>
__global__ void __launch_bounds__(256) gemv_kernel(const Params p) {
  constexpr int GROUPS = 32 / LPR;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int sub = lane / LPR;
  const int sl = lane % LPR;
  const int rows_per_block = (blockDim.x >> 5) * GROUPS * R;
  const int block_row0 = blockIdx.x * rows_per_block;
  const int row0 = block_row0 + (warp * GROUPS + sub) * R;

  const int kv = p.K >> 3;  // 16-byte vectors per row
  const int split = blockIdx.y;
  const int vbeg = (int)((int64_t)kv * split / p.ksplit);
  const int vend = (int)((int64_t)kv * (split + 1) / p.ksplit);

  const __nv_bfloat16* const xp = static_cast<const __nv_bfloat16*>(p.x);
  const __nv_bfloat16* const wp = static_cast<const __nv_bfloat16*>(p.w);
  const __nv_bfloat16* wrow[R];
  bool rvalid[R];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    rvalid[r] = row0 + r < p.N;
    wrow[r] = wp + (int64_t)(rvalid[r] ? row0 + r : 0) * p.ldw;
  }

  float acc[R][MT];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[r][m] = 0.f;

  for (int v0 = vbeg + sl; v0 < vend; v0 += LPR * U) {
    uint4 wv[U][R];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int v = v0 + u * LPR;
#pragma unroll
      for (int r = 0; r < R; ++r) {
        wv[u][r] = (v < vend && rvalid[r]) ? ld_weight(wrow[r] + (int64_t)v * 8) : make_uint4(0, 0, 0, 0);
      }
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int v = v0 + u * LPR;
      if (v < vend) {
        float wf[R][8];
#pragma unroll
        for (int r = 0; r < R; ++r) widen8<F16>(wv[u][r], wf[r]);
#pragma unroll
        for (int m = 0; m < MT; ++m) {
          if (m < p.M) {
            float xf[8];
            widen8<F16>(__ldg(reinterpret_cast<const uint4*>(xp + (int64_t)m * p.ldx + (int64_t)v * 8)), xf);
#pragma unroll
            for (int r = 0; r < R; ++r)
#pragma unroll
              for (int e = 0; e < 8; ++e) acc[r][m] = __fmaf_rn(wf[r][e], xf[e], acc[r][m]);
          }
        }
      }
    }
  }

  // Butterfly over the LPR lanes of this row group; every lane ends with the same sums.
#pragma unroll
  for (int off = LPR / 2; off >= 1; off >>= 1)
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int m = 0; m < MT; ++m) acc[r][m] += __shfl_xor_sync(0xffffffffu, acc[r][m], off);

  if (p.ksplit == 1) {
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int m = 0; m < MT; ++m)
        if (sl == (r * MT + m) % LPR && m < p.M && rvalid[r]) store_out<F16>(p, m, row0 + r, acc[r][m]);
    return;
  }

  // Split-K: publish partials, the last block of this column adds them in split order.
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int m = 0; m < MT; ++m)
      if (sl == (r * MT + m) % LPR && m < p.M && rvalid[r])
        p.ws[((int64_t)split * p.M + m) * p.N + row0 + r] = acc[r][m];

  __shared__ unsigned is_last;
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned prev = atomicAdd(&p.counters[blockIdx.x], 1u);
    is_last = prev == (unsigned)(p.ksplit - 1);
  }
  __syncthreads();
  if (!is_last) return;
  __threadfence();

  const int nrows = min(rows_per_block, p.N - block_row0);
  for (int i = threadIdx.x; i < nrows * p.M; i += blockDim.x) {
    const int m = i / nrows;
    const int n = block_row0 + i % nrows;
    float s = __ldcg(&p.ws[(int64_t)m * p.N + n]);
    for (int k = 1; k < p.ksplit; ++k) s += __ldcg(&p.ws[((int64_t)k * p.M + m) * p.N + n]);
    store_out<F16>(p, m, n, s);
  }
  if (threadIdx.x == 0) p.counters[blockIdx.x] = 0u;
}

}  // namespace llg
