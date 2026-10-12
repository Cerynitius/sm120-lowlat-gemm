// SPDX-License-Identifier: Apache-2.0
//
// Tensor-core GEMM for SM120, all supported precisions, any M:
//
//   C[p, q] = sum_k A[p, k] * B[q, k]          (both operands K-contiguous, i.e. "TN")
//
// The host maps (activation x, weight w) onto (A, B): for few tokens the weight is A (its rows fill the 16-row MMA
// M dimension, tokens use the 8-wide N dimension) and the result is stored transposed; for many tokens x is A.
//
// One pipeline serves every precision. At the byte level the m16n8k16 (16-bit), m16n8k32 (8-bit) and
// m16n8k64 (packed fp4) MMAs consume the same fragment: per k step each row contributes 32 bytes, thread
// (groupID g, threadID-in-group t) holds bytes [4t, 4t+4) and [16+4t, 16+4t+4) of rows g and g+8 for A and of
// row g for B. Tiles are staged with cp.async into 128-byte-row (or 64/256-byte-row) shared buffers with an XOR
// swizzle of the 16-byte chunks, and loaded with ldmatrix.
//
// Mixed precision (W8A16, W4A16: low-bit weight as A, 16-bit activation as B) runs 16-bit MMAs. The weight
// fragment is loaded as bytes and widened exactly in registers; the K positions are permuted consistently on both
// operands (as in Marlin), so the activation fragment for one MMA is 4 consecutive K elements of one row.
#pragma once

#include <type_traits>

#include "ptx.cuh"
#include "tc_params.h"

namespace llg {

// ---------------------------------------------------------------------------------------------------------------
// Precision traits

template <int KIND>
struct Kind;

struct KindBase {
  static constexpr bool kMixed = false;
  static constexpr bool kBlockScaled = false;  // NVFP4 x NVFP4
  static constexpr bool kWeightBlockScaled = false;  // W4A16
  static constexpr bool kIntAcc = false;
};

template <>
struct Kind<KIND_BF16> : KindBase {
  static constexpr int kABits = 16, kBBits = 16, kBK = 64;
};
template <>
struct Kind<KIND_F16> : KindBase {
  static constexpr int kABits = 16, kBBits = 16, kBK = 64;
};
template <>
struct Kind<KIND_FP8> : KindBase {
  static constexpr int kABits = 8, kBBits = 8, kBK = 128;
};
template <>
struct Kind<KIND_INT8> : KindBase {
  static constexpr int kABits = 8, kBBits = 8, kBK = 128;
  static constexpr bool kIntAcc = true;
};
template <>
struct Kind<KIND_NVFP4> : KindBase {
  static constexpr int kABits = 4, kBBits = 4, kBK = 256;
  static constexpr bool kBlockScaled = true;
};
template <>
struct Kind<KIND_W8A16_BF16> : KindBase {
  static constexpr int kABits = 8, kBBits = 16, kBK = 128;
  static constexpr bool kMixed = true;
};
template <>
struct Kind<KIND_W8A16_F16> : Kind<KIND_W8A16_BF16> {};
template <>
struct Kind<KIND_W4A16_BF16> : KindBase {
  static constexpr int kABits = 4, kBBits = 16, kBK = 128;
  static constexpr bool kMixed = true;
  static constexpr bool kWeightBlockScaled = true;
};
template <>
struct Kind<KIND_W4A16_F16> : Kind<KIND_W4A16_BF16> {};

// ---------------------------------------------------------------------------------------------------------------
// Tile configurations (BP x BQ block tile, WP x WQ warp tile)

template <int CFG>
struct Cfg;
#define LLG_CFG(ID, BP_, BQ_, WP_, WQ_)                                    \
  template <>                                                              \
  struct Cfg<ID> {                                                         \
    static constexpr int BP = BP_, BQ = BQ_, WP = WP_, WQ = WQ_;           \
    static constexpr int kWarps = (BP_ / WP_) * (BQ_ / WQ_);               \
    static constexpr int kThreads = kWarps * 32;                           \
  };
LLG_CFG(0, 64, 8, 16, 8)       // 4 warps   decode-sized (weight as A, <= 8 tokens per tile)
LLG_CFG(1, 128, 16, 32, 16)    // 4 warps
LLG_CFG(2, 128, 32, 32, 32)    // 4 warps
LLG_CFG(3, 128, 64, 64, 32)    // 4 warps
LLG_CFG(4, 64, 64, 32, 32)     // 4 warps
LLG_CFG(5, 128, 128, 64, 64)   // 4 warps
LLG_CFG(6, 128, 128, 64, 32)   // 8 warps
LLG_CFG(7, 256, 128, 64, 64)   // 8 warps
#undef LLG_CFG

// ---------------------------------------------------------------------------------------------------------------
// Shared-memory geometry

template <int ROW_BYTES>
__device__ __forceinline__ int swz(int row, int chunk) {
  constexpr int C = ROW_BYTES / 16;
  if constexpr (C >= 8) {
    return chunk ^ (row & 7);
  } else if constexpr (C == 4) {
    return chunk ^ ((row >> 1) & 3);
  } else {
    static_assert(C == 2, "row must be 32, 64 or >= 128 bytes");
    return chunk ^ ((row >> 2) & 1);
  }
}

template <int ROW_BYTES>
__device__ __forceinline__ uint32_t tile_addr(uint32_t base, int row, int byte) {
  return base + row * ROW_BYTES + (swz<ROW_BYTES>(row, byte >> 4) << 4) + (byte & 15);
}

template <int KIND, int CFG>
struct Layout {
  using K = Kind<KIND>;
  using C = Cfg<CFG>;
  static constexpr int kARow = K::kBK * K::kABits / 8;  // bytes per A row per stage
  static constexpr int kBRow = K::kBK * K::kBBits / 8;
  static constexpr int kASF = K::kBlockScaled || K::kWeightBlockScaled ? K::kBK / 16 : 0;  // scale bytes/row/stage
  static constexpr int kBSF = K::kBlockScaled ? K::kBK / 16 : 0;
  static constexpr int kAOff = 0;
  static constexpr int kBOff = kAOff + C::BP * kARow;
  static constexpr int kASFOff = kBOff + C::BQ * kBRow;
  static constexpr int kBSFOff = kASFOff + C::BP * kASF;
  static constexpr int kStageBytes = (kBSFOff + C::BQ * kBSF + 127) / 128 * 128;
};

constexpr int kMaxSmem = 99 * 1024;

// ---------------------------------------------------------------------------------------------------------------
// Epilogue

__device__ __forceinline__ float load_vec(const float* p, int len, int i) {
  return len == 0 ? 1.f : (len == 1 ? p[0] : p[i]);
}

__device__ __forceinline__ float load_bias(const TcParams& prm, int i) {
  switch (prm.bias_dtype) {
    case DT_BF16: return __bfloat162float(static_cast<const __nv_bfloat16*>(prm.bias)[i]);
    case DT_F16: return __half2float(static_cast<const __half*>(prm.bias)[i]);
    case DT_F32: return static_cast<const float*>(prm.bias)[i];
    default: return 0.f;
  }
}

__device__ __forceinline__ void epilogue_store(const TcParams& prm, float alpha, int p, int q, float acc) {
  float v = acc * alpha * load_vec(prm.row_scale, prm.row_scale_len, p) * load_vec(prm.col_scale, prm.col_scale_len, q);
  if (prm.bias_dtype != DT_NONE) v += load_bias(prm, prm.bias_on_p ? p : q);
  const int64_t idx = prm.swap ? (int64_t)q * prm.ldy + p : (int64_t)p * prm.ldy + q;
  switch (prm.out_dtype) {
    case DT_BF16: static_cast<__nv_bfloat16*>(prm.y)[idx] = __float2bfloat16_rn(v); break;
    case DT_F16: static_cast<__half*>(prm.y)[idx] = __float2half_rn(v); break;
    default: static_cast<float*>(prm.y)[idx] = v; break;
  }
}

// ---------------------------------------------------------------------------------------------------------------
// The kernel

template <int KIND, int CFG>
__global__ void __launch_bounds__(Cfg<CFG>::kThreads) tc_gemm_kernel(const TcParams prm) {
  using K = Kind<KIND>;
  using C = Cfg<CFG>;
  using L = Layout<KIND, CFG>;
  constexpr int BP = C::BP, BQ = C::BQ, WP = C::WP, WQ = C::WQ;
  constexpr int MP = WP / 16, NQ = WQ / 8;  // MMAs per warp along P and Q
  constexpr int WARPS_P = BP / WP;
  using Acc = typename std::conditional<K::kIntAcc, int32_t, float>::type;

  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t sbase = smem_addr(smem);

  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int wp0 = (warp % WARPS_P) * WP, wq0 = (warp / WARPS_P) * WQ;
  const int p0 = blockIdx.x * BP, q0 = blockIdx.y * BQ;
  const int split = blockIdx.z, ksplit = gridDim.z;

  const uint8_t* const A = static_cast<const uint8_t*>(prm.a);
  const uint8_t* const B = static_cast<const uint8_t*>(prm.b);
  const uint8_t* const ASF = static_cast<const uint8_t*>(prm.a_sf);
  const uint8_t* const BSF = static_cast<const uint8_t*>(prm.b_sf);
  const int a_kbytes = (int)((int64_t)prm.K * K::kABits / 8);
  const int b_kbytes = (int)((int64_t)prm.K * K::kBBits / 8);
  const int sf_kbytes = prm.K / 16;

  const int nk = (prm.K + K::kBK - 1) / K::kBK;
  const int kt_beg = (int)((int64_t)nk * split / ksplit);
  const int kt_end = (int)((int64_t)nk * (split + 1) / ksplit);
  const int stages = prm.stages;

  // ---- producer: one stage = A tile, B tile (+ scale tiles) for K tile kt ----
  auto load_stage = [&](int slot, int kt) {
    const uint32_t st = sbase + slot * L::kStageBytes;
    {
      constexpr int CH = L::kARow / 16, N = BP * CH;
      for (int i = tid; i < N; i += C::kThreads) {
        const int r = i / CH, c = i % CH;
        const int gr = p0 + r, kb = kt * L::kARow + c * 16;
        const bool ok = gr < prm.P && kb < a_kbytes;
        cp_async16(tile_addr<L::kARow>(st + L::kAOff, r, c * 16), ok ? A + (int64_t)gr * prm.lda + kb : A, ok ? 16 : 0);
      }
    }
    {
      constexpr int CH = L::kBRow / 16, N = BQ * CH;
      for (int i = tid; i < N; i += C::kThreads) {
        const int r = i / CH, c = i % CH;
        const int gr = q0 + r, kb = kt * L::kBRow + c * 16;
        const bool ok = gr < prm.Q && kb < b_kbytes;
        cp_async16(tile_addr<L::kBRow>(st + L::kBOff, r, c * 16), ok ? B + (int64_t)gr * prm.ldb + kb : B, ok ? 16 : 0);
      }
    }
    if constexpr (L::kASF > 0) {  // 4-byte pieces: K % 64 == 0 is enough
      constexpr int CH = L::kASF / 4, N = BP * CH;
      for (int i = tid; i < N; i += C::kThreads) {
        const int r = i / CH, c = i % CH;
        const int gr = p0 + r, kb = kt * L::kASF + c * 4;
        const bool ok = gr < prm.P && kb < sf_kbytes;
        cp_async4(st + L::kASFOff + r * L::kASF + c * 4, ok ? ASF + (int64_t)gr * prm.lda_sf + kb : ASF, ok ? 4 : 0);
      }
    }
    if constexpr (L::kBSF > 0) {
      constexpr int CH = L::kBSF / 4, N = BQ * CH;
      for (int i = tid; i < N; i += C::kThreads) {
        const int r = i / CH, c = i % CH;
        const int gr = q0 + r, kb = kt * L::kBSF + c * 4;
        const bool ok = gr < prm.Q && kb < sf_kbytes;
        cp_async4(st + L::kBSFOff + r * L::kBSF + c * 4, ok ? BSF + (int64_t)gr * prm.ldb_sf + kb : BSF, ok ? 4 : 0);
      }
    }
  };

  Acc acc[MP][NQ][4];
#pragma unroll
  for (int i = 0; i < MP; ++i)
#pragma unroll
    for (int j = 0; j < NQ; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[i][j][e] = 0;

  // ldmatrix lane -> (row, 16-byte chunk) of the 16x32-byte A block and of a 16x32-byte pair of B blocks
  const int a_lrow = (lane & 7) + ((lane >> 3) & 1) * 8, a_lchunk = lane >> 4;
  const int b_lrow = (lane & 7) + (lane >> 4) * 8, b_lchunk = (lane >> 3) & 1;

  // ---- consumer: all MMAs of one stage ----
  auto compute_stage = [&](int slot) {
    const uint32_t st = sbase + slot * L::kStageBytes;
    const uint32_t sa = st + L::kAOff, sb = st + L::kBOff;
    if constexpr (!K::kMixed) {
      constexpr int KSTEPS = L::kARow / 32;
#pragma unroll
      for (int ks = 0; ks < KSTEPS; ++ks) {
        uint32_t af[MP][4], bf[NQ][2];
#pragma unroll
        for (int i = 0; i < MP; ++i)
          ldsm_x4(tile_addr<L::kARow>(sa, wp0 + i * 16 + a_lrow, ks * 32 + a_lchunk * 16), af[i][0], af[i][1], af[i][2],
                  af[i][3]);
        if constexpr (NQ == 1) {
          ldsm_x2(tile_addr<L::kBRow>(sb, wq0 + (lane & 7), ks * 32 + ((lane >> 3) & 1) * 16), bf[0][0], bf[0][1]);
        } else {
#pragma unroll
          for (int j = 0; j < NQ; j += 2)
            ldsm_x4(tile_addr<L::kBRow>(sb, wq0 + j * 8 + b_lrow, ks * 32 + b_lchunk * 16), bf[j][0], bf[j][1],
                    bf[j + 1][0], bf[j + 1][1]);
        }
        uint32_t sfa[MP], sfb[NQ];
        if constexpr (K::kBlockScaled) {
#pragma unroll
          for (int i = 0; i < MP; ++i) sfa[i] = lds32(st + L::kASFOff + (wp0 + i * 16 + g + 8 * (lane & 1)) * L::kASF + ks * 4);
#pragma unroll
          for (int j = 0; j < NQ; ++j) sfb[j] = lds32(st + L::kBSFOff + (wq0 + j * 8 + g) * L::kBSF + ks * 4);
        }
#pragma unroll
        for (int i = 0; i < MP; ++i)
#pragma unroll
          for (int j = 0; j < NQ; ++j) {
            if constexpr (KIND == KIND_BF16) mma_bf16(acc[i][j], af[i], bf[j][0], bf[j][1]);
            else if constexpr (KIND == KIND_F16) mma_f16(acc[i][j], af[i], bf[j][0], bf[j][1]);
            else if constexpr (KIND == KIND_FP8) mma_e4m3(acc[i][j], af[i], bf[j][0], bf[j][1]);
            else if constexpr (KIND == KIND_INT8) mma_s8(acc[i][j], af[i], bf[j][0], bf[j][1]);
            else mma_nvfp4(acc[i][j], af[i], bf[j][0], bf[j][1], sfa[i], sfb[j]);
          }
      }
    } else {
      // Mixed: a group = 32 weight bytes per row = 32 (fp8) or 64 (fp4) K = 2 or 4 MMAs of k16.
      constexpr bool kFp4 = K::kABits == 4;
      constexpr bool kBf16 = KIND == KIND_W8A16_BF16 || KIND == KIND_W4A16_BF16;
      constexpr int GROUPS = L::kARow / 32;
      constexpr int JN = kFp4 ? 4 : 2;
      constexpr int GK = 16 * JN;  // K per group
#pragma unroll
      for (int gi = 0; gi < GROUPS; ++gi) {
        uint32_t aw[MP][4];  // raw weight bytes: rows g / g+8, K halves 0 / 1 (as ldmatrix returns them)
#pragma unroll
        for (int i = 0; i < MP; ++i)
          ldsm_x4(tile_addr<L::kARow>(sa, wp0 + i * 16 + a_lrow, gi * 32 + a_lchunk * 16), aw[i][0], aw[i][1], aw[i][2],
                  aw[i][3]);
        // block scales (W4A16): 4 bytes per row per group = K blocks [4gi, 4gi+4), as bf16x2/f16x2 pairs
        uint32_t sc[MP][2][2];  // [i][row g / g+8][block pair: blocks {0,1} and {2,3} of the group] -> packed x2
        if constexpr (K::kWeightBlockScaled) {
#pragma unroll
          for (int i = 0; i < MP; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              const uint32_t s4 = lds32(st + L::kASFOff + (wp0 + i * 16 + g + 8 * h) * L::kASF + gi * 4);
              // this thread's K inside the group: half 0 -> 8t..8t+7 (block t/2), half 1 -> 32+8t.. (block 2+t/2)
              const uint32_t b0 = (s4 >> (8 * (t >> 1))) & 0xff, b1 = (s4 >> (8 * (2 + (t >> 1)))) & 0xff;
              const uint32_t s0 = kBf16 ? e4m3x2_to_bf16x2(b0 | (b0 << 8)) : e4m3x2_to_f16x2(b0 | (b0 << 8));
              const uint32_t s1 = kBf16 ? e4m3x2_to_bf16x2(b1 | (b1 << 8)) : e4m3x2_to_f16x2(b1 | (b1 << 8));
              sc[i][h][0] = s0;
              sc[i][h][1] = s1;
            }
        }
#pragma unroll
        for (int jm = 0; jm < JN; ++jm) {
          // ---- activation fragment for this MMA: 4 consecutive K of row (q = g) ----
          // logical K of this thread inside the group: fp8: 16*jm + 4t + {0..3}; fp4: half*32 + 8t + 4*(jm&1) + {0..3}
          const int kel = kFp4 ? ((jm >> 1) * 32 + 8 * t + 4 * (jm & 1)) : (16 * jm + 4 * t);
          uint32_t bf[NQ][2];
#pragma unroll
          for (int j = 0; j < NQ; ++j) {
            const uint2 v = lds64(tile_addr<L::kBRow>(sb, wq0 + j * 8 + g, (gi * GK + kel) * 2));
            bf[j][0] = v.x;
            bf[j][1] = v.y;
          }
          // ---- weight fragment: widen the 4 bytes / nibble pairs this MMA uses ----
#pragma unroll
          for (int i = 0; i < MP; ++i) {
            uint32_t af[4];
            if constexpr (!kFp4) {
              const uint32_t lo = aw[i][jm * 2], hi = aw[i][jm * 2 + 1];  // row g / row g+8, K 16*jm + 4t..+3
              af[0] = kBf16 ? e4m3x2_to_bf16x2(lo & 0xffff) : e4m3x2_to_f16x2(lo & 0xffff);  // row g, K 4t, 4t+1
              af[1] = kBf16 ? e4m3x2_to_bf16x2(hi & 0xffff) : e4m3x2_to_f16x2(hi & 0xffff);  // row g+8
              af[2] = kBf16 ? e4m3x2_to_bf16x2(lo >> 16) : e4m3x2_to_f16x2(lo >> 16);        // row g, K 4t+2, 4t+3
              af[3] = kBf16 ? e4m3x2_to_bf16x2(hi >> 16) : e4m3x2_to_f16x2(hi >> 16);
            } else {
              const int half = jm >> 1, sub = jm & 1;  // K half of the group, which 4 of the thread's 8 nibbles
              const uint32_t lo = aw[i][half * 2], hi = aw[i][half * 2 + 1];
              const int sh = 16 * sub;
              af[0] = kBf16 ? e2m1x2_to_bf16x2((lo >> sh) & 0xff) : e2m1x2_to_f16x2((lo >> sh) & 0xff);
              af[1] = kBf16 ? e2m1x2_to_bf16x2((hi >> sh) & 0xff) : e2m1x2_to_f16x2((hi >> sh) & 0xff);
              af[2] = kBf16 ? e2m1x2_to_bf16x2((lo >> (sh + 8)) & 0xff) : e2m1x2_to_f16x2((lo >> (sh + 8)) & 0xff);
              af[3] = kBf16 ? e2m1x2_to_bf16x2((hi >> (sh + 8)) & 0xff) : e2m1x2_to_f16x2((hi >> (sh + 8)) & 0xff);
              // exact: |e2m1 * e4m3| has <= 6 significant bits and lies in [2^-10, 2688]
              const uint32_t s_g = sc[i][0][half], s_g8 = sc[i][1][half];
              af[0] = kBf16 ? mul_bf16x2(af[0], s_g) : mul_f16x2(af[0], s_g);
              af[2] = kBf16 ? mul_bf16x2(af[2], s_g) : mul_f16x2(af[2], s_g);
              af[1] = kBf16 ? mul_bf16x2(af[1], s_g8) : mul_f16x2(af[1], s_g8);
              af[3] = kBf16 ? mul_bf16x2(af[3], s_g8) : mul_f16x2(af[3], s_g8);
            }
#pragma unroll
            for (int j = 0; j < NQ; ++j) {
              if constexpr (kBf16) mma_bf16(acc[i][j], af, bf[j][0], bf[j][1]);
              else mma_f16(acc[i][j], af, bf[j][0], bf[j][1]);
            }
          }
        }
      }
    }
  };

  // ---- multistage main loop ----
  const int ntiles = kt_end - kt_beg;
  for (int s = 0; s < stages - 1; ++s) {
    if (s < ntiles) load_stage(s, kt_beg + s);
    cp_async_commit();
  }
  for (int it = 0; it < ntiles; ++it) {
    // groups committed so far: stages-1+it; the one for `it` has landed when <= stages-2 remain pending
    cp_async_wait_dyn(stages - 2);
    __syncthreads();
    const int pf = it + stages - 1;
    if (pf < ntiles) load_stage(pf % stages, kt_beg + pf);
    cp_async_commit();
    compute_stage(it % stages);
  }
  cp_async_wait<0>();

  // ---- epilogue ----
  const float alpha = prm.alpha_ptr ? prm.alpha * *prm.alpha_ptr : prm.alpha;
  if (ksplit == 1) {
#pragma unroll
    for (int i = 0; i < MP; ++i)
#pragma unroll
      for (int j = 0; j < NQ; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int p = p0 + wp0 + i * 16 + g + (e >> 1) * 8;
          const int q = q0 + wq0 + j * 8 + 2 * t + (e & 1);
          if (p < prm.P && q < prm.Q) epilogue_store(prm, alpha, p, q, (float)acc[i][j][e]);
        }
    return;
  }

  // split-K: publish partials (fp32, or int32 bit patterns), the last split of this tile reduces in order
  Acc* const ws = static_cast<Acc*>(prm.ws);
#pragma unroll
  for (int i = 0; i < MP; ++i)
#pragma unroll
    for (int j = 0; j < NQ; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int p = p0 + wp0 + i * 16 + g + (e >> 1) * 8;
        const int q = q0 + wq0 + j * 8 + 2 * t + (e & 1);
        if (p < prm.P && q < prm.Q) ws[((int64_t)split * prm.P + p) * prm.Q + q] = acc[i][j][e];
      }
  __shared__ unsigned is_last;
  __threadfence();
  __syncthreads();
  const unsigned tile = blockIdx.x + gridDim.x * blockIdx.y;
  if (tid == 0) is_last = atomicAdd(&prm.counters[tile], 1u) == (unsigned)(ksplit - 1);
  __syncthreads();
  if (!is_last) return;
  __threadfence();
  const int np = min(BP, prm.P - p0), nq = min(BQ, prm.Q - q0);
  for (int idx = tid; idx < np * nq; idx += C::kThreads) {
    const int p = p0 + idx / nq, q = q0 + idx % nq;
    Acc s = __ldcg(&ws[(int64_t)p * prm.Q + q]);
    for (int k = 1; k < ksplit; ++k) s += __ldcg(&ws[((int64_t)k * prm.P + p) * prm.Q + q]);
    epilogue_store(prm, alpha, p, q, (float)s);
  }
  if (tid == 0) prm.counters[tile] = 0u;
}

// ---------------------------------------------------------------------------------------------------------------
// Host-side launch for one (kind, config)

template <int KIND, int CFG>
bool tc_launch_one(const TcParams& prm, cudaStream_t stream) {
  using L = Layout<KIND, CFG>;
  using C = Cfg<CFG>;
  const int max_stages = kMaxSmem / L::kStageBytes;
  if (max_stages < 2) return false;
  TcParams p = prm;
  const int cap = max_stages < kMaxStages ? max_stages : kMaxStages;
  p.stages = prm.stages < 2 ? 2 : (prm.stages > cap ? cap : prm.stages);
  const int smem = p.stages * L::kStageBytes;
  static unsigned attr_set = 0;  // per instantiation, one bit per device
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess || dev >= 32) return false;
  if (!(attr_set & (1u << dev))) {
    if (cudaFuncSetAttribute(tc_gemm_kernel<KIND, CFG>, cudaFuncAttributeMaxDynamicSharedMemorySize, kMaxSmem) !=
        cudaSuccess)
      return false;
    attr_set |= 1u << dev;
  }
  const dim3 grid((p.P + C::BP - 1) / C::BP, (p.Q + C::BQ - 1) / C::BQ, p.ksplit);
  tc_gemm_kernel<KIND, CFG><<<grid, C::kThreads, smem, stream>>>(p);
  return true;
}

template <int KIND>
bool tc_launch_kind_impl(const TcParams& prm, int cfg, cudaStream_t stream) {
  switch (cfg) {
    case 0: return tc_launch_one<KIND, 0>(prm, stream);
    case 1: return tc_launch_one<KIND, 1>(prm, stream);
    case 2: return tc_launch_one<KIND, 2>(prm, stream);
    case 3: return tc_launch_one<KIND, 3>(prm, stream);
    case 4: return tc_launch_one<KIND, 4>(prm, stream);
    case 5: return tc_launch_one<KIND, 5>(prm, stream);
    case 6: return tc_launch_one<KIND, 6>(prm, stream);
    case 7: return tc_launch_one<KIND, 7>(prm, stream);
    default: return false;
  }
}

}  // namespace llg
