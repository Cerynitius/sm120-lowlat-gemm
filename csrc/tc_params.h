// SPDX-License-Identifier: Apache-2.0
// Host-side view of the tensor-core GEMM parameters (no CUDA language extensions).
#pragma once

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace llg {

enum TcKind : int {
  KIND_BF16 = 0,         // bf16 x bf16
  KIND_F16 = 1,          // f16 x f16
  KIND_FP8 = 2,          // e4m3 x e4m3
  KIND_INT8 = 3,         // s8 x s8 (s32 accumulate)
  KIND_NVFP4 = 4,        // e2m1 x e2m1 with ue4m3 scales per 16 K
  KIND_W8A16_BF16 = 5,   // e4m3 weight (A) x bf16 activation (B)
  KIND_W8A16_F16 = 6,    // e4m3 weight (A) x f16 activation (B)
  KIND_W4A16_BF16 = 7,   // e2m1 weight + ue4m3 scales per 16 K (A) x bf16 activation (B)
  KIND_W4A16_F16 = 8,    // same with f16 activation
  KIND_COUNT = 9,
};

enum DType : int { DT_NONE = 0, DT_BF16 = 1, DT_F16 = 2, DT_F32 = 3 };

constexpr int kNumCfgs = 10;
constexpr int kMaxStages = 16;

struct TcParams {
  // C[p, q] = sum_k A[p, k] B[q, k]; strides in bytes
  const void* a;
  const void* b;
  int64_t lda, ldb;
  const void* a_sf;  // block scales of A ([P, K/16] e4m3 bytes), NVFP4 and W4A16
  const void* b_sf;  // block scales of B, NVFP4
  int64_t lda_sf, ldb_sf;
  // epilogue: y = acc * alpha * (*alpha_ptr) * row_scale[p] * col_scale[q] + bias
  const float* row_scale;
  int row_scale_len;  // 0: none, 1: scalar, otherwise per p
  const float* col_scale;
  int col_scale_len;  // 0: none, 1: scalar, otherwise per q
  const void* bias;
  int bias_dtype;     // DType
  bool bias_on_p;     // bias indexed by p (else q)
  const float* alpha_ptr;
  float alpha;
  void* y;            // y[p, q] (swap = false) or y[q, p] (swap = true), row stride ldy elements
  int64_t ldy;
  int out_dtype;      // DType
  bool swap;
  int P, Q, K;
  int ksplit;
  int stages;
  void* ws;           // split-K partials [ksplit, P, Q] (fp32 or int32)
  unsigned* counters; // per output tile, zero between launches
  bool cluster;       // split-K partials reduced over distributed shared memory (ksplit blocks per cluster)
  const void* pf;     // optional: L2 prefetch of [pf, pf + pf_bytes) (the next GEMM's weight)
  int64_t pf_bytes;
  bool pdl;           // launch with programmatic dependent launch (set by the launcher)
  bool pdl_early;     // the grid fits in one wave: let the next kernel launch right away
};

// Defined per kind in tc_<kind>.cu; false if the configuration cannot run (shared memory).
template <int KIND>
bool tc_launch_kind(const TcParams& p, int cfg, cudaStream_t stream);

// Bytes of shared memory one pipeline stage needs (0 for an invalid kind/config).
int tc_stage_bytes(int kind, int cfg);
// Programmatic dependent launch for every later GEMM launch (process-wide; see pdl.cuh).
void set_pdl_flag(bool on);
bool pdl_enabled();
// Tile shape of a config.
void tc_cfg_tile(int cfg, int* bp, int* bq, int* threads);

}  // namespace llg
