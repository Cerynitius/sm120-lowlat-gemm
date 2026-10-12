// SPDX-License-Identifier: Apache-2.0
// Host-side view of the kernel parameters (no CUDA language extensions, safe for the C++ compiler).
#pragma once

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace llg {

struct Params {
  const void* x;       // bf16 or f16 [M, K], row stride ldx (elements), unit column stride
  const void* w;       // same type [N, K], row stride ldw
  void* y;             // [M, N], row stride ldy; input type, or fp32 when out_fp32
  int64_t ldx, ldw, ldy;
  int M, N, K;
  int ksplit;
  float* ws;           // [ksplit, M, N] partial sums (ksplit > 1)
  unsigned* counters;  // one arrival counter per block column, zero between launches
  bool out_fp32;
  bool f16;            // x, w (and a 16-bit y) are f16 instead of bf16
};

struct LaunchConfig {
  int rows;           // rows per row group (R)
  int unroll;         // 16-byte vectors in flight per row and lane (U)
  int lanes_per_row;  // lanes per row group (LPR)
  int warps;          // warps per block
  int ksplit;         // K partitions across blocks
};

// Defined per LPR in gemv_lpr{8,16,32}.cu. Returns false for an unsupported (M bucket, rows, unroll) combination.
template <int LPR>
bool launch_lpr(const Params& p, const LaunchConfig& c, cudaStream_t stream);

}  // namespace llg
