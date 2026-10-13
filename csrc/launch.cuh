// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "gemv.cuh"

namespace llg {

// Token capacities compiled per LPR; M is rounded up to the next one.
inline int token_bucket(int m) {
  if (m <= 4) return m;
  if (m <= 6) return 6;
  if (m <= 8) return 8;
  return 16;
}

template <int MT, int R, int U, int LPR>
void launch_one(const Params& p, const LaunchConfig& c, cudaStream_t stream) {
  const int rows_per_block = c.warps * (32 / LPR) * R;
  const dim3 grid((p.N + rows_per_block - 1) / rows_per_block, c.ksplit);
  Params q = p;
  q.pdl = pdl_enabled();
  q.pdl_early = (int64_t)grid.x * grid.y <= sm_count_current();
  q.cluster = p.cluster && c.ksplit > 1;
  const size_t smem = q.cluster ? (size_t)rows_per_block * p.M * sizeof(float) : 0;
  const dim3 cluster(1, q.cluster ? c.ksplit : 1, 1);
  if (p.f16) {
    launch_kernel(gemv_kernel<MT, R, U, LPR, true>, grid, dim3(c.warps * 32), smem, stream, q.pdl, cluster, q);
  } else {
    launch_kernel(gemv_kernel<MT, R, U, LPR, false>, grid, dim3(c.warps * 32), smem, stream, q.pdl, cluster, q);
  }
}

// (R, U) pairs with R * U <= 8 keep the in-flight weight vectors at <= 32 registers per lane.
template <int MT, int LPR>
bool launch_ru(const Params& p, const LaunchConfig& c, cudaStream_t stream) {
  switch (c.rows * 16 + c.unroll) {
    case 1 * 16 + 2: launch_one<MT, 1, 2, LPR>(p, c, stream); return true;
    case 1 * 16 + 4: launch_one<MT, 1, 4, LPR>(p, c, stream); return true;
    case 1 * 16 + 8: launch_one<MT, 1, 8, LPR>(p, c, stream); return true;
    case 2 * 16 + 2: launch_one<MT, 2, 2, LPR>(p, c, stream); return true;
    case 2 * 16 + 4: launch_one<MT, 2, 4, LPR>(p, c, stream); return true;
    case 4 * 16 + 2: launch_one<MT, 4, 2, LPR>(p, c, stream); return true;
    default: return false;
  }
}

template <int LPR>
bool launch_lpr_impl(const Params& p, const LaunchConfig& c, cudaStream_t stream) {
  switch (token_bucket(p.M)) {
    case 1: return launch_ru<1, LPR>(p, c, stream);
    case 2: return launch_ru<2, LPR>(p, c, stream);
    case 3: return launch_ru<3, LPR>(p, c, stream);
    case 4: return launch_ru<4, LPR>(p, c, stream);
    case 6: return launch_ru<6, LPR>(p, c, stream);
    case 8: return launch_ru<8, LPR>(p, c, stream);
    case 16: return launch_ru<16, LPR>(p, c, stream);
    default: return false;
  }
}

}  // namespace llg
