// SPDX-License-Identifier: Apache-2.0
#include <atomic>

#include "tc_gemm.cuh"

namespace llg {

template <int KIND>
static int stage_bytes_kind(int cfg) {
  switch (cfg) {
    case 0: return Layout<KIND, 0>::kStageBytes;
    case 1: return Layout<KIND, 1>::kStageBytes;
    case 2: return Layout<KIND, 2>::kStageBytes;
    case 3: return Layout<KIND, 3>::kStageBytes;
    case 4: return Layout<KIND, 4>::kStageBytes;
    case 5: return Layout<KIND, 5>::kStageBytes;
    case 6: return Layout<KIND, 6>::kStageBytes;
    case 7: return Layout<KIND, 7>::kStageBytes;
    case 8: return Layout<KIND, 8>::kStageBytes;
    case 9: return Layout<KIND, 9>::kStageBytes;
    default: return 0;
  }
}

int tc_stage_bytes(int kind, int cfg) {
  switch (kind) {
    case KIND_BF16: return stage_bytes_kind<KIND_BF16>(cfg);
    case KIND_F16: return stage_bytes_kind<KIND_F16>(cfg);
    case KIND_FP8: return stage_bytes_kind<KIND_FP8>(cfg);
    case KIND_INT8: return stage_bytes_kind<KIND_INT8>(cfg);
    case KIND_NVFP4: return stage_bytes_kind<KIND_NVFP4>(cfg);
    case KIND_W8A16_BF16: return stage_bytes_kind<KIND_W8A16_BF16>(cfg);
    case KIND_W8A16_F16: return stage_bytes_kind<KIND_W8A16_F16>(cfg);
    case KIND_W4A16_BF16: return stage_bytes_kind<KIND_W4A16_BF16>(cfg);
    case KIND_W4A16_F16: return stage_bytes_kind<KIND_W4A16_F16>(cfg);
    default: return 0;
  }
}

template <int CFG>
static void tile_of(int* bp, int* bq, int* threads) {
  *bp = Cfg<CFG>::BP;
  *bq = Cfg<CFG>::BQ;
  *threads = Cfg<CFG>::kThreads;
}

void tc_cfg_tile(int cfg, int* bp, int* bq, int* threads) {
  switch (cfg) {
    case 0: tile_of<0>(bp, bq, threads); break;
    case 1: tile_of<1>(bp, bq, threads); break;
    case 2: tile_of<2>(bp, bq, threads); break;
    case 3: tile_of<3>(bp, bq, threads); break;
    case 4: tile_of<4>(bp, bq, threads); break;
    case 5: tile_of<5>(bp, bq, threads); break;
    case 6: tile_of<6>(bp, bq, threads); break;
    case 7: tile_of<7>(bp, bq, threads); break;
    case 8: tile_of<8>(bp, bq, threads); break;
    case 9: tile_of<9>(bp, bq, threads); break;
    default: *bp = *bq = *threads = 0;
  }
}

}  // namespace llg

namespace llg {

static std::atomic<bool> g_pdl{false};

bool pdl_enabled() { return g_pdl.load(std::memory_order_relaxed); }
void set_pdl_flag(bool on) { g_pdl.store(on, std::memory_order_relaxed); }

int sm_count_current() {
  static int counts[64] = {};
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return 0;
  if (counts[dev] == 0) cudaDeviceGetAttribute(&counts[dev], cudaDevAttrMultiProcessorCount, dev);
  return counts[dev];
}

}  // namespace llg
