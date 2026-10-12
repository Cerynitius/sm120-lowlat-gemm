// SPDX-License-Identifier: Apache-2.0
#include "tc_gemm.cuh"

namespace llg {

template <>
bool tc_launch_kind<KIND_FP8>(const TcParams& p, int cfg, cudaStream_t stream) {
  return tc_launch_kind_impl<KIND_FP8>(p, cfg, stream);
}

}  // namespace llg
