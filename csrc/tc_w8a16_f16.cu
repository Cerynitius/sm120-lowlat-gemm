// SPDX-License-Identifier: Apache-2.0
#include "tc_gemm.cuh"

namespace llg {

template <>
bool tc_launch_kind<KIND_W8A16_F16>(const TcParams& p, int cfg, cudaStream_t stream) {
  return tc_launch_kind_impl<KIND_W8A16_F16>(p, cfg, stream);
}

}  // namespace llg
