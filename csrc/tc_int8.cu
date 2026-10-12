// SPDX-License-Identifier: Apache-2.0
#include "tc_gemm.cuh"

namespace llg {

template <>
bool tc_launch_kind<KIND_INT8>(const TcParams& p, int cfg, cudaStream_t stream) {
  return tc_launch_kind_impl<KIND_INT8>(p, cfg, stream);
}

}  // namespace llg
