// SPDX-License-Identifier: Apache-2.0
#include "launch.cuh"

namespace llg {

template <>
bool launch_lpr<8>(const Params& p, const LaunchConfig& c, cudaStream_t stream) {
  return launch_lpr_impl<8>(p, c, stream);
}

}  // namespace llg
