// SPDX-License-Identifier: Apache-2.0
#include "launch.cuh"

namespace llg {

template <>
bool launch_lpr<32>(const Params& p, const LaunchConfig& c, cudaStream_t stream) {
  return launch_lpr_impl<32>(p, c, stream);
}

}  // namespace llg
