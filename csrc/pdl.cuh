// SPDX-License-Identifier: Apache-2.0
//
// Programmatic dependent launch (PDL). With PDL a GEMM may start while the previous kernel in the stream is still
// running: it streams its (constant) weights first, then waits for the previous grid before it touches activations,
// outputs or the split-K workspace. It is enabled only through set_pdl(), because it assumes that no kernel still in
// flight writes the weights.
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include <utility>

namespace llg {

// Waits until the prerequisite grid has completed and its memory is visible (no-op without PDL).
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }

// Lets the next kernel in the stream launch (it still waits for this grid before reading our results).
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

// Pulls slice `part` of `parts` of [ptr, ptr + bytes) into L2 (fire and forget; ptr 16-byte aligned). A GEMM issues
// this for the weight of the GEMM that runs next, once its own weight is in flight, so that weight streams from DRAM
// while the kernels in between (norms, activations) run.
__device__ __forceinline__ void l2_prefetch_slice(const void* ptr, int64_t bytes, int64_t part, int64_t parts) {
  if (ptr == nullptr || bytes < 16) return;
  constexpr int64_t kChunk = 16384;
  const int64_t gran = bytes >> 4;
  const int64_t b0 = gran * part / parts * 16, b1 = gran * (part + 1) / parts * 16;
  for (int64_t off = b0 + (int64_t)threadIdx.x * kChunk; off < b1; off += (int64_t)blockDim.x * kChunk) {
    const uint32_t n = (uint32_t)(b1 - off < kChunk ? b1 - off : kChunk);
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(static_cast<const char*>(ptr) + off), "r"(n)
                 : "memory");
  }
}

bool pdl_enabled();
int sm_count_current();

// Launch with the programmatic stream serialization attribute when PDL is enabled, and in thread block clusters of
// `cluster` blocks unless that is (1, 1, 1).
template <typename... KArgs, typename... Args>
cudaError_t launch_kernel(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t stream, bool pdl,
                          dim3 cluster, Args&&... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attr[2];
  int n = 0;
  if (pdl) {
    attr[n].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[n].val.programmaticStreamSerializationAllowed = 1;
    ++n;
  }
  if (cluster.x * cluster.y * cluster.z > 1) {
    attr[n].id = cudaLaunchAttributeClusterDimension;
    attr[n].val.clusterDim.x = cluster.x;
    attr[n].val.clusterDim.y = cluster.y;
    attr[n].val.clusterDim.z = cluster.z;
    ++n;
  }
  cfg.attrs = attr;
  cfg.numAttrs = n;
  return cudaLaunchKernelEx(&cfg, kernel, std::forward<Args>(args)...);
}

}  // namespace llg
