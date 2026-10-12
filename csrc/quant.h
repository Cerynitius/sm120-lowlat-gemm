// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace llg {

enum QuantMode : int {
  QUANT_FP8_STATIC = 0,
  QUANT_FP8_PER_TOKEN = 1,
  QUANT_INT8_PER_TOKEN = 2,
  QUANT_NVFP4 = 3,
};

struct QuantParams {
  int mode;
  int in_dtype;            // 1 bf16, 2 f16, 3 f32
  const void* x;           // [M, K]
  int64_t ldx;             // elements
  void* q;                 // [M, K] bytes (fp8 / int8) or [M, K/2] (packed fp4)
  int64_t ldq;             // bytes
  void* sf_out;            // nvfp4: [M, K/16] e4m3 bytes
  int64_t ldsf;            // bytes
  const float* scale_in;   // fp8 static: the scale; nvfp4: the global scale g
  float* scale_out;        // per-token scales [M]
  float min_scale;         // lower bound for per-token scales (all-zero rows)
  int M, K;
};

bool quant_launch(const QuantParams& p, cudaStream_t stream);

}  // namespace llg
