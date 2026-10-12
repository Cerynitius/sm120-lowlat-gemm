// SPDX-License-Identifier: Apache-2.0
// Activation quantization for the W8A8 / W4A4 GEMMs. One block per row.
//   fp8 static:     q = sat_e4m3(x / scale)                                   (scale: 1 float, device)
//   fp8 per token:  s[m] = amax(x[m]) / 448;  q = sat_e4m3(x / s[m])
//   int8 per token: s[m] = amax(x[m]) / 127;  q = rn_sat_int8(x / s[m])
//   nvfp4:          per 16 K: sf = e4m3(amax / 6 * g);  q = e2m1(x * (g / sf))  (g: global scale, 1 float, device)
//                   dequant: x ~= q * sf / g
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include "ptx.cuh"
#include "quant.h"

namespace llg {

namespace {

constexpr int kThreads = 256;

template <typename T>
__device__ __forceinline__ float to_f(T v);
template <>
__device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <>
__device__ __forceinline__ float to_f<__half>(__half v) { return __half2float(v); }
template <>
__device__ __forceinline__ float to_f<float>(float v) { return v; }

__device__ __forceinline__ float block_max(float v) {
  __shared__ float red[kThreads / 32];
#pragma unroll
  for (int o = 16; o >= 1; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  float r = 0.f;
#pragma unroll
  for (int i = 0; i < kThreads / 32; ++i) r = fmaxf(r, red[i]);
  __syncthreads();
  return r;
}

template <typename T, bool PER_TOKEN>
__global__ void __launch_bounds__(kThreads) quant_fp8_kernel(const QuantParams p) {
  const int m = blockIdx.x;
  const T* x = static_cast<const T*>(p.x) + (int64_t)m * p.ldx;
  uint8_t* q = static_cast<uint8_t*>(p.q) + (int64_t)m * p.ldq;
  float scale;
  if constexpr (PER_TOKEN) {
    float a = 0.f;
    for (int k = threadIdx.x; k < p.K; k += kThreads) a = fmaxf(a, fabsf(to_f(x[k])));
    a = block_max(a);
    scale = fmaxf(a / 448.f, p.min_scale);
    if (threadIdx.x == 0) p.scale_out[m] = scale;
  } else {
    scale = *p.scale_in;
  }
  const float inv = 1.f / scale;
  for (int k = 2 * threadIdx.x; k < p.K; k += 2 * kThreads) {
    const float lo = to_f(x[k]) * inv, hi = k + 1 < p.K ? to_f(x[k + 1]) * inv : 0.f;
    const uint16_t two = f32x2_to_e4m3x2(lo, hi);
    q[k] = two & 0xff;
    if (k + 1 < p.K) q[k + 1] = two >> 8;
  }
}

template <typename T>
__global__ void __launch_bounds__(kThreads) quant_int8_kernel(const QuantParams p) {
  const int m = blockIdx.x;
  const T* x = static_cast<const T*>(p.x) + (int64_t)m * p.ldx;
  int8_t* q = static_cast<int8_t*>(p.q) + (int64_t)m * p.ldq;
  float a = 0.f;
  for (int k = threadIdx.x; k < p.K; k += kThreads) a = fmaxf(a, fabsf(to_f(x[k])));
  a = block_max(a);
  const float scale = fmaxf(a / 127.f, p.min_scale);
  if (threadIdx.x == 0) p.scale_out[m] = scale;
  const float inv = 1.f / scale;
  for (int k = threadIdx.x; k < p.K; k += kThreads) {
    const float v = fminf(fmaxf(rintf(to_f(x[k]) * inv), -127.f), 127.f);
    q[k] = static_cast<int8_t>(v);
  }
}

// One thread per 16-element block.
template <typename T>
__global__ void __launch_bounds__(kThreads) quant_nvfp4_kernel(const QuantParams p) {
  const int m = blockIdx.x;
  const T* x = static_cast<const T*>(p.x) + (int64_t)m * p.ldx;
  uint8_t* q = static_cast<uint8_t*>(p.q) + (int64_t)m * p.ldq;
  uint8_t* sf = static_cast<uint8_t*>(p.sf_out) + (int64_t)m * p.ldsf;
  const float g = *p.scale_in;
  const int nblk = p.K / 16;
  for (int b = threadIdx.x; b < nblk; b += kThreads) {
    float v[16];
    float a = 0.f;
#pragma unroll
    for (int e = 0; e < 16; ++e) {
      v[e] = to_f(x[b * 16 + e]);
      a = fmaxf(a, fabsf(v[e]));
    }
    const uint16_t s2 = f32x2_to_e4m3x2(a / 6.f * g, 0.f);
    const uint32_t s8 = s2 & 0xff;
    const float sfv = __half2float(__ushort_as_half((uint16_t)(e4m3x2_to_f16x2(s8) & 0xffff)));
    const float inv = sfv > 0.f ? g / sfv : 0.f;
    uint32_t w[2] = {0, 0};
#pragma unroll
    for (int e = 0; e < 16; e += 2) {
      const uint32_t byte = f32x2_to_e2m1x2(v[e] * inv, v[e + 1] * inv);
      w[e / 8] |= byte << (8 * ((e / 2) % 4));
    }
    *reinterpret_cast<uint2*>(q + b * 8) = make_uint2(w[0], w[1]);
    sf[b] = static_cast<uint8_t>(s8);
  }
}

template <typename T>
bool launch_typed(const QuantParams& p, cudaStream_t s) {
  const dim3 grid(p.M), block(kThreads);
  switch (p.mode) {
    case QUANT_FP8_STATIC: quant_fp8_kernel<T, false><<<grid, block, 0, s>>>(p); return true;
    case QUANT_FP8_PER_TOKEN: quant_fp8_kernel<T, true><<<grid, block, 0, s>>>(p); return true;
    case QUANT_INT8_PER_TOKEN: quant_int8_kernel<T><<<grid, block, 0, s>>>(p); return true;
    case QUANT_NVFP4: quant_nvfp4_kernel<T><<<grid, block, 0, s>>>(p); return true;
    default: return false;
  }
}

}  // namespace

bool quant_launch(const QuantParams& p, cudaStream_t stream) {
  switch (p.in_dtype) {
    case 1: return launch_typed<__nv_bfloat16>(p, stream);
    case 2: return launch_typed<__half>(p, stream);
    case 3: return launch_typed<float>(p, stream);
    default: return false;
  }
}

}  // namespace llg
