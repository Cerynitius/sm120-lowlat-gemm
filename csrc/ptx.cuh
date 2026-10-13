// SPDX-License-Identifier: Apache-2.0
// Thin PTX wrappers for SM120 (compile for sm_120a: the FP8 kind::f8f6f4 MMA, the NVFP4 block-scaled MMA and the
// e2m1/e4m3 conversions are arch-specific features).
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace llg {

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// 16-byte global -> shared copy; src_bytes = 0 zero-fills the destination.
// ---- thread block clusters (distributed shared memory) ----
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}

__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

// Address of the same shared-memory location in block `rank` of this cluster.
__device__ __forceinline__ uint32_t cluster_map(uint32_t addr, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank));
  return r;
}

__device__ __forceinline__ uint32_t ld_cluster_u32(uint32_t addr) {
  uint32_t v;
  asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(v) : "r"(addr) : "memory");
  return v;
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, uint32_t src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(src_bytes));
}

// 4-byte variant (L1-allocating .ca is the only form for sizes < 16).
__device__ __forceinline__ void cp_async4(uint32_t dst, const void* src, uint32_t src_bytes) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(dst), "l"(src), "r"(src_bytes));
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

// wait_group needs an immediate; stages is a runtime value (2..8)
__device__ __forceinline__ void cp_async_wait_dyn(int pending) {
  switch (pending) {
    case 0: cp_async_wait<0>(); break;
    case 1: cp_async_wait<1>(); break;
    case 2: cp_async_wait<2>(); break;
    case 3: cp_async_wait<3>(); break;
    case 4: cp_async_wait<4>(); break;
    case 5: cp_async_wait<5>(); break;
    case 6: cp_async_wait<6>(); break;
    case 7: cp_async_wait<7>(); break;
    case 8: cp_async_wait<8>(); break;
    case 9: cp_async_wait<9>(); break;
    case 10: cp_async_wait<10>(); break;
    case 11: cp_async_wait<11>(); break;
    case 12: cp_async_wait<12>(); break;
    case 13: cp_async_wait<13>(); break;
    default: cp_async_wait<14>(); break;
  }
}

__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}

__device__ __forceinline__ void ldsm_x2(uint32_t addr, uint32_t& r0, uint32_t& r1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(addr));
}

__device__ __forceinline__ uint32_t lds32(uint32_t addr) {
  uint32_t v;
  asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(v) : "r"(addr));
  return v;
}

__device__ __forceinline__ uint2 lds64(uint32_t addr) {
  uint2 v;
  asm volatile("ld.shared.v2.b32 {%0, %1}, [%2];\n" : "=r"(v.x), "=r"(v.y) : "r"(addr));
  return v;
}

__device__ __forceinline__ uint4 lds128(uint32_t addr) {
  uint4 v;
  asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr));
  return v;
}

// ---- MMA: D = A (16xK, row) * B (Kx8, col) + D -------------------------------------------------------------

__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_f16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_e4m3(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
      "{%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_s8(int32_t (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// NVFP4: e2m1 x e2m1, one ue4m3 scale per 16 K. sfa: the 4 scales (K blocks 0..3 of this k64 step, byte j = block
// j) of A row (groupID + 8 * (lane & 1)); sfb: the 4 scales of B column groupID. Selectors 0/0 as in CUTLASS.
__device__ __forceinline__ void mma_nvfp4(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1,
                                          uint32_t sfa, uint32_t sfb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "h"((uint16_t)0),
        "h"((uint16_t)0), "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));
}

// ---- exact widening conversions (low half of the result <- low bits of the input) -----------------------------

__device__ __forceinline__ uint32_t e4m3x2_to_bf16x2(uint32_t two) {
  uint32_t r;
  asm("cvt.rn.bf16x2.e4m3x2 %0, %1;\n" : "=r"(r) : "h"((uint16_t)two));
  return r;
}

__device__ __forceinline__ uint32_t e4m3x2_to_f16x2(uint32_t two) {
  uint32_t r;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(r) : "h"((uint16_t)two));
  return r;
}

__device__ __forceinline__ uint32_t e2m1x2_to_bf16x2(uint32_t byte) {
  uint32_t r;
  asm("{\n .reg .b8 t;\n .reg .b16 h;\n cvt.u16.u32 h, %1;\n mov.b16 {t, _}, h;\n"
      " cvt.rn.bf16x2.e2m1x2 %0, t;\n}\n"
      : "=r"(r)
      : "r"(byte));
  return r;
}

__device__ __forceinline__ uint32_t e2m1x2_to_f16x2(uint32_t byte) {
  uint32_t r;
  asm("{\n .reg .b8 t;\n .reg .b16 h;\n cvt.u16.u32 h, %1;\n mov.b16 {t, _}, h;\n"
      " cvt.rn.f16x2.e2m1x2 %0, t;\n}\n"
      : "=r"(r)
      : "r"(byte));
  return r;
}

__device__ __forceinline__ uint32_t mul_bf16x2(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("mul.rn.bf16x2 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

__device__ __forceinline__ uint32_t mul_f16x2(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

// ---- narrowing conversions for quantization ---------------------------------------------------------------------

// Two floats -> e4m3x2 (lo -> low byte), saturating.
__device__ __forceinline__ uint16_t f32x2_to_e4m3x2(float lo, float hi) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}

// Two floats -> one byte of e2m1x2 (lo -> low nibble), saturating.
__device__ __forceinline__ uint32_t f32x2_to_e2m1x2(float lo, float hi) {
  uint32_t r;
  asm("{\n .reg .b8 t;\n .reg .b16 h;\n cvt.rn.satfinite.e2m1x2.f32 t, %1, %2;\n mov.b16 h, {t, t};\n"
      " cvt.u32.u16 %0, h;\n}\n"
      : "=r"(r)
      : "f"(hi), "f"(lo));
  return r & 0xffu;
}

}  // namespace llg
