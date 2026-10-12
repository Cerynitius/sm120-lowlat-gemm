// SPDX-License-Identifier: Apache-2.0
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "params.h"

namespace llg {

namespace {

constexpr int64_t kMaxTokens = 16;

bool aligned16(const void* ptr) { return (reinterpret_cast<uintptr_t>(ptr) & 15) == 0; }

void check_config(int64_t rows, int64_t unroll, int64_t lanes_per_row, int64_t warps, int64_t ksplit) {
  const int64_t ru = rows * 16 + unroll;
  TORCH_CHECK(ru == 18 || ru == 20 || ru == 24 || ru == 34 || ru == 36 || ru == 66,
              "lowlat_gemm: unsupported (rows, unroll) = (", rows, ", ", unroll,
              "); supported: (1,2) (1,4) (1,8) (2,2) (2,4) (4,2)");
  TORCH_CHECK(lanes_per_row == 8 || lanes_per_row == 16 || lanes_per_row == 32,
              "lowlat_gemm: lanes_per_row must be 8, 16 or 32, got ", lanes_per_row);
  TORCH_CHECK(warps >= 1 && warps <= 8, "lowlat_gemm: warps must be in [1, 8], got ", warps);
  TORCH_CHECK(ksplit >= 1 && ksplit <= 64, "lowlat_gemm: ksplit must be in [1, 64], got ", ksplit);
}

int64_t grid_columns(int64_t n, int64_t rows, int64_t lanes_per_row, int64_t warps) {
  const int64_t rows_per_block = warps * (32 / lanes_per_row) * rows;
  return (n + rows_per_block - 1) / rows_per_block;
}

void run(const at::Tensor& y, const at::Tensor& x, const at::Tensor& w, const at::Tensor& ws,
         const at::Tensor& counters, int64_t rows, int64_t unroll, int64_t lanes_per_row, int64_t warps,
         int64_t ksplit) {
  check_config(rows, unroll, lanes_per_row, warps, ksplit);
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && y.is_cuda(), "lowlat_gemm: tensors must be on CUDA");
  TORCH_CHECK(x.device() == w.device() && y.device() == x.device(), "lowlat_gemm: tensors on different devices");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16,
              "lowlat_gemm: x and w must be bfloat16, got ", x.scalar_type(), " and ", w.scalar_type());
  TORCH_CHECK(y.scalar_type() == at::kBFloat16 || y.scalar_type() == at::kFloat,
              "lowlat_gemm: output must be bfloat16 or float32, got ", y.scalar_type());
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && y.dim() == 2, "lowlat_gemm: x, w, y must be 2-D");
  const int64_t m = x.size(0), k = x.size(1), n = w.size(0);
  TORCH_CHECK(w.size(1) == k, "lowlat_gemm: x is [", m, ", ", k, "] but w is [", n, ", ", w.size(1), "]");
  TORCH_CHECK(y.size(0) == m && y.size(1) == n, "lowlat_gemm: output must be [", m, ", ", n, "]");
  TORCH_CHECK(m >= 1 && m <= kMaxTokens, "lowlat_gemm: M must be in [1, ", kMaxTokens, "], got ", m);
  TORCH_CHECK(k % 8 == 0 && k > 0, "lowlat_gemm: K must be a positive multiple of 8, got ", k);
  TORCH_CHECK(n >= 1 && n <= INT32_MAX && k <= INT32_MAX, "lowlat_gemm: N or K too large");
  TORCH_CHECK(x.stride(1) == 1 && w.stride(1) == 1 && y.stride(1) == 1,
              "lowlat_gemm: x, w, y need unit stride in the last dimension");
  TORCH_CHECK(x.stride(0) % 8 == 0 && w.stride(0) % 8 == 0,
              "lowlat_gemm: x and w row strides must be multiples of 8 elements");
  TORCH_CHECK(aligned16(x.data_ptr()) && aligned16(w.data_ptr()),
              "lowlat_gemm: x and w must be 16-byte aligned");
  const int64_t columns = grid_columns(n, rows, lanes_per_row, warps);
  TORCH_CHECK(columns <= INT32_MAX, "lowlat_gemm: grid too large");
  if (ksplit > 1) {
    TORCH_CHECK(ws.is_cuda() && ws.device() == x.device() && ws.scalar_type() == at::kFloat && ws.is_contiguous(),
                "lowlat_gemm: workspace must be a contiguous float32 tensor on the same device");
    TORCH_CHECK(counters.is_cuda() && counters.device() == x.device() && counters.scalar_type() == at::kInt &&
                    counters.is_contiguous(),
                "lowlat_gemm: counters must be a contiguous int32 tensor on the same device");
    TORCH_CHECK(ws.numel() >= ksplit * m * n, "lowlat_gemm: workspace has ", ws.numel(), " floats, needs ",
                ksplit * m * n);
    TORCH_CHECK(counters.numel() >= columns, "lowlat_gemm: counters has ", counters.numel(), " entries, needs ",
                columns);
  }

  Params p;
  p.x = x.data_ptr();
  p.w = w.data_ptr();
  p.y = y.data_ptr();
  p.ldx = x.stride(0);
  p.ldw = w.stride(0);
  p.ldy = y.stride(0);
  p.M = static_cast<int>(m);
  p.N = static_cast<int>(n);
  p.K = static_cast<int>(k);
  p.ksplit = static_cast<int>(ksplit);
  p.ws = ksplit > 1 ? ws.data_ptr<float>() : nullptr;
  p.counters = ksplit > 1 ? reinterpret_cast<unsigned*>(counters.data_ptr<int>()) : nullptr;
  p.out_fp32 = y.scalar_type() == at::kFloat;

  LaunchConfig c{static_cast<int>(rows), static_cast<int>(unroll), static_cast<int>(lanes_per_row),
                 static_cast<int>(warps), static_cast<int>(ksplit)};
  const c10::cuda::CUDAGuard guard(x.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  bool ok = false;
  switch (lanes_per_row) {
    case 8: ok = launch_lpr<8>(p, c, stream); break;
    case 16: ok = launch_lpr<16>(p, c, stream); break;
    case 32: ok = launch_lpr<32>(p, c, stream); break;
  }
  TORCH_CHECK(ok, "lowlat_gemm: no kernel for M=", m, " rows=", rows, " unroll=", unroll);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

at::Tensor gemm(const at::Tensor& x, const at::Tensor& w, const at::Tensor& ws, const at::Tensor& counters,
                int64_t rows, int64_t unroll, int64_t lanes_per_row, int64_t warps, int64_t ksplit,
                bool out_fp32) {
  auto y = at::empty({x.size(0), w.size(0)}, x.options().dtype(out_fp32 ? at::kFloat : at::kBFloat16));
  run(y, x, w, ws, counters, rows, unroll, lanes_per_row, warps, ksplit);
  return y;
}

void gemm_out(at::Tensor& y, const at::Tensor& x, const at::Tensor& w, const at::Tensor& ws,
              const at::Tensor& counters, int64_t rows, int64_t unroll, int64_t lanes_per_row, int64_t warps,
              int64_t ksplit) {
  run(y, x, w, ws, counters, rows, unroll, lanes_per_row, warps, ksplit);
}

at::Tensor gemm_meta(const at::Tensor& x, const at::Tensor& w, const at::Tensor&, const at::Tensor&, int64_t,
                     int64_t, int64_t, int64_t, int64_t, bool out_fp32) {
  return at::empty({x.size(0), w.size(0)}, x.options().dtype(out_fp32 ? at::kFloat : at::kBFloat16));
}

void gemm_out_meta(at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                   int64_t, int64_t, int64_t, int64_t, int64_t) {}

}  // namespace llg

TORCH_LIBRARY(lowlat_gemm, m) {
  m.def(
      "gemm(Tensor x, Tensor w, Tensor ws, Tensor counters, int rows, int unroll, int lanes_per_row, "
      "int warps, int ksplit, bool out_fp32=False) -> Tensor");
  m.def(
      "gemm_out(Tensor(a!) y, Tensor x, Tensor w, Tensor ws, Tensor counters, int rows, int unroll, "
      "int lanes_per_row, int warps, int ksplit) -> ()");
}

TORCH_LIBRARY_IMPL(lowlat_gemm, CUDA, m) {
  m.impl("gemm", &llg::gemm);
  m.impl("gemm_out", &llg::gemm_out);
}

TORCH_LIBRARY_IMPL(lowlat_gemm, Meta, m) {
  m.impl("gemm", &llg::gemm_meta);
  m.impl("gemm_out", &llg::gemm_out_meta);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
