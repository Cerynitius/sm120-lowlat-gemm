// SPDX-License-Identifier: Apache-2.0
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "params.h"
#include "quant.h"
#include "tc_params.h"

namespace llg {

namespace {

bool aligned(const void* ptr, uintptr_t n) { return (reinterpret_cast<uintptr_t>(ptr) & (n - 1)) == 0; }

int dtype_code(at::ScalarType t) {
  switch (t) {
    case at::kBFloat16: return DT_BF16;
    case at::kHalf: return DT_F16;
    case at::kFloat: return DT_F32;
    default: return DT_NONE;
  }
}

void check_same_device(const at::Tensor& ref, const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.device() == ref.device(), "lowlat_gemm: ", name, " must be on ", ref.device());
}

// ================================================================================================================
// Skinny CUDA-core GEMM (bf16 / f16, M <= 16)

constexpr int64_t kSkinnyMaxTokens = 16;

void skinny_check_config(int64_t rows, int64_t unroll, int64_t lanes_per_row, int64_t warps, int64_t ksplit) {
  const int64_t ru = rows * 16 + unroll;
  TORCH_CHECK(ru == 18 || ru == 20 || ru == 24 || ru == 34 || ru == 36 || ru == 66,
              "lowlat_gemm: unsupported (rows, unroll) = (", rows, ", ", unroll,
              "); supported: (1,2) (1,4) (1,8) (2,2) (2,4) (4,2)");
  TORCH_CHECK(lanes_per_row == 8 || lanes_per_row == 16 || lanes_per_row == 32,
              "lowlat_gemm: lanes_per_row must be 8, 16 or 32, got ", lanes_per_row);
  TORCH_CHECK(warps >= 1 && warps <= 8, "lowlat_gemm: warps must be in [1, 8], got ", warps);
  TORCH_CHECK(ksplit >= 1 && ksplit <= 64, "lowlat_gemm: ksplit must be in [1, 64], got ", ksplit);
}

void skinny_run(const at::Tensor& y, const at::Tensor& x, const at::Tensor& w, const at::Tensor& ws,
                const at::Tensor& counters, int64_t rows, int64_t unroll, int64_t lanes_per_row, int64_t warps,
                int64_t ksplit) {
  skinny_check_config(rows, unroll, lanes_per_row, warps, ksplit);
  TORCH_CHECK(x.is_cuda(), "lowlat_gemm: x must be on CUDA");
  check_same_device(x, w, "w");
  check_same_device(x, y, "y");
  const auto in_t = x.scalar_type();
  TORCH_CHECK((in_t == at::kBFloat16 || in_t == at::kHalf) && w.scalar_type() == in_t,
              "lowlat_gemm.skinny: x and w must both be bfloat16 or both float16, got ", in_t, " and ",
              w.scalar_type());
  TORCH_CHECK(y.scalar_type() == in_t || y.scalar_type() == at::kFloat,
              "lowlat_gemm.skinny: output must be ", in_t, " or float32, got ", y.scalar_type());
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && y.dim() == 2, "lowlat_gemm: x, w, y must be 2-D");
  const int64_t m = x.size(0), k = x.size(1), n = w.size(0);
  TORCH_CHECK(w.size(1) == k, "lowlat_gemm: x is [", m, ", ", k, "] but w is [", n, ", ", w.size(1), "]");
  TORCH_CHECK(y.size(0) == m && y.size(1) == n, "lowlat_gemm: output must be [", m, ", ", n, "]");
  TORCH_CHECK(m >= 1 && m <= kSkinnyMaxTokens, "lowlat_gemm.skinny: M must be in [1, 16], got ", m);
  TORCH_CHECK(k % 8 == 0 && k > 0, "lowlat_gemm.skinny: K must be a positive multiple of 8, got ", k);
  TORCH_CHECK(n >= 1 && n <= INT32_MAX && k <= INT32_MAX, "lowlat_gemm: N or K too large");
  TORCH_CHECK(x.stride(1) == 1 && w.stride(1) == 1 && y.stride(1) == 1,
              "lowlat_gemm: x, w, y need unit stride in the last dimension");
  TORCH_CHECK(x.stride(0) % 8 == 0 && w.stride(0) % 8 == 0,
              "lowlat_gemm: x and w row strides must be multiples of 8 elements");
  TORCH_CHECK(aligned(x.data_ptr(), 16) && aligned(w.data_ptr(), 16), "lowlat_gemm: x and w must be 16-byte aligned");
  const int64_t rows_per_block = warps * (32 / lanes_per_row) * rows;
  const int64_t columns = (n + rows_per_block - 1) / rows_per_block;
  if (ksplit > 1) {
    check_same_device(x, ws, "workspace");
    check_same_device(x, counters, "counters");
    TORCH_CHECK(ws.scalar_type() == at::kFloat && ws.is_contiguous() && counters.scalar_type() == at::kInt &&
                    counters.is_contiguous(),
                "lowlat_gemm: workspace must be contiguous float32, counters contiguous int32");
    TORCH_CHECK(ws.numel() >= ksplit * m * n, "lowlat_gemm: workspace has ", ws.numel(), " floats, needs ",
                ksplit * m * n);
    TORCH_CHECK(counters.numel() >= columns, "lowlat_gemm: counters has ", counters.numel(), ", needs ", columns);
  }

  Params p{};
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
  p.f16 = in_t == at::kHalf;

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
  TORCH_CHECK(ok, "lowlat_gemm.skinny: no kernel for M=", m, " rows=", rows, " unroll=", unroll);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ================================================================================================================
// Tensor-core GEMM

struct KindInfo {
  int a_bits, b_bits;
  bool a_sf, b_sf;  // block scales present
  int k_align;      // K must be a multiple of this
};

KindInfo kind_info(int64_t kind) {
  switch (kind) {
    case KIND_BF16:
    case KIND_F16: return {16, 16, false, false, 8};
    case KIND_FP8:
    case KIND_INT8: return {8, 8, false, false, 16};
    case KIND_NVFP4: return {4, 4, true, true, 64};
    case KIND_W8A16_BF16:
    case KIND_W8A16_F16: return {8, 16, false, false, 16};
    case KIND_W4A16_BF16:
    case KIND_W4A16_F16: return {4, 16, true, false, 64};
    default: TORCH_CHECK(false, "lowlat_gemm: unknown kind ", kind);
  }
}

bool tc_dispatch(const TcParams& p, int64_t kind, int cfg, cudaStream_t s) {
  switch (kind) {
    case KIND_BF16: return tc_launch_kind<KIND_BF16>(p, cfg, s);
    case KIND_F16: return tc_launch_kind<KIND_F16>(p, cfg, s);
    case KIND_FP8: return tc_launch_kind<KIND_FP8>(p, cfg, s);
    case KIND_INT8: return tc_launch_kind<KIND_INT8>(p, cfg, s);
    case KIND_NVFP4: return tc_launch_kind<KIND_NVFP4>(p, cfg, s);
    case KIND_W8A16_BF16: return tc_launch_kind<KIND_W8A16_BF16>(p, cfg, s);
    case KIND_W8A16_F16: return tc_launch_kind<KIND_W8A16_F16>(p, cfg, s);
    case KIND_W4A16_BF16: return tc_launch_kind<KIND_W4A16_BF16>(p, cfg, s);
    case KIND_W4A16_F16: return tc_launch_kind<KIND_W4A16_F16>(p, cfg, s);
    default: return false;
  }
}

void check_operand(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.dim() == 2, "lowlat_gemm: ", name, " must be 2-D");
  TORCH_CHECK(t.stride(1) == 1, "lowlat_gemm: ", name, " needs unit stride in the last dimension");
  TORCH_CHECK((t.stride(0) * t.element_size()) % 16 == 0, "lowlat_gemm: ", name,
              " row stride must be a multiple of 16 bytes");
  TORCH_CHECK(aligned(t.data_ptr(), 16), "lowlat_gemm: ", name, " must be 16-byte aligned");
}

void check_scales(const at::Tensor& t, int64_t rows, int64_t kblocks, const char* name) {
  TORCH_CHECK(t.dim() == 2 && t.element_size() == 1, "lowlat_gemm: ", name, " must be a 2-D 1-byte (e4m3) tensor");
  TORCH_CHECK(t.size(0) == rows && t.size(1) == kblocks, "lowlat_gemm: ", name, " must be [", rows, ", ", kblocks,
              "], got ", t.sizes());
  TORCH_CHECK(t.stride(1) == 1 && t.stride(0) % 4 == 0 && aligned(t.data_ptr(), 4), "lowlat_gemm: ", name,
              " needs unit column stride, a row stride that is a multiple of 4 and 4-byte alignment");
}

const float* opt_vec(const std::optional<at::Tensor>& t, const at::Tensor& ref, int64_t full, int* len,
                     const char* name) {
  if (!t.has_value()) {
    *len = 0;
    return nullptr;
  }
  check_same_device(ref, *t, name);
  TORCH_CHECK(t->scalar_type() == at::kFloat && t->is_contiguous(), "lowlat_gemm: ", name,
              " must be a contiguous float32 tensor");
  TORCH_CHECK(t->numel() == 1 || t->numel() == full, "lowlat_gemm: ", name, " must have 1 or ", full,
              " elements, got ", t->numel());
  *len = t->numel() == 1 ? 1 : static_cast<int>(full);
  return t->data_ptr<float>();
}

void tc_gemm_impl(at::Tensor& y, const at::Tensor& a, const at::Tensor& b, const std::optional<at::Tensor>& a_sf,
                  const std::optional<at::Tensor>& b_sf, const std::optional<at::Tensor>& row_scale,
                  const std::optional<at::Tensor>& col_scale, const std::optional<at::Tensor>& bias,
                  const std::optional<at::Tensor>& alpha_t, double alpha, int64_t kind, bool swap, bool bias_on_p,
                  int64_t cfg, int64_t stages, int64_t ksplit, const at::Tensor& ws, const at::Tensor& counters) {
  const KindInfo ki = kind_info(kind);
  TORCH_CHECK(cfg >= 0 && cfg < kNumCfgs, "lowlat_gemm: cfg must be in [0, ", kNumCfgs, "), got ", cfg);
  TORCH_CHECK(ksplit >= 1 && ksplit <= 64, "lowlat_gemm: ksplit must be in [1, 64], got ", ksplit);
  TORCH_CHECK(a.is_cuda(), "lowlat_gemm: operands must be on CUDA");
  check_same_device(a, b, "b");
  check_same_device(a, y, "y");
  check_operand(a, "a");
  check_operand(b, "b");
  const int64_t P = a.size(0), Q = b.size(0);
  const int64_t K = a.size(1) * a.element_size() * 8 / ki.a_bits;
  const int64_t Kb = b.size(1) * b.element_size() * 8 / ki.b_bits;
  TORCH_CHECK(Kb == K, "lowlat_gemm: a holds K=", K, " but b holds K=", Kb);
  TORCH_CHECK(K > 0 && K % ki.k_align == 0, "lowlat_gemm: K must be a positive multiple of ", ki.k_align,
              " for this kind, got ", K);
  TORCH_CHECK(P >= 1 && Q >= 1 && P <= INT32_MAX && Q <= INT32_MAX && K <= INT32_MAX,
              "lowlat_gemm: empty or too large problem");
  TORCH_CHECK(y.dim() == 2 && y.stride(1) == 1, "lowlat_gemm: y must be 2-D with unit last stride");
  TORCH_CHECK(swap ? (y.size(0) == Q && y.size(1) == P) : (y.size(0) == P && y.size(1) == Q), "lowlat_gemm: y must be [",
              swap ? Q : P, ", ", swap ? P : Q, "], got ", y.sizes());
  const int out_dt = dtype_code(y.scalar_type());
  TORCH_CHECK(out_dt != DT_NONE, "lowlat_gemm: y must be bfloat16, float16 or float32");

  TcParams p{};
  p.a = a.data_ptr();
  p.b = b.data_ptr();
  p.lda = a.stride(0) * a.element_size();
  p.ldb = b.stride(0) * b.element_size();
  if (ki.a_sf) {
    TORCH_CHECK(a_sf.has_value(), "lowlat_gemm: this kind needs block scales for a");
    check_same_device(a, *a_sf, "a_sf");
    check_scales(*a_sf, P, K / 16, "a_sf");
    p.a_sf = a_sf->data_ptr();
    p.lda_sf = a_sf->stride(0);
  }
  if (ki.b_sf) {
    TORCH_CHECK(b_sf.has_value(), "lowlat_gemm: this kind needs block scales for b");
    check_same_device(a, *b_sf, "b_sf");
    check_scales(*b_sf, Q, K / 16, "b_sf");
    p.b_sf = b_sf->data_ptr();
    p.ldb_sf = b_sf->stride(0);
  }
  p.row_scale = opt_vec(row_scale, a, P, &p.row_scale_len, "row_scale");
  p.col_scale = opt_vec(col_scale, a, Q, &p.col_scale_len, "col_scale");
  if (bias.has_value()) {
    check_same_device(a, *bias, "bias");
    p.bias_dtype = dtype_code(bias->scalar_type());
    TORCH_CHECK(p.bias_dtype != DT_NONE && bias->is_contiguous() && bias->numel() == (bias_on_p ? P : Q),
                "lowlat_gemm: bias must be a contiguous bf16/f16/f32 vector of ", bias_on_p ? P : Q, " elements");
    p.bias = bias->data_ptr();
  }
  p.bias_on_p = bias_on_p;
  if (alpha_t.has_value()) {
    check_same_device(a, *alpha_t, "alpha");
    TORCH_CHECK(alpha_t->scalar_type() == at::kFloat && alpha_t->numel() == 1,
                "lowlat_gemm: alpha tensor must be one float32");
    p.alpha_ptr = alpha_t->data_ptr<float>();
  }
  p.alpha = static_cast<float>(alpha);
  p.y = y.data_ptr();
  p.ldy = y.stride(0);
  p.out_dtype = out_dt;
  p.swap = swap;
  p.P = static_cast<int>(P);
  p.Q = static_cast<int>(Q);
  p.K = static_cast<int>(K);
  p.ksplit = static_cast<int>(ksplit);
  p.stages = static_cast<int>(stages);
  if (ksplit > 1) {
    int bp, bq, threads;
    tc_cfg_tile(static_cast<int>(cfg), &bp, &bq, &threads);
    const int64_t tiles = ((P + bp - 1) / bp) * ((Q + bq - 1) / bq);
    check_same_device(a, ws, "workspace");
    check_same_device(a, counters, "counters");
    TORCH_CHECK(ws.element_size() == 4 && ws.is_contiguous() && counters.scalar_type() == at::kInt &&
                    counters.is_contiguous(),
                "lowlat_gemm: workspace must be a contiguous 4-byte tensor, counters contiguous int32");
    TORCH_CHECK(ws.numel() >= ksplit * P * Q, "lowlat_gemm: workspace has ", ws.numel(), " elements, needs ",
                ksplit * P * Q);
    TORCH_CHECK(counters.numel() >= tiles, "lowlat_gemm: counters has ", counters.numel(), ", needs ", tiles);
    p.ws = ws.data_ptr();
    p.counters = reinterpret_cast<unsigned*>(counters.data_ptr<int>());
  }
  const c10::cuda::CUDAGuard guard(a.device());
  const bool ok = tc_dispatch(p, kind, static_cast<int>(cfg), c10::cuda::getCurrentCUDAStream());
  if (!ok) {
    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(false, "lowlat_gemm: config ", cfg, " cannot be launched for this kind: ",
                err == cudaSuccess ? "stage does not fit shared memory" : cudaGetErrorString(err));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ================================================================================================================
// Quantization

void quant_impl(const at::Tensor& x, at::Tensor& q, const std::optional<at::Tensor>& sf,
                const std::optional<at::Tensor>& scale_in, const std::optional<at::Tensor>& scale_out, int64_t mode,
                double min_scale) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.stride(1) == 1,
              "lowlat_gemm.quant: x must be a 2-D CUDA tensor with unit last stride");
  check_same_device(x, q, "q");
  const int in_dt = dtype_code(x.scalar_type());
  TORCH_CHECK(in_dt != DT_NONE, "lowlat_gemm.quant: x must be bf16, f16 or f32");
  const int64_t M = x.size(0), K = x.size(1);
  QuantParams p{};
  p.mode = static_cast<int>(mode);
  p.in_dtype = in_dt;
  p.x = x.data_ptr();
  p.ldx = x.stride(0);
  p.M = static_cast<int>(M);
  p.K = static_cast<int>(K);
  p.min_scale = static_cast<float>(min_scale);
  TORCH_CHECK(q.dim() == 2 && q.size(0) == M && q.stride(1) == 1 && q.element_size() == 1,
              "lowlat_gemm.quant: q must be a 2-D 1-byte tensor with M rows");
  p.q = q.data_ptr();
  p.ldq = q.stride(0);
  auto need_scale_out = [&]() {
    TORCH_CHECK(scale_out.has_value() && scale_out->scalar_type() == at::kFloat && scale_out->is_contiguous() &&
                    scale_out->numel() == M,
                "lowlat_gemm.quant: per-token modes need scale_out float32[M]");
    check_same_device(x, *scale_out, "scale_out");
    p.scale_out = scale_out->data_ptr<float>();
  };
  auto need_scale_in = [&]() {
    TORCH_CHECK(scale_in.has_value() && scale_in->scalar_type() == at::kFloat && scale_in->numel() == 1,
                "lowlat_gemm.quant: this mode needs scale_in float32[1]");
    check_same_device(x, *scale_in, "scale_in");
    p.scale_in = scale_in->data_ptr<float>();
  };
  switch (mode) {
    case QUANT_FP8_STATIC:
      TORCH_CHECK(q.size(1) == K, "lowlat_gemm.quant: q must be [M, K]");
      need_scale_in();
      break;
    case QUANT_FP8_PER_TOKEN:
    case QUANT_INT8_PER_TOKEN:
      TORCH_CHECK(q.size(1) == K, "lowlat_gemm.quant: q must be [M, K]");
      need_scale_out();
      break;
    case QUANT_NVFP4:
      TORCH_CHECK(K % 16 == 0, "lowlat_gemm.quant: nvfp4 needs K % 16 == 0");
      TORCH_CHECK(q.size(1) == K / 2 && q.stride(0) % 8 == 0 && aligned(q.data_ptr(), 8),
                  "lowlat_gemm.quant: q must be [M, K/2] bytes with 8-byte aligned rows");
      TORCH_CHECK(sf.has_value() && sf->dim() == 2 && sf->size(0) == M && sf->size(1) == K / 16 &&
                      sf->element_size() == 1 && sf->stride(1) == 1,
                  "lowlat_gemm.quant: nvfp4 needs sf [M, K/16] bytes");
      check_same_device(x, *sf, "sf");
      p.sf_out = sf->data_ptr();
      p.ldsf = sf->stride(0);
      need_scale_in();
      break;
    default: TORCH_CHECK(false, "lowlat_gemm.quant: unknown mode ", mode);
  }
  if (M == 0) return;
  const c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(quant_launch(p, c10::cuda::getCurrentCUDAStream()), "lowlat_gemm.quant: launch failed");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// ---- registered entry points -----------------------------------------------------------------------------------

at::Tensor skinny(const at::Tensor& x, const at::Tensor& w, const at::Tensor& ws, const at::Tensor& counters,
                  int64_t rows, int64_t unroll, int64_t lanes_per_row, int64_t warps, int64_t ksplit, bool out_fp32) {
  auto y = at::empty({x.size(0), w.size(0)}, x.options().dtype(out_fp32 ? at::kFloat : x.scalar_type()));
  skinny_run(y, x, w, ws, counters, rows, unroll, lanes_per_row, warps, ksplit);
  return y;
}

void skinny_out(at::Tensor& y, const at::Tensor& x, const at::Tensor& w, const at::Tensor& ws,
                const at::Tensor& counters, int64_t rows, int64_t unroll, int64_t lanes_per_row, int64_t warps,
                int64_t ksplit) {
  skinny_run(y, x, w, ws, counters, rows, unroll, lanes_per_row, warps, ksplit);
}

at::Tensor skinny_meta(const at::Tensor& x, const at::Tensor& w, const at::Tensor&, const at::Tensor&, int64_t,
                       int64_t, int64_t, int64_t, int64_t, bool out_fp32) {
  return at::empty({x.size(0), w.size(0)}, x.options().dtype(out_fp32 ? at::kFloat : x.scalar_type()));
}

void tc_gemm(at::Tensor& y, const at::Tensor& a, const at::Tensor& b, const std::optional<at::Tensor>& a_sf,
             const std::optional<at::Tensor>& b_sf, const std::optional<at::Tensor>& row_scale,
             const std::optional<at::Tensor>& col_scale, const std::optional<at::Tensor>& bias,
             const std::optional<at::Tensor>& alpha_t, double alpha, int64_t kind, bool swap, bool bias_on_p,
             int64_t cfg, int64_t stages, int64_t ksplit, const at::Tensor& ws, const at::Tensor& counters) {
  tc_gemm_impl(y, a, b, a_sf, b_sf, row_scale, col_scale, bias, alpha_t, alpha, kind, swap, bias_on_p, cfg, stages,
               ksplit, ws, counters);
}

void quant(const at::Tensor& x, at::Tensor& q, const std::optional<at::Tensor>& sf,
           const std::optional<at::Tensor>& scale_in, const std::optional<at::Tensor>& scale_out, int64_t mode,
           double min_scale) {
  quant_impl(x, q, sf, scale_in, scale_out, mode, min_scale);
}

std::vector<int64_t> tc_info(int64_t kind, int64_t cfg) {
  int bp, bq, threads;
  tc_cfg_tile(static_cast<int>(cfg), &bp, &bq, &threads);
  return {bp, bq, threads, tc_stage_bytes(static_cast<int>(kind), static_cast<int>(cfg))};
}

}  // namespace llg

TORCH_LIBRARY(lowlat_gemm, m) {
  m.def(
      "skinny(Tensor x, Tensor w, Tensor ws, Tensor counters, int rows, int unroll, int lanes_per_row, "
      "int warps, int ksplit, bool out_fp32=False) -> Tensor");
  m.def(
      "skinny_out(Tensor(a!) y, Tensor x, Tensor w, Tensor ws, Tensor counters, int rows, int unroll, "
      "int lanes_per_row, int warps, int ksplit) -> ()");
  m.def(
      "tc_gemm(Tensor(a!) y, Tensor a, Tensor b, Tensor? a_sf, Tensor? b_sf, Tensor? row_scale, "
      "Tensor? col_scale, Tensor? bias, Tensor? alpha_t, float alpha, int kind, bool swap, bool bias_on_p, "
      "int cfg, int stages, int ksplit, Tensor ws, Tensor counters) -> ()");
  m.def(
      "quant(Tensor x, Tensor(a!) q, Tensor(b!)? sf, Tensor? scale_in, Tensor(c!)? scale_out, int mode, "
      "float min_scale) -> ()");
  m.def("tc_info(int kind, int cfg) -> int[]", &llg::tc_info);
}

TORCH_LIBRARY_IMPL(lowlat_gemm, CUDA, m) {
  m.impl("skinny", &llg::skinny);
  m.impl("skinny_out", &llg::skinny_out);
  m.impl("tc_gemm", &llg::tc_gemm);
  m.impl("quant", &llg::quant);
}

TORCH_LIBRARY_IMPL(lowlat_gemm, Meta, m) { m.impl("skinny", &llg::skinny_meta); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
