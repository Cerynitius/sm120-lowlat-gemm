# sm120-lowlat-gemm

Low-latency GEMM for NVIDIA Blackwell **SM120** GPUs (RTX PRO 6000 Blackwell, RTX 50 series), any M, with the
precisions used in LLM inference:

| x (activation) | w (weight) | name |
|---|---|---|
| bf16 / f16 | same | W16A16 |
| bf16 / f16 | float8_e4m3fn | W8A16 |
| bf16 / f16 | NVFP4 (e2m1 + e4m3 scale per 16) | W4A16 |
| float8_e4m3fn | float8_e4m3fn | W8A8 FP8 |
| int8 | int8 | W8A8 INT8 |
| NVFP4 | NVFP4 | W4A4 |

```
y = alpha * (x * x_scale) @ (w * w_scale)^T + bias        fp32 accumulate (int32 for INT8)
```

- Tensor-core kernel (`mma.sync` for every precision, including the SM120 block-scaled NVFP4 MMA) with a
  cp.async multi-stage pipeline. Small batches put the weight on the 16-row MMA side, so 1-8 tokens fill one
  m16n8 tile; large batches use square tiles.
- Mixed precision (W8A16, W4A16) widens the weight exactly in registers; no dequantized copy.
- Split-K with a deterministic in-kernel reduction; results are run-to-run identical.
- A CUDA-core streaming kernel for W16A16 with at most 16 rows whose results are bit-identical for every M.
- Per-(precision, N, K) plans pick engine and tiling per M; a tuner writes them.
- Activation quantizers: FP8 (static or per-token), INT8 (per-token), NVFP4.
- CUDA-graph safe custom ops (`torch.ops.lowlat_gemm`); optional vLLM plugin.

## Install

Requires PyTorch with CUDA 12.8+ (13.x recommended) and `nvcc`. Arch-specific targets (`sm_120a`) are required
for the FP8 / FP4 instructions.

```bash
pip install --no-build-isolation .
# or in a source tree:
python setup.py build_ext --inplace
```

| Variable | Meaning |
|---|---|
| `LLG_CUDA_ARCHS` | SM targets, `;`-separated (default `120a`; e.g. `120a;121a`) |
| `LLG_PTXAS_VERBOSE=1` | print register / spill usage per kernel |

## Use

```python
import torch, lowlat_gemm as L

x = torch.randn(4, 4096, device="cuda", dtype=torch.bfloat16)

# W16A16
w = torch.randn(11008, 4096, device="cuda", dtype=torch.bfloat16)
y = L.gemm(x, w)

# W8A16, per-channel weight scale
w8 = w.to(torch.float8_e4m3fn)
y = L.gemm(x, w8, w_scale=torch.ones(11008, device="cuda"))

# W8A8 FP8, dynamic per-token activation scale
xq, xs = L.quantize_fp8(x)
y = L.gemm(xq, w8, x_scale=xs, w_scale=torch.ones(11008, device="cuda"))

# W8A8 INT8
xq, xs = L.quantize_int8(x)
y = L.gemm(xq, w_int8, x_scale=xs, w_scale=w_int8_scale)

# NVFP4 weight + block scales ([N, K/16] e4m3), global scale as w_scale
y = L.gemm(x, w_fp4, w_block_scale=w_sf, w_scale=weight_scale_2)           # W4A16
g = torch.tensor([global_scale], device="cuda")
xq, x_sf = L.quantize_nvfp4(x, g)
y = L.gemm(xq, w_fp4, x_block_scale=x_sf, w_block_scale=w_sf,
           x_scale=1 / global_scale, w_scale=weight_scale_2)               # W4A4
```

| Argument | Meaning |
|---|---|
| `x_scale` | float, or float32 tensor with 1 or M elements |
| `w_scale` | float, or float32 tensor with 1 or N elements |
| `x_block_scale`, `w_block_scale` | NVFP4 block scales, `[rows, K/16]` float8_e4m3fn |
| `alpha` | float, or 1-element float32 tensor (stays valid inside CUDA graphs) |
| `bias` | `[N]` bf16 / f16 / f32 |
| `out_dtype` | bf16 (default for quantized inputs), f16, f32 |
| `out` | preallocated `[M, N]` output |
| `prefetch`, `prefetch_bytes` | a contiguous tensor (e.g. the next GEMM's weight) to pull into L2 once this GEMM's weight is streaming; first `prefetch_bytes` (0 = all) |

### Launch options

```python
L.set_pdl(True)   # programmatic dependent launch for later calls (process-wide)
```

With PDL a GEMM may start while the previous kernel is still running and stream its weight early. Enable it only
when no kernel still in flight writes the weights (inference with fixed weights). CUDA graphs keep the setting
they were captured with.

FP4 tensors are `torch.float4_e2m1fn_x2` (or `uint8` together with a block scale), two values per byte, low
nibble first, shape `[rows, K/2]`. K must be a multiple of 8 (16-bit), 16 (8-bit) or 64 (FP4); rows must be
16-byte aligned. `L.supports(x, w)` checks this.

### Plans

```python
from lowlat_gemm import Plan, TcChoice
L.set_plan("bf16", 11008, 4096, Plan(tc={1: TcChoice(cfg=0, swap=True, stages=6, ksplit=4)}))
L.load_plans("plans.json")   # takes precedence over the packaged plans
```

`TcChoice(..., cluster=True)` and `SkinnyOrder(..., cluster=True)` reduce split-K partials inside a thread block
cluster (`ksplit <= 8`, no workspace); results are identical to the workspace path.

Lookup order: `set_plan()`, the file in `LLG_PLAN_FILE`, a packaged plan for the device, built-in heuristics.
A tuned entry serves token counts from its M up to the next tuned M; above twice the largest tuned M the heuristics
apply. `L.prefer_lowlat(kind, n, k, m)` is False where the plan recorded the vendor library as faster.
Before capturing a CUDA graph, size the split-K workspace:

```python
L.reserve_workspace(torch.device("cuda", 0), *L.workspace_need("bf16", 11008, 4096, max_m=16))
```

### Tuning, benchmarks, tests

```bash
python tools/tune.py  --shapes shapes/qwen3.8-flash-next-tp1.json --kinds bf16,fp8,nvfp4 --out plans.json \
    [--tokens 1,2,4,8,16] [--max-calls 256]
python tools/bench.py --shapes shapes/qwen3.8-flash-next-tp1.json --kinds bf16,fp8,nvfp4 --plans plans.json \
    --json bench.json
python tools/prefer_vendor.py --plans plans.json --bench bench.json   # record where the vendor library wins
python tools/decode_step.py --step shapes/qwen3.8-flash-next-tp1-step.json --tokens 1,2,4,8
pytest tests
python tests/emulate_tc.py        # CPU thread-level emulation of the tensor-core kernel's indexing
```

A shape file lists `{"name", "n", "k"}` entries.

| bench.py / tune.py option | Meaning |
|---|---|
| `--pdl` | programmatic dependent launch |
| `--between write` | a small kernel rewrites x before every GEMM; its own time is subtracted |
| `--prefetch-next` (bench) | every call prefetches the next call's weight (`--prefetch-max-mb`) |
| `--max-calls N` (tune) | cap the calls per timed graph |

`decode_step.py` replays the projections of one decode step in execution order (per-layer weights, producer
kernels and a routed-expert stand-in between them) for the vendor GEMM and lowlat_gemm with and without PDL and
next-weight prefetch.

## vLLM

Installing the package registers a `vllm.general_plugins` entry point that does nothing unless enabled.

```bash
LLG_VLLM=1 LLG_PLAN_FILE=plans.json vllm serve <Qwen3.8-Flash-Next checkpoint> ...
```

On SM120 it routes the model's unquantized BF16 projections and LM head (and the MTP drafter's) to
`lowlat_gemm` for batches up to `LLG_VLLM_MAX_TOKENS` (default 16), except token counts the plan leaves to the
vendor library.

| Variable | Meaning |
|---|---|
| `LLG_VLLM=1` | enable the plugin |
| `LLG_VLLM_MODELS` | hook targets, `,`-separated (default `qwen4_exp`) |
| `LLG_VLLM_MAX_TOKENS` | largest batch routed to lowlat_gemm (default 16) |
| `LLG_PLAN_FILE` | plan file from `tools/tune.py` |
| `LLG_PDL` | programmatic dependent launch (default 1) |
| `LLG_PREFETCH` | each routed GEMM prefetches the next projection's weight into L2; order recorded from eager forwards (default 1) |
| `LLG_PREFETCH_MAX_MB` | cap per prefetch (default 16) |

## License

Apache-2.0
