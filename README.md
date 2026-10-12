# sm120-lowlat-gemm

Low-latency skinny BF16 GEMM for NVIDIA Blackwell **SM120** GPUs (RTX PRO 6000 Blackwell, RTX 50 series):

```
y[M, N] = x[M, K] @ w[N, K]^T        bf16 in, fp32 accumulate, bf16 or fp32 out, 1 <= M <= 16
```

Built for the decode and speculative-verify steps of LLM serving, where every projection is a memory-bound
matrix-vector product and the batch has a handful of tokens.

- Weights stream once from DRAM in 16-byte vectors (L1 bypass, 256-byte L2 prefetch); no weight repacking,
  `w` is the plain `nn.Linear` layout.
- Optional split-K with a deterministic in-kernel reduction (no atomics on data, no extra launch).
- **Batch invariant:** the summation order depends only on `(K, lanes_per_row, ksplit)`, which a plan fixes per
  `(N, K)`. Row `i` of an `M`-row call is bit-identical to the 1-row call on `x[i]`, for every `M` and every tile.
- CUDA-graph safe; a `torch.ops.lowlat_gemm` custom op with a Meta implementation.
- Optional vLLM plugin for Qwen3.8-Flash-Next (`qwen4_exp`) on SM120.

## Install

Requires PyTorch with CUDA 12.8+ (13.x recommended) and a CUDA toolkit with `nvcc`.

```bash
pip install --no-build-isolation .
# or, in a source tree:
python setup.py build_ext --inplace
```

| Variable | Meaning |
|---|---|
| `LLG_CUDA_ARCHS` | SM versions to build, `;`-separated (default `120`; e.g. `120;121`) |
| `LLG_PTXAS_VERBOSE=1` | print register / spill usage per kernel |

## Use

```python
import torch, lowlat_gemm

x = torch.randn(4, 2560, device="cuda", dtype=torch.bfloat16)
w = torch.randn(16384, 2560, device="cuda", dtype=torch.bfloat16)

y = lowlat_gemm.gemm(x, w)                            # [4, 16384] bf16
y32 = lowlat_gemm.gemm(x, w, out_dtype=torch.float32)
lowlat_gemm.gemm(x, w, out=y)                         # write into an existing buffer
lowlat_gemm.eligible(x, w)                            # False -> use torch.nn.functional.linear
```

Inputs: `x` is `[..., K]` with 1-16 rows, unit last-dim stride, row stride a multiple of 8, 16-byte aligned;
`w` is `[N, K]` with the same constraints; `K % 8 == 0`.

### Plans

A plan per `(N, K)` holds the summation order (`lanes_per_row`, `ksplit`) and a tile per token count
(`rows`, `unroll`, `warps`). Lookup order: `set_plan()`, then the plan file in `LLG_PLAN_FILE`, then a packaged
plan for the device, then a built-in heuristic.

```python
from lowlat_gemm import Plan, Tile
lowlat_gemm.set_plan(336, 10240, Plan(lanes_per_row=32, ksplit=8, tiles={1: Tile(1, 8, 4), 4: Tile(2, 4, 4)}))
lowlat_gemm.load_plans("plans.json")
```

Split-K plans use a per-device workspace. Before capturing a CUDA graph, size it once:

```python
lowlat_gemm.reserve_workspace(torch.device("cuda", 0), floats=ksplit * 16 * N, counters=N)
```

### Tuning and benchmarks

```bash
python tools/tune.py  --shapes shapes/qwen3.8-flash-next-tp1.json --out plans.json
python tools/bench.py --shapes shapes/qwen3.8-flash-next-tp1.json --plans plans.json
pytest tests
```

A shape file lists `{"name", "n", "k"}` entries. Timings replay a CUDA graph over weight copies that together
exceed the L2 cache, so every call reads its weight from DRAM.

## vLLM

Installing the package registers a `vllm.general_plugins` entry point; it does nothing unless enabled.

```bash
LLG_VLLM=1 LLG_PLAN_FILE=plans.json vllm serve <Qwen3.8-Flash-Next checkpoint> ...
```

On SM120 it routes the model's unquantized BF16 projections and LM head (and the MTP drafter's) to
`lowlat_gemm` for batches of up to 16 tokens; larger batches, biased layers and `VLLM_BATCH_INVARIANT=1` keep
vLLM's implementation.

| Variable | Meaning |
|---|---|
| `LLG_VLLM=1` | enable the plugin |
| `LLG_VLLM_MODELS` | hook targets, `,`-separated (default `qwen4_exp`) |
| `LLG_PLAN_FILE` | plan file from `tools/tune.py` |

## License

Apache-2.0
