# SPDX-License-Identifier: Apache-2.0
"""vLLM general plugin: route unquantized BF16/FP16 projections of small batches to lowlat_gemm on SM120.

Enabled only with LLG_VLLM=1. Hooks the Qwen4Exp (Qwen3.8-Flash-Next) low-latency GEMM entry point, which upstream
vLLM fills only for SM90/SM103 at TP=4 shapes. Calls with more than LLG_VLLM_MAX_TOKENS tokens, a bias,
VLLM_BATCH_INVARIANT=1, or a token count where the plan measured the vendor GEMM as faster keep the upstream
implementation.

    LLG_VLLM=1               enable
    LLG_VLLM_MODELS=...      comma-separated hook targets (default "qwen4_exp")
    LLG_VLLM_MAX_TOKENS=16   largest batch routed to lowlat_gemm
    LLG_PLAN_FILE=...        tuned plan file (tools/tune.py); otherwise the built-in defaults
    LLG_PDL=1                programmatic dependent launch (weights are fixed during serving; 0 disables)
    LLG_PREFETCH=1           each routed GEMM pulls the next BF16 projection's weight into L2 (0 disables); the order
                             is recorded from eager forwards (profiling, capture warm-ups)
    LLG_PREFETCH_MAX_MB=16   cap on the bytes prefetched per GEMM
"""

from __future__ import annotations

import logging
import os

MAX_TOKENS = int(os.environ.get("LLG_VLLM_MAX_TOKENS", "16"))
PREFETCH = os.environ.get("LLG_PREFETCH", "1") == "1"
PREFETCH_MAX = int(os.environ.get("LLG_PREFETCH_MAX_MB", "16")) << 20

# forward order of the swapped projections: weight data_ptr -> weight of the projection that runs next
_next_weight: dict = {}
_last_weight: list = [None]


def _record(w) -> None:
    import torch

    if torch.cuda.is_current_stream_capturing():
        return
    prev = _last_weight[0]
    if prev is not None and prev is not w:
        _next_weight.setdefault(prev.data_ptr(), w)
    _last_weight[0] = w

logger = logging.getLogger("lowlat_gemm")


def _is_sm120() -> bool:
    import torch

    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 12


def _make_methods():
    import vllm.envs as envs
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod

    import lowlat_gemm

    class _Apply:
        def apply(self, layer, x, bias=None):
            if PREFETCH:
                _record(layer.weight)
            m = x.numel() // max(1, x.shape[-1])
            if (
                bias is None
                and not envs.VLLM_BATCH_INVARIANT
                and 1 <= m <= MAX_TOKENS
                and lowlat_gemm.supports(x, layer.weight)
                and lowlat_gemm.prefer_lowlat(lowlat_gemm.kind_of(x, layer.weight), *layer.weight.shape, m, x.device)
            ):
                nxt = _next_weight.get(layer.weight.data_ptr()) if PREFETCH else None
                if nxt is not None:
                    nbytes = min(PREFETCH_MAX, nxt.numel() * nxt.element_size())
                    return lowlat_gemm.gemm(x, layer.weight, prefetch=nxt, prefetch_bytes=nbytes)
                return lowlat_gemm.gemm(x, layer.weight)
            return super().apply(layer, x, bias)

    class LowLatLinearMethod(_Apply, UnquantizedLinearMethod):
        pass

    class LowLatEmbeddingMethod(_Apply, UnquantizedEmbeddingMethod):
        pass

    return UnquantizedLinearMethod, UnquantizedEmbeddingMethod, LowLatLinearMethod, LowLatEmbeddingMethod


def install(module) -> int:
    """Swap the quant method of every unquantized BF16 linear / LM head under `module`. Returns the count."""
    import torch
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    import lowlat_gemm

    lin_cls, emb_cls, lowlat_lin, lowlat_emb = _make_methods()
    device = torch.device("cuda", torch.cuda.current_device())
    swapped, elems, counters = 0, 0, 0
    for child in module.modules():
        is_linear = isinstance(child, LinearBase) and type(child.quant_method) is lin_cls
        is_head = isinstance(child, ParallelLMHead) and type(child.quant_method) is emb_cls
        if not (is_linear or is_head):
            continue
        weight = getattr(child, "weight", None)
        if weight is None or weight.dim() != 2 or weight.dtype not in (torch.bfloat16, torch.float16):
            continue
        n, k = weight.shape
        kind = "bf16" if weight.dtype == torch.bfloat16 else "f16"
        if k % lowlat_gemm.KINDS[kind][2]:
            continue
        child.quant_method = lowlat_lin() if is_linear else lowlat_emb()
        swapped += 1
        e, c = lowlat_gemm.workspace_need(kind, n, k, MAX_TOKENS, device)
        elems, counters = max(elems, e), max(counters, c)
    # CUDA graph capture runs before any eager small-batch call, so the split-K buffers are sized here.
    lowlat_gemm.reserve_workspace(device, elems, counters)
    lowlat_gemm.set_pdl(os.environ.get("LLG_PDL", "1") == "1")
    return swapped


def _hook_qwen4_exp() -> bool:
    try:
        import vllm.models.qwen4_exp.nvidia.low_latency_gemm as ll
        import vllm.models.qwen4_exp.nvidia.model as model_mod
        import vllm.models.qwen4_exp.nvidia.mtp as mtp_mod
    except ImportError as e:
        logger.warning("lowlat_gemm: qwen4_exp hook unavailable: %s: %s", type(e).__name__, e)
        return False
    upstream = ll.enable_qwen4_exp_low_latency_gemm

    def enable(module, dtype):
        import torch

        upstream(module, dtype)
        if dtype != torch.bfloat16 or not _is_sm120():
            return
        n = install(module)
        logger.info("lowlat_gemm: %d BF16 projections of %s routed to lowlat_gemm", n, type(module).__name__)

    model_mod.enable_qwen4_exp_low_latency_gemm = enable
    mtp_mod.enable_qwen4_exp_low_latency_gemm = enable
    return True


_HOOKS = {"qwen4_exp": _hook_qwen4_exp}


def register() -> None:
    if os.environ.get("LLG_VLLM", "0") != "1":
        return
    targets = [t.strip() for t in os.environ.get("LLG_VLLM_MODELS", "qwen4_exp").split(",") if t.strip()]
    for t in targets:
        hook = _HOOKS.get(t)
        if hook is None:
            logger.warning("lowlat_gemm: unknown LLG_VLLM_MODELS entry %r (known: %s)", t, ", ".join(_HOOKS))
            continue
        if hook():
            logger.info("lowlat_gemm: vLLM hook installed for %s", t)
