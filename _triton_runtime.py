"""Triton kernel availability detection for INT-Crush.

Probes for each Triton kernel at import time and exposes boolean flags
and (optionally) the kernel callables.  Also shrinks the Triton autotune
cache to avoid VRAM bloat from benchmark allocations.
"""

import logging

import torch

log = logging.getLogger(__name__)

# ── Kernel availability flags ────────────────────────────────────────────────

TRITON_AVAILABLE = False
TRITON_INT4_UNPACK = False
TRITON_INT8_GEMM = False
TRITON_INT4_INT8_UNPACK = False
TRITON_DYNQUANT = False
TRITON_W4A8_GEMM = False
TRITON_FUSED_QGD = False

# ── User toggles ─────────────────────────────────────────────────────────────
# Set USE_FUSED_W8A8 = True to use the fused quantize+GEMM kernel for W8A8
# inference instead of the default two-kernel path (quantize -> GEMM).
USE_FUSED_W8A8: bool = False

# Set USE_CUDA_GRAPHS = True to enable CUDA Graph capture for the INT8
# two-kernel path. Eliminates launch overhead.
# Requires PyTorch >= 2.0 and a CUDA-capable GPU.
USE_CUDA_GRAPHS: bool = True

# ── Kernel callables (None when unavailable) ────────────────────────────────

dynamic_quantize_activation = None
fused_int8_gemm_dequant = None
fused_quant_int8_gemm_dequant = None
fused_w4a8_gemm_dequant = None
unpack_int4_to_int8 = None
unpack_int4_to_float16 = None

# ── Probe each kernel ───────────────────────────────────────────────────────
# Each entry: (module_relative_path, callable_name, flag_name)
# All probes are independent — one failing kernel does not block others.

_PROBES = [
    ("kernels.triton_int4_unpack",   "unpack_int4_to_float16",    "TRITON_INT4_UNPACK"),
    ("kernels.triton_quantize",      "dynamic_quantize_activation","TRITON_DYNQUANT"),
    ("kernels.triton_w4a8_gemm",     "fused_w4a8_gemm_dequant",   "TRITON_W4A8_GEMM"),
    ("kernels.triton_int8_gemm",     "fused_int8_gemm_dequant",   "TRITON_INT8_GEMM"),
    ("kernels.triton_int8_gemm",     "fused_quant_int8_gemm_dequant", "TRITON_FUSED_QGD"),
    ("kernels.triton_int4_to_int8_unpack", "unpack_int4_to_int8", "TRITON_INT4_INT8_UNPACK"),
]

import importlib as _importlib
import os as _os
import sys as _sys

if _os.environ.get("INTCRUSH_NO_CUDA_GRAPHS", "0") == "1":
    USE_CUDA_GRAPHS = False

for _mod_path, _attr, _flag in _PROBES:
    try:
        _mod = _importlib.import_module("." + _mod_path, package=__package__)
        globals()[_attr] = getattr(_mod, _attr)
        globals()[_flag] = True
        log.info("[INT-Crush] Triton %s kernel loaded", _attr)
    except Exception as _exc:
        log.debug("[INT-Crush] Triton %s not available: %s", _attr, _exc)

# ── Derived flags ────────────────────────────────────────────────────────────

if TRITON_W4A8_GEMM and TRITON_DYNQUANT and TRITON_INT4_INT8_UNPACK:
    log.info("[INT-Crush] Triton W4A8 GEMM + dynamic quantizer + INT4 unpack loaded")
if TRITON_INT8_GEMM and TRITON_DYNQUANT:
    TRITON_AVAILABLE = True
    if not TRITON_W4A8_GEMM:
        log.info("[INT-Crush] Triton INT8 GEMM + dynamic quantizer loaded (W4A8 will use W8A8 kernel)")
if TRITON_INT4_INT8_UNPACK:
    log.info("[INT-Crush] Triton INT4->INT8 unpack kernel loaded")

# ── Shrink Triton autotune cache ────────────────────────────────────────────
#
# The default benchmark cache allocates a large temporary buffer on each
# autotune run, which can cause VRAM bloat.  Monkey-patch it to use at
# most 32 MiB (or 1/8 of free memory, whichever is smaller).
# Below 4 MiB the cache is disabled entirely (insufficient space for autotuning).

_AUTOTUNE_CACHE_BUDGET_BYTES = 32 * 1024 * 1024   # 32 MiB hard cap
_AUTOTUNE_CACHE_VRAM_FRACTION = 8                  # use at most 1/8 of free VRAM
_AUTOTUNE_CACHE_MIN_BYTES = 4 * 1024 * 1024        # 4 MiB floor — disable below this

try:
    import triton.backends.nvidia.driver as _triton_nvidia

    _orig_get_cache = _triton_nvidia.CUDADriver.get_empty_cache_for_benchmark

    def _get_cache_small(self):
        try:
            free_mem, _ = torch.cuda.mem_get_info()
            budget = min(_AUTOTUNE_CACHE_BUDGET_BYTES, free_mem // _AUTOTUNE_CACHE_VRAM_FRACTION)
        except Exception:
            budget = _AUTOTUNE_CACHE_BUDGET_BYTES
        if budget < _AUTOTUNE_CACHE_MIN_BYTES:
            return torch.empty(0, dtype=torch.int, device="cuda")
        return torch.empty(budget // 4, dtype=torch.int, device="cuda")

    _triton_nvidia.CUDADriver.get_empty_cache_for_benchmark = _get_cache_small
except Exception:
    pass


__all__ = [
    "TRITON_AVAILABLE",
    "TRITON_INT4_UNPACK",
    "TRITON_INT8_GEMM",
    "TRITON_INT4_INT8_UNPACK",
    "TRITON_DYNQUANT",
    "TRITON_W4A8_GEMM",
    "TRITON_FUSED_QGD",
    "USE_FUSED_W8A8",
    "USE_CUDA_GRAPHS",
    "dynamic_quantize_activation",
    "fused_int8_gemm_dequant",
    "fused_quant_int8_gemm_dequant",
    "fused_w4a8_gemm_dequant",
    "unpack_int4_to_int8",
    "unpack_int4_to_float16",
]
