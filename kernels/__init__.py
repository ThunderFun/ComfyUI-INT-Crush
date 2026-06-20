"""Triton INT8 kernels for fused GEMM + dynamic activation quantization."""

from .triton_quantize import (
    dynamic_quantize_activation,
    fused_layernorm_quantize,
    fused_rmsnorm_quantize,
    fused_softmax_quantize,
)
from .triton_int8_gemm import (
    fused_int8_gemm_dequant,
    fused_quant_int8_gemm_dequant,
    int8_gemm_int8out,
)

__all__ = [
    "dynamic_quantize_activation",
    "fused_layernorm_quantize",
    "fused_rmsnorm_quantize",
    "fused_softmax_quantize",
    "fused_int8_gemm_dequant",
    "fused_quant_int8_gemm_dequant",
    "int8_gemm_int8out",
]
