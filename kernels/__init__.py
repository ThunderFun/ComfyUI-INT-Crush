"""Triton INT8 kernels for fused GEMM + dynamic activation quantization."""

from .triton_quantize import dynamic_quantize_activation
from .triton_int8_gemm import fused_int8_gemm_dequant

__all__ = ["dynamic_quantize_activation", "fused_int8_gemm_dequant"]
