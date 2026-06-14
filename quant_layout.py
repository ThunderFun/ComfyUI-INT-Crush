"""QuantizedTensor layout classes for INT-Crush INT4/INT8 formats.

These bridge INT-Crush's packed weight format with ComfyUI's QuantizedTensor
dispatch system, enabling VBAR demand-paging, async offload, and LoRA
compatibility through ComfyUI's standard cast_bias_weight pipeline.

Each layout class defines:
  - Params dataclass: metadata stored alongside packed weights
  - state_dict_tensors(): serialization layout for safetensors
  - dequantize(): full dequantization (used for LoRA patching and fallback)
  - quantize(): float → packed (used by QuantizedTensor.from_float() for LoRA requantization)
  - supports_fast_matmul(): check for fused Triton kernel availability
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass

from ._quant_utils import unpack_int4, INT4_SCALE_DIVISOR

# BaseLayoutParams is required by QuantizedTensor.__tensor_flatten__ which
# calls dataclasses.fields(self._params) to discover tensor sub-fields.
try:
    from comfy_kitchen.tensor.base import BaseLayoutParams
except ImportError:
    # Minimal fallback when comfy_kitchen is unavailable (e.g. unit tests).
    # The real BaseLayoutParams is a @dataclass(frozen=True) with three fields:
    #   scale: torch.Tensor
    #   orig_dtype: torch.dtype
    #   orig_shape: tuple[int, ...]
    @dataclass(frozen=True)
    class BaseLayoutParams:
        scale: torch.Tensor
        orig_dtype: torch.dtype
        orig_shape: tuple

        def __post_init__(self):
            pass


class IntCrushInt4Layout:
    """Layout class for INT-Crush packed INT4 (uint8, 2 values per byte).

    Stores weights as packed uint8 with per-row FP16 scales. Optionally
    supports zero-point (asymmetric quantization) and PermuQuant channel
    permutation indices.
    """

    @dataclass(frozen=True)
    class Params(BaseLayoutParams):
        rot_need: bool = True
        rot_size: int = 256
        perm: torch.Tensor | None = None
        zp: torch.Tensor | None = None

        def _tensor_fields(self) -> list[str]:
            fields = ["scale"]
            if self.perm is not None:
                fields.append("perm")
            if self.zp is not None:
                fields.append("zp")
            return fields

        def _validate_tensor_fields(self):
            if isinstance(self.scale, torch.Tensor):
                object.__setattr__(self, "scale", self.scale.to(dtype=torch.float32, non_blocking=True))

    @classmethod
    def state_dict_tensors(cls, qdata, params):
        tensors = {"": qdata, "weight_scale": params.scale}
        if params.perm is not None:
            tensors["weight.perm"] = params.perm
        if params.zp is not None:
            tensors["weight_zp"] = params.zp
        return tensors

    @classmethod
    def dequantize(cls, qdata, params):
        """Full dequantization to float — used for LoRA patching and PyTorch fallback."""
        in_features = params.orig_shape[1]
        unpacked = unpack_int4(qdata, in_features)
        if params.zp is not None:
            return (unpacked.float() - params.zp.reshape(-1, 1).float()) * params.scale.reshape(-1, 1).float()
        return unpacked.float() * params.scale.reshape(-1, 1).float()

    @classmethod
    def supports_fast_matmul(cls):
        try:
            from .kernels.triton_int8_gemm import fused_int8_gemm_dequant
            from .kernels.triton_quantize import dynamic_quantize_activation
            from .kernels.triton_int4_to_int8_unpack import unpack_int4_to_int8
            return True
        except Exception:
            return False

    @classmethod
    def quantize(cls, tensor, scale=None, stochastic_rounding=0, inplace_ops=False):
        """Quantize a float tensor to packed INT4 with per-row scales.

        Used by QuantizedTensor.from_float() during LoRA requantization.
        The input tensor must already be in rotated+permuted space.
        """
        max_vals = tensor.abs().amax(dim=1, keepdim=True)
        if scale is None or (isinstance(scale, str) and scale == "recalculate"):
            scales = (max_vals / INT4_SCALE_DIVISOR).clamp(min=1e-8).to(torch.float16)
        elif isinstance(scale, torch.Tensor):
            scales = scale.to(torch.float16)
        else:
            scales = torch.tensor([scale], dtype=torch.float16, device=tensor.device)

        w_scaled = tensor / scales.to(tensor.dtype)
        int_rounded = w_scaled.round().clamp(-8, 7).to(torch.int8)
        # Pack two INT4 values per byte (low nibble = even index).
        K = int_rounded.shape[-1]
        if K % 2 != 0:
            pad = torch.zeros(*int_rounded.shape[:-1], 1, dtype=torch.int8, device=int_rounded.device)
            int_rounded = torch.cat([int_rounded, pad], dim=-1)
        q_i8 = torch.where(int_rounded < 0, 16 + int_rounded, int_rounded).to(torch.uint8)
        packed = q_i8[..., 0::2] | (q_i8[..., 1::2] << 4)

        orig_dtype = tensor.dtype
        orig_shape = tuple(tensor.shape)
        params = cls.Params(
            scale=scales.reshape(-1), orig_dtype=orig_dtype, orig_shape=orig_shape,
        )
        return packed, params


class IntCrushInt8Layout:
    """Layout class for INT-Crush INT8 (plain int8) with per-row FP16 scales."""

    @dataclass(frozen=True)
    class Params(BaseLayoutParams):
        rot_need: bool = True
        rot_size: int = 256
        perm: torch.Tensor | None = None

        def _tensor_fields(self) -> list[str]:
            fields = ["scale"]
            if self.perm is not None:
                fields.append("perm")
            return fields

        def _validate_tensor_fields(self):
            if isinstance(self.scale, torch.Tensor):
                object.__setattr__(self, "scale", self.scale.to(dtype=torch.float32, non_blocking=True))

    @classmethod
    def state_dict_tensors(cls, qdata, params):
        tensors = {"": qdata, "weight_scale": params.scale}
        if params.perm is not None:
            tensors["weight.perm"] = params.perm
        return tensors

    @classmethod
    def dequantize(cls, qdata, params):
        """Full dequantization to float — used for LoRA patching and PyTorch fallback."""
        return qdata.float() * params.scale.reshape(-1, 1).float()

    @classmethod
    def supports_fast_matmul(cls):
        try:
            from .kernels.triton_int8_gemm import fused_int8_gemm_dequant
            from .kernels.triton_quantize import dynamic_quantize_activation
            return True
        except Exception:
            return False

    @classmethod
    def quantize(cls, tensor, scale=None, stochastic_rounding=0, inplace_ops=False):
        """Quantize a float tensor to INT8 with per-row scales.

        Used by QuantizedTensor.from_float() during LoRA requantization.
        The input tensor must already be in rotated space.
        """
        max_vals = tensor.abs().amax(dim=1, keepdim=True)
        if scale is None or (isinstance(scale, str) and scale == "recalculate"):
            scales = (max_vals / 127.0).clamp(min=1e-8).to(torch.float32)
        elif isinstance(scale, torch.Tensor):
            scales = scale.to(torch.float32)
        else:
            scales = torch.tensor([scale], dtype=torch.float32, device=tensor.device)

        w_scaled = tensor / scales.to(tensor.dtype)
        int_rounded = w_scaled.round().clamp(-128, 127).to(torch.int8)

        orig_dtype = tensor.dtype
        orig_shape = tuple(tensor.shape)
        params = cls.Params(
            scale=scales.reshape(-1), orig_dtype=orig_dtype, orig_shape=orig_shape,
        )
        return int_rounded, params


def register_intcrush_layouts():
    """Register INT-Crush INT4/INT8 formats with ComfyUI's quantization system.

    Idempotent and safe to call even when comfy_kitchen is unavailable (no-op).
    """
    try:
        from comfy.quant_ops import register_layout_class, QUANT_ALGOS

        register_layout_class("IntCrushInt4Layout", IntCrushInt4Layout)
        register_layout_class("IntCrushInt8Layout", IntCrushInt8Layout)

        QUANT_ALGOS.setdefault("int4_crush", {
            "storage_t": torch.uint8,
            "parameters": {"weight_scale", "weight_zp", "weight.perm"},
            "comfy_tensor_layout": "IntCrushInt4Layout",
        })
        QUANT_ALGOS.setdefault("int8_crush", {
            "storage_t": torch.int8,
            "parameters": {"weight_scale", "weight.perm"},
            "comfy_tensor_layout": "IntCrushInt8Layout",
        })
    except ImportError:
        pass
