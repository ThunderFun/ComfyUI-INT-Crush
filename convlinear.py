"""ConvLinear4bit/ConvLinear8bit — standalone quantized linear modules.

Drop-in nn.Linear replacements that perform:
  1. Group-wise Hadamard rotation on activations (optional)
  2. Per-row symmetric INT4/INT8 quantization of weights
  3. On-the-fly dequantization in forward()

INT4 weights are packed 2-per-byte (uint8); INT8 weights are plain int8.
Both use per-row FP16 scales for dequantization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import _quant_utils as _qu
from ._quant_utils import INT4_SCALE_DIVISOR

__all__ = ["ConvLinear4bit", "ConvLinear8bit"]


def _prepare_rotated_weight(
    module: torch.nn.Linear,
    rot_need: bool,
    rot_size: int,
    perm: torch.Tensor | None = None,
) -> torch.Tensor:
    """Get the float weight matrix, optionally rotated and permuted.

    1. If *rot_need*, apply Hadamard rotation (rot_size is validated as power-of-2).
    2. If *perm* is provided, apply PermuQuant channel permutation (column-select).
    Returns a 2-D float tensor ready for scale/quantize.
    """
    if rot_need and not _qu.is_power_of_two(rot_size):
        raise ValueError(f"rot_size must be a power of 2, got {rot_size}")

    W = module.weight.float()
    if rot_need:
        W = _qu.rotate_activations(W, rot_size)
    if perm is not None:
        W = W[:, perm]
    return W


class ConvLinear8bit(nn.Module):
    """W8A16 linear layer with optional Hadamard rotation and per-row quantization.

    Weights are plain int8 with per-row FP16 scales. Dequantized to float16
    in forward() before the matrix multiply.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        rot_need: bool = True,
        bias: torch.Tensor | None = None,
        rot_size: int = 256,
    ):
        super().__init__()
        self.in_features = weight.shape[-1]
        self.out_features = weight.shape[-2]
        self.rot_need = rot_need
        self.rot_size = rot_size
        self.register_buffer("weight", weight)
        self.register_buffer("scale", scale)

        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: rotate activations → dequant weights → F.linear."""
        if self.rot_need:
            x_rot = _qu.rotate_activations(x, self.rot_size)
        else:
            x_rot = x

        weight_f16 = self.weight.to(x_rot.dtype) * self.scale.to(x_rot.dtype)
        bias = self.bias.to(x_rot.dtype) if self.bias is not None else None
        return F.linear(x_rot, weight_f16, bias)

    @staticmethod
    def from_float(
        module: nn.Linear,
        rot_need: bool = True,
        rot_size: int = 256,
    ) -> "ConvLinear8bit":
        """Convert an nn.Linear to ConvLinear8bit.

        Optionally rotates the weight matrix with a Hadamard transform
        before quantization. Per-row scales are computed from the rotated
        (or original) weight's max absolute value.
        """
        weight_matrix = _prepare_rotated_weight(module, rot_need, rot_size)

        in_features = weight_matrix.shape[1]
        weight_scales = _qu.calculate_scales_int8(weight_matrix, in_features)
        int8_weights = _qu.quantize_weights_int8(weight_matrix, weight_scales, in_features)
        bias = module.bias.data.clone() if module.bias is not None else None

        return ConvLinear8bit(
            weight=int8_weights,
            scale=weight_scales,
            rot_need=rot_need,
            bias=bias,
            rot_size=rot_size,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"rot_size={self.rot_size}, "
            f"rot_need={self.rot_need}"
        )


class ConvLinear4bit(nn.Module):
    """W4A16 linear layer with optional Hadamard rotation and per-row quantization.

    Weights are packed INT4 (2 values per uint8 byte) with per-row FP16 scales.
    Dequantized to float16 in forward() before the matrix multiply.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        rot_need: bool = True,
        bias: torch.Tensor | None = None,
        rot_size: int = 256,
        perm: torch.Tensor | None = None,
    ):
        super().__init__()
        self.in_features = weight.shape[-1] * 2  # 2 INT4 values packed per byte
        self.out_features = weight.shape[-2]
        self.rot_need = rot_need
        self.rot_size = rot_size
        self._perm = perm  # optional PermuQuant channel permutation indices
        self.register_buffer("weight", weight)
        self.register_buffer("scale", scale)

        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: rotate → permute → dequant weights → F.linear."""
        if self.rot_need:
            x_rot = _qu.rotate_activations(x, self.rot_size)
        else:
            x_rot = x

        # Apply PermuQuant channel permutation if present.
        if self._perm is not None:
            x_rot = x_rot[..., self._perm.to(x_rot.device)]

        unpacked = _qu.unpack_int4(self.weight, self.in_features)  # [out, in] int8
        weight_f16 = unpacked.to(x_rot.dtype) * self.scale.to(x_rot.dtype)
        bias = self.bias.to(x_rot.dtype) if self.bias is not None else None
        return F.linear(x_rot, weight_f16, bias)

    @staticmethod
    def from_float(
        module: nn.Linear,
        rot_need: bool = True,
        rot_size: int = 256,
        perm: torch.Tensor | None = None,
    ) -> "ConvLinear4bit":
        """Convert an nn.Linear to ConvLinear4bit.

        Optionally rotates the weight matrix with a Hadamard transform and
        applies PermuQuant channel permutation before quantization. Per-row
        scales are computed from the max absolute value of each output row.
        """
        weight_matrix = _prepare_rotated_weight(module, rot_need, rot_size, perm)

        in_features = weight_matrix.shape[1]
        weight_scales = _qu.calculate_scales(weight_matrix, in_features)
        int_rounded = _qu.quantize_weights(weight_matrix, weight_scales, in_features)
        packed_weight = _qu.pack_int4(int_rounded).to(torch.uint8)
        bias = module.bias.data.clone() if module.bias is not None else None

        return ConvLinear4bit(
            weight=packed_weight,
            scale=weight_scales,
            rot_need=rot_need,
            bias=bias,
            rot_size=rot_size,
            perm=perm,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"rot_size={self.rot_size}, "
            f"rot_need={self.rot_need}"
        )
