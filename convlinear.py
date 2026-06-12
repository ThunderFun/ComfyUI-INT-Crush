"""ConvLinear4bit/ConvLinear8bit — W4A16/W8A16 inference modules.

Drop-in nn.Linear replacements with:
  1. Group-wise Hadamard rotation on activations
  2. Symmetric per-row INT4/INT8 quantization
  3. On-the-fly dequantization in forward()

Weights: packed INT4 (uint8) or plain INT8 (int8) with per-row FP16 scales.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import _quant_utils as _qu
from ._quant_utils import INT4_SCALE_DIVISOR


class ConvLinear8bit(nn.Module):
    """W8A8 linear layer with Hadamard rotation + per-row quantization.

    Weights are stored as plain INT8 (int8) with per-row FP16 scales.
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
        """Forward pass: rotate -> dequant weights -> F.linear."""
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
        """Convert an nn.Linear to ConvLinear8bit with per-row quantization.

        Args:
            module: source nn.Linear (weights are read, not modified)
            rot_need: whether to apply Hadamard rotation to the weight matrix
            rot_size: Hadamard group size (power of 2)

        Returns:
            ConvLinear8bit with INT8 weights and per-row float16 scales
        """
        if rot_need and not _qu._is_power_of_two(rot_size):
            raise ValueError(f"rot_size must be a power of 2, got {rot_size}")

        if rot_need:
            W = module.weight.float()
            weight_matrix = _qu.rotate_activations(W, rot_size)
        else:
            weight_matrix = module.weight.float()

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
    """W4A4 linear layer with Hadamard rotation + per-row quantization.

    Weights are stored as packed INT4 (uint8) with per-row FP16 scales.
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
        self.in_features = weight.shape[-1] * 2  # packed: 2 values per byte
        self.out_features = weight.shape[-2]
        self.rot_need = rot_need
        self.rot_size = rot_size
        self._perm = perm  # PermuQuant permutation indices
        self.register_buffer("weight", weight)
        self.register_buffer("scale", scale)

        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: rotate -> permute -> dequant weights -> F.linear."""
        if self.rot_need:
            x_rot = _qu.rotate_activations(x, self.rot_size)
        else:
            x_rot = x

        # Apply PermuQuant permutation to activations if present
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
        """Convert an nn.Linear to ConvLinear4bit with per-row quantization.

        Args:
            module: source nn.Linear (weights are read, not modified)
            rot_need: whether to apply Hadamard rotation to the weight matrix
            rot_size: Hadamard group size (power of 2)
            perm: optional PermuQuant channel permutation indices applied to
                  in_features before quantization

        Returns:
            ConvLinear4bit with packed INT4 (uint8) weights and per-row float16 scales
        """
        if rot_need and not _qu._is_power_of_two(rot_size):
            raise ValueError(f"rot_size must be a power of 2, got {rot_size}")

        if rot_need:
            W = module.weight.float()
            weight_matrix = _qu.rotate_activations(W, rot_size)
        else:
            weight_matrix = module.weight.float()

        if perm is not None:
            weight_matrix = weight_matrix[:, perm]

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
