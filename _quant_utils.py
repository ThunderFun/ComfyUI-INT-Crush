"""Quantization primitives for INT-Crush INT4/INT8 inference.

Provides:
  - Hadamard rotation (Regular for powers-of-4, Sylvester fallback)
  - INT4 pack/unpack (two's complement, 2 values per uint8 byte)
  - INT8 pack/unpack (identity; stored as plain int8)
  - Per-group scale calculation for symmetric quantization
  - Weight quantization (INT4 and INT8)
  - Per-token activation scale calculation
"""

import math
import torch
import torch.nn.functional as F

__all__ = [
    "make_hadamard_regular",
    "is_power_of_two",
    "rotate_activations",
    "pack_int4",
    "unpack_int4",
    "pack_int8",
    "unpack_int8",
    "validate_int4_range",
    "calculate_scales",
    "calculate_scales_int8",
    "quantize_weights",
    "quantize_weights_int8",
    "INT4_MIN",
    "INT4_MAX",
    "INT4_SCALE_DIVISOR",
    "INT4_TWOS_COMPLEMENT_BIAS",
    "INT8_MIN",
    "INT8_MAX",
    "INT8_SCALE_DIVISOR",
    "DEFAULT_GROUP_SIZE",
]


def _is_power_of_two(n: int) -> bool:
    """Return True if n is a positive power of 2 (1, 2, 4, 8, ...)."""
    return n > 0 and (n & (n - 1)) == 0


# Public alias — used by convlinear.py to validate rot_size.
is_power_of_two = _is_power_of_two


def _is_power_of_four(n: int) -> bool:
    """Return True if n is a positive power of 4 (4, 16, 64, 256, ...).

    Used to choose between Regular Hadamard (balanced row sums, preferred
    for rotation) and Sylvester construction (any power of 2, fallback).
    See ConvRot (arXiv:2512.03673) for why balanced row sums matter.
    """
    if n < 4:
        return False
    return (n & (n - 1)) == 0 and (n & 0x55555555) == n


# Cache for precomputed Hadamard matrices.  Keyed by (n, dtype-str, device, [variant]).
# Intentionally unbounded — the set of distinct (n, dtype, device) tuples is tiny in practice
# (typical models use rot_size 16–4096 with float16 on a single GPU).
_H_cache: dict[tuple, torch.Tensor] = {}


def make_hadamard_regular(n: int, dtype: torch.dtype = torch.float16, device: str = "cpu") -> torch.Tensor:
    """Normalized Regular Hadamard matrix of size n (must be a power of 4).

    Constructed via Kronecker products of the 4x4 base. Balanced row sums
    (each row sums to ±1) prevent outlier aggregation during rotation
    (ConvRot, arXiv:2512.03673).
    """
    if not _is_power_of_four(n):
        raise ValueError(f"Regular Hadamard requires power of 4, got {n}. Use rot_size=16/64/256/1024/4096.")

    key = (n, str(dtype), device)
    if key in _H_cache:
        return _H_cache[key]

    H4 = torch.tensor([
        [ 1.0,  1.0,  1.0, -1.0],
        [ 1.0,  1.0, -1.0,  1.0],
        [ 1.0, -1.0,  1.0,  1.0],
        [-1.0,  1.0,  1.0,  1.0],
    ], dtype=dtype, device=device) / 2.0

    H = H4
    while H.shape[0] < n:
        H = torch.kron(H, H4)

    _H_cache[key] = H
    return H


def _make_hadamard_sylvester(n: int, dtype: torch.dtype = torch.float16, device: str = "cpu") -> torch.Tensor:
    """Normalized Sylvester Hadamard matrix of size n (any power of 2).

    Fallback for sizes that are not powers of 4 (e.g. 32, 128, 512).
    """
    key = (n, str(dtype), device, "sylvester")
    if key in _H_cache:
        return _H_cache[key]
    if not _is_power_of_two(n):
        raise ValueError(f"n must be a power of 2, got {n}")
    H = torch.tensor([[1.0]], dtype=dtype, device=device)
    while H.shape[0] < n:
        H = torch.kron(
            torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=dtype, device=device),
            H,
        )
    H = H * (1.0 / math.sqrt(n))
    _H_cache[key] = H
    return H


def rotate_activations(x: torch.Tensor, rot_size: int, H: torch.Tensor | None = None) -> torch.Tensor:
    """Apply group-wise Hadamard rotation to the last dimension of x.

    Pads the feature dimension to a multiple of rot_size if needed, then
    multiplies each group by the Hadamard matrix. Prefers Regular Hadamard
    (powers of 4) with Sylvester fallback for other power-of-2 sizes.
    """
    if not _is_power_of_two(rot_size):
        raise ValueError(f"rot_size must be power of 2, got {rot_size}")

    orig_features = x.shape[-1]
    if orig_features % rot_size != 0:
        pad = rot_size - (orig_features % rot_size)
        x = F.pad(x, (0, pad))

    in_features = x.shape[-1]
    num_groups = in_features // rot_size
    leading_shape = x.shape[:-1]
    x_flat = x.reshape(-1, num_groups, rot_size)

    if H is None:
        if _is_power_of_four(rot_size):
            H = make_hadamard_regular(rot_size, dtype=x.dtype, device=str(x.device))
        else:
            H = _make_hadamard_sylvester(rot_size, dtype=x.dtype, device=str(x.device))
    if H.device != x.device or H.dtype != x.dtype:
        H = H.to(dtype=x.dtype, device=x.device)
        _H_cache[(rot_size, str(x.dtype), str(x.device))] = H
    x_rot = torch.matmul(x_flat, H.T)

    return x_rot.reshape(*leading_shape, in_features)


# ── INT4 constants ────────────────────────────────────────────────────────────

INT4_MIN = -8
INT4_MAX = 7
INT4_SCALE_DIVISOR = 7.0  # INT4_MAX — symmetric-quant divisor; scale = max_abs / 7, q = round(W/scale) clamped to [-8, 7]
INT4_TWOS_COMPLEMENT_BIAS = 1 << 4  # 16 — two's complement bias for 4-bit signed-to-unsigned encoding
DEFAULT_GROUP_SIZE = 128


def validate_int4_range(tensor: torch.Tensor) -> None:
    """Raise ValueError if any element falls outside the INT4 range [-8, 7]."""
    if tensor.numel() == 0:
        return
    if tensor.min() < INT4_MIN or tensor.max() > INT4_MAX:
        raise ValueError(
            f"Values must be in [{INT4_MIN}, {INT4_MAX}], "
            f"got [{tensor.min().item()}, {tensor.max().item()}]"
        )


def pack_int4(values: torch.Tensor) -> torch.Tensor:
    """Pack INT4 values into uint8: low nibble = even index, high nibble = odd.

    Two's complement encoding; odd-length tensors are zero-padded before packing.
    """
    validate_int4_range(values)
    values = values.to(torch.int8)
    K = values.shape[-1]
    if K % 2 != 0:
        pad = torch.zeros(*values.shape[:-1], 1, dtype=values.dtype, device=values.device)
        values = torch.cat([values, pad], dim=-1)
    q_i8 = torch.where(values < 0, INT4_TWOS_COMPLEMENT_BIAS + values, values).to(torch.uint8)
    return q_i8[..., 0::2] | (q_i8[..., 1::2] << 4)


def unpack_int4(packed: torch.Tensor, K: int) -> torch.Tensor:
    """Unpack uint8-packed INT4 back to int8 values.

    K is the original (unpacked) length; extra padding bytes are trimmed.
    """
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    result = torch.zeros(*packed.shape[:-1], packed.shape[-1] * 2, dtype=torch.int8, device=packed.device)
    result[..., 0::2] = low
    result[..., 1::2] = high
    return result[..., :K]


# ── INT8 constants ────────────────────────────────────────────────────────────

INT8_MIN = -128
INT8_MAX = 127
INT8_SCALE_DIVISOR = 127.0  # INT8_MAX — symmetric-quant divisor; scale = max_abs / 127, q = round(W/scale) clamped to [-128, 127]


def pack_int8(values: torch.Tensor) -> torch.Tensor:
    """Validate INT8 range and cast to int8 (no packing needed)."""
    if values.numel() > 0 and (values.min() < INT8_MIN or values.max() > INT8_MAX):
        raise ValueError(
            f"Values must be in [{INT8_MIN}, {INT8_MAX}], "
            f"got [{values.min().item()}, {values.max().item()}]"
        )
    return values.to(torch.int8)


def unpack_int8(packed: torch.Tensor, K: int) -> torch.Tensor:
    """Slice INT8 tensor to original length (no unpacking needed)."""
    return packed[..., :K]


# ── Shared helpers ───────────────────────────────────────────────────────────

def _pad_to_group_multiple(W: torch.Tensor, group_size: int) -> tuple[torch.Tensor, int]:
    """Pad the feature (last) dimension of a 2-D weight tensor to a multiple of group_size.

    Returns (W_padded, in_features) where in_features is the padded size.
    """
    in_features = W.shape[1]
    if in_features % group_size != 0:
        pad = group_size - (in_features % group_size)
        W = F.pad(W, (0, pad))
        in_features = W.shape[1]
    return W, in_features


def _calculate_scales(W: torch.Tensor, scale_divisor: float,
                      group_size: int = DEFAULT_GROUP_SIZE) -> torch.Tensor:
    """Per-group symmetric scales: max-abs per group / scale_divisor.

    When group_size == in_features this reduces to per-row quantization.
    """
    if W.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {W.dim()}D")

    out_features = W.shape[0]
    W, in_features = _pad_to_group_multiple(W, group_size)

    num_groups = in_features // group_size
    W_grouped = W.reshape(out_features, num_groups, group_size)
    max_vals = W_grouped.abs().amax(dim=2)
    return (max_vals / scale_divisor).clamp(min=1e-8).to(torch.float16)


def _quantize_weights(W: torch.Tensor, scales: torch.Tensor,
                      clamp_min: int, clamp_max: int,
                      group_size: int = DEFAULT_GROUP_SIZE) -> torch.Tensor:
    """Quantize weights using precomputed per-group scales.

    Divides each group by its scale, rounds to nearest, clamps to [clamp_min, clamp_max].
    When group_size == in_features this is per-row quantization.
    """
    out_features = W.shape[0]
    W, in_features = _pad_to_group_multiple(W, group_size)

    num_groups = in_features // group_size
    W_grouped = W.reshape(out_features, num_groups, group_size)
    W_scaled = W_grouped / scales.unsqueeze(2).to(W.dtype)
    W_rounded = W_scaled.round().clamp(clamp_min, clamp_max)
    return W_rounded.reshape(out_features, in_features).to(torch.int8)


# ── Typed wrappers for backward compatibility ────────────────────────────────

def calculate_scales(W: torch.Tensor, group_size: int = DEFAULT_GROUP_SIZE) -> torch.Tensor:
    """Per-group symmetric scales for INT4 quantization (divisor=7.0)."""
    return _calculate_scales(W, INT4_SCALE_DIVISOR, group_size)


def calculate_scales_int8(W: torch.Tensor, group_size: int = DEFAULT_GROUP_SIZE) -> torch.Tensor:
    """Per-group symmetric scales for INT8 quantization (divisor=127.0)."""
    return _calculate_scales(W, INT8_SCALE_DIVISOR, group_size)


def quantize_weights(W: torch.Tensor, scales: torch.Tensor, group_size: int = DEFAULT_GROUP_SIZE) -> torch.Tensor:
    """Quantize weights to INT4 (clamp [-8, 7])."""
    return _quantize_weights(W, scales, INT4_MIN, INT4_MAX, group_size)


def quantize_weights_int8(W: torch.Tensor, scales: torch.Tensor, group_size: int = DEFAULT_GROUP_SIZE) -> torch.Tensor:
    """Quantize weights to INT8 (clamp [-128, 127])."""
    return _quantize_weights(W, scales, INT8_MIN, INT8_MAX, group_size)



