"""Quantization utilities for INT-Crush INT4/INT8 inference.

Hadamard rotation, INT4/INT8 pack/unpack, per-group scale calculation,
weight quantization, and activation scale calculation.
"""

import math
import torch
import torch.nn.functional as F


def _is_power_of_two(n: int) -> bool:
    """Return True if n is a positive power of 2 (1, 2, 4, 8, ...)."""
    return n > 0 and (n & (n - 1)) == 0


def _is_power_of_four(n: int) -> bool:
    """Return True if n is a positive power of 4 (4, 16, 64, 256, ...).

    Used to decide between Regular Hadamard (balanced row sums) and
    Sylvester construction for the rotation matrix.
    """
    if n < 4:
        return False
    return (n & (n - 1)) == 0 and (n & 0x55555555) == n


_H_cache: dict[tuple, torch.Tensor] = {}


def make_hadamard_regular(n: int, dtype: torch.dtype = torch.float16, device: str = "cpu") -> torch.Tensor:
    """Normalized Regular Hadamard matrix of size n (power of 4).

    Balanced row sums prevent outlier aggregation during rotation (ConvRot, arXiv:2512.03673).
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
    """Construct a normalized Sylvester Hadamard matrix of size n (any power of 2)."""
    if not _is_power_of_two(n):
        raise ValueError(f"n must be a power of 2, got {n}")
    H = torch.tensor([[1.0]], dtype=dtype, device=device)
    while H.shape[0] < n:
        H = torch.kron(
            torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=dtype, device=device),
            H,
        )
    return H * (1.0 / math.sqrt(n))


def rotate_activations(x: torch.Tensor, rot_size: int, H: torch.Tensor | None = None) -> torch.Tensor:
    """Group-wise Hadamard rotation on activations.

    Uses Regular Hadamard for powers of 4, Sylvester fallback otherwise.
    """
    if not _is_power_of_two(rot_size):
        raise ValueError(f"rot_size must be a power of 2, got {rot_size}")

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
    x_rot = torch.matmul(x_flat, H.T)

    return x_rot.reshape(*leading_shape, in_features)


INT4_MIN = -8
INT4_MAX = 7
INT4_SCALE_DIVISOR = 7.0
DEFAULT_GROUP_SIZE = 128


def validate_int4_range(tensor: torch.Tensor) -> None:
    if tensor.numel() == 0:
        return
    if tensor.min() < INT4_MIN or tensor.max() > INT4_MAX:
        raise ValueError(
            f"Values must be in [{INT4_MIN}, {INT4_MAX}], "
            f"got [{tensor.min().item()}, {tensor.max().item()}]"
        )


def pack_int4(values: torch.Tensor) -> torch.Tensor:
    """Pack INT4 values: two's complement, low nibble = even index, uint8."""
    validate_int4_range(values)
    values = values.to(torch.int8)
    K = values.shape[-1]
    if K % 2 != 0:
        pad = torch.zeros(*values.shape[:-1], 1, dtype=values.dtype, device=values.device)
        values = torch.cat([values, pad], dim=-1)
    q_i8 = torch.where(values < 0, 2 ** 4 + values, values).to(torch.uint8)
    return q_i8[..., 0::2] | (q_i8[..., 1::2] << 4)


def unpack_int4(packed: torch.Tensor, K: int) -> torch.Tensor:
    """Unpack packed INT4 back to int8 values."""
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    result = torch.zeros(*packed.shape[:-1], packed.shape[-1] * 2, dtype=torch.int8, device=packed.device)
    result[..., 0::2] = low
    result[..., 1::2] = high
    return result[..., :K]


INT8_MIN = -128
INT8_MAX = 127
INT8_SCALE_DIVISOR = 127.0


def pack_int8(values: torch.Tensor) -> torch.Tensor:
    """INT8 is stored unpacked; just validate and cast."""
    if values.numel() > 0 and (values.min() < INT8_MIN or values.max() > INT8_MAX):
        raise ValueError(
            f"Values must be in [{INT8_MIN}, {INT8_MAX}], "
            f"got [{values.min().item()}, {values.max().item()}]"
        )
    return values.to(torch.int8)


def unpack_int8(packed: torch.Tensor, K: int) -> torch.Tensor:
    """INT8 unpack is a no-op."""
    return packed[..., :K]


def calculate_scales(W: torch.Tensor, group_size: int = DEFAULT_GROUP_SIZE) -> torch.Tensor:
    """Per-group scales for INT4 quantization (per-row when group_size == in_features).

    Args:
        W: [out_features, in_features] weight tensor
        group_size: number of channels per group

    Returns:
        scales: [out_features, num_groups] float16 per-group scales
    """
    if W.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {W.dim()}D")

    out_features, in_features = W.shape
    if in_features % group_size != 0:
        pad = group_size - (in_features % group_size)
        W = F.pad(W, (0, pad))
        in_features = W.shape[1]

    num_groups = in_features // group_size
    W_grouped = W.reshape(out_features, num_groups, group_size)
    max_vals = W_grouped.abs().amax(dim=2)
    scales = (max_vals / INT4_SCALE_DIVISOR).clamp(min=1e-8).to(torch.float16)
    return scales


def calculate_scales_int8(W: torch.Tensor, group_size: int = DEFAULT_GROUP_SIZE) -> torch.Tensor:
    """Per-group scales for INT8 quantization (per-row when group_size == in_features).

    Args:
        W: [out_features, in_features] weight tensor
        group_size: number of channels per group

    Returns:
        scales: [out_features, num_groups] float16 per-group scales
    """
    if W.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {W.dim()}D")

    out_features, in_features = W.shape
    if in_features % group_size != 0:
        pad = group_size - (in_features % group_size)
        W = F.pad(W, (0, pad))
        in_features = W.shape[1]

    num_groups = in_features // group_size
    W_grouped = W.reshape(out_features, num_groups, group_size)
    max_vals = W_grouped.abs().amax(dim=2)
    scales = (max_vals / INT8_SCALE_DIVISOR).clamp(min=1e-8).to(torch.float16)
    return scales


def quantize_weights(W: torch.Tensor, scales: torch.Tensor, group_size: int = DEFAULT_GROUP_SIZE) -> torch.Tensor:
    """Quantize weights to INT4 using precomputed per-group scales (per-row when group_size == in_features).

    Args:
        W: [out_features, in_features] rotated weight tensor
        scales: [out_features, num_groups] per-group scales
        group_size: number of channels per group

    Returns:
        quantized: [out_features, in_features] INT4 values in [-8, 7]
    """
    out_features, in_features = W.shape
    if in_features % group_size != 0:
        pad = group_size - (in_features % group_size)
        W = F.pad(W, (0, pad))
        in_features = W.shape[1]

    num_groups = in_features // group_size
    W_grouped = W.reshape(out_features, num_groups, group_size)
    W_scaled = W_grouped / scales.unsqueeze(2).to(W.dtype)
    W_rounded = W_scaled.round().clamp(INT4_MIN, INT4_MAX)
    return W_rounded.reshape(out_features, in_features).to(torch.int8)


def quantize_weights_int8(W: torch.Tensor, scales: torch.Tensor, group_size: int = DEFAULT_GROUP_SIZE) -> torch.Tensor:
    """Quantize weights to INT8 using precomputed per-group scales (per-row when group_size == in_features).

    Args:
        W: [out_features, in_features] rotated weight tensor
        scales: [out_features, num_groups] per-group scales
        group_size: number of channels per group

    Returns:
        quantized: [out_features, in_features] INT8 values in [-128, 127]
    """
    out_features, in_features = W.shape
    if in_features % group_size != 0:
        pad = group_size - (in_features % group_size)
        W = F.pad(W, (0, pad))
        in_features = W.shape[1]

    num_groups = in_features // group_size
    W_grouped = W.reshape(out_features, num_groups, group_size)
    W_scaled = W_grouped / scales.unsqueeze(2).to(W.dtype)
    W_rounded = W_scaled.round().clamp(INT8_MIN, INT8_MAX)
    return W_rounded.reshape(out_features, in_features).to(torch.int8)


def calculate_activation_scales(x: torch.Tensor, group_size: int = DEFAULT_GROUP_SIZE) -> torch.Tensor:
    """Per-group scales for activation quantization (per-token when group_size == K).

    Args:
        x: [..., K] activation tensor (float16), arbitrary leading dims
        group_size: number of channels per group

    Returns:
        scales: [..., num_groups] float16 per-group scales
    """
    orig_shape = x.shape
    K = orig_shape[-1]
    x_flat = x.reshape(-1, K)

    if K % group_size != 0:
        pad = group_size - (K % group_size)
        x_flat = F.pad(x_flat, (0, pad))
        K = x_flat.shape[1]

    num_groups = K // group_size
    x_grouped = x_flat.reshape(x_flat.shape[0], num_groups, group_size)
    max_vals = x_grouped.abs().amax(dim=2)
    scales = (max_vals / INT4_SCALE_DIVISOR).clamp(min=1e-8).to(torch.float16)

    # Reshape to match leading dims: [..., num_groups]
    return scales.reshape(*orig_shape[:-1], num_groups)
