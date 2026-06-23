"""Shared test fixtures for INT-Crush kernel and quantization tests.

Provides reusable factories for int8 and int4 test data that are
duplicated across multiple test modules.
"""

import torch
from _intcrush import load

_qu = load("_quant_utils")
INT4_SCALE_DIVISOR = _qu.INT4_SCALE_DIVISOR


def make_int4_layer(N: int, K: int, device: str = "cpu"):
    """Create a random INT4 quantized layer (packed weight, scale, K).

    Returns (packed, scale_flat, K_orig) where packed is [N, K//2] uint8
    and scale_flat is [N] float16 per-row scales.
    """
    pack_int4 = _qu.pack_int4
    W = torch.randn(N, K, dtype=torch.float32, device=device)
    max_vals = W.abs().amax(dim=1)
    scales = (max_vals / INT4_SCALE_DIVISOR).clamp(min=1e-8).to(torch.float16)
    W_scaled = W / scales.unsqueeze(1).to(W.dtype)
    int_rounded = W_scaled.round().clamp(-8, 7).to(torch.int8)
    packed = pack_int4(int_rounded).to(torch.uint8)
    return packed, scales.reshape(-1), K


def make_int8_inputs(M: int, N: int, K: int, device: str = "cpu"):
    """Create random int8 activations, weights, and scales for GEMM.

    Scales are chosen to keep dequantized values within fp16 range.
    Typical real-world scales are 0.001–0.1 (not the [0, 1] range that
    would overflow fp16 for large K).

    Returns (x_int8, w_int8, s_a, s_w).
    """
    x_int8 = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=device)
    w_int8 = torch.randint(-128, 127, (N, K), dtype=torch.int8, device=device)
    # max |dequant| ≈ K * 127 * 127 * s_a * s_w; keep under ~30 000 for fp16.
    max_scale = min(0.1, 30_000.0 / (K * 127 * 127))
    s_a = (torch.rand(M, dtype=torch.float32, device=device) * max_scale + 1e-4)
    s_w = (torch.rand(N, dtype=torch.float32, device=device) * max_scale + 1e-4)
    return x_int8, w_int8, s_a, s_w


def make_int8_asymmetric_layer(N: int, K: int, device: str = "cpu"):
    """Create asymmetric INT8 quantized weights: W = (q - zp) * scale.

    Uses full-range asymmetric quantization where w_min maps to q=-128
    and w_max maps to q=127, with a per-row zero-point offset.

    Returns (q_int8, scale, zp) where:
      - q_int8 is [N, K] int8 quantized weights
      - scale  is [N] float32 per-row scales
      - zp     is [N] float32 per-row zero-points (may be outside [-128, 127])
    """
    W = torch.randn(N, K, dtype=torch.float32, device=device)
    w_min = W.amin(dim=1, keepdim=True)
    w_max = W.amax(dim=1, keepdim=True)
    # Asymmetric scale: range / 255 (full int8 range)
    scale = ((w_max - w_min) / 255.0).clamp(min=1e-8).to(torch.float32)
    # Zero-point: the quantized value that maps to float zero.
    # w_min -> q=-128 => zp = -128 - w_min/scale
    zp = (-128.0 - w_min / scale).round()
    q = (W / scale + zp).round().clamp(-128, 127).to(torch.int8)
    return q, scale.reshape(-1), zp.reshape(-1).to(torch.float32)
