"""
Dynamic per-token INT8 quantization — fused Triton implementation.

Step 1 of the inference pipeline:
  Input:  x_bf16 [M, K]
  Output: x_int8 [M, K], s_a [M]

Single fused Triton kernel that computes the per-token scale and quantizes
in two internal loops within the same block launch.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _compute_row_scales_kernel(
    x_ptr,
    s_ptr,
    stride_xm,
    K,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs_k = tl.arange(0, BLOCK_K)
    x_base = x_ptr + pid * stride_xm

    row_max = tl.zeros([], dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        cols = k * BLOCK_K + offs_k
        mask = cols < K
        vals = tl.load(x_base + cols, mask=mask, other=0.0).to(tl.float32)
        abs_vals = tl.abs(vals)
        block_max = tl.max(abs_vals, axis=0)
        row_max = tl.maximum(row_max, block_max)

    scale = tl.maximum(row_max, 1e-8) / 127.0
    tl.store(s_ptr + pid, scale)


@triton.jit
def _fused_quantize_kernel(
    x_ptr,
    s_ptr,
    out_ptr,
    stride_xm,
    stride_outm,
    K,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs_k = tl.arange(0, BLOCK_K)
    x_base = x_ptr + pid * stride_xm

    # Pass 1: compute row max
    row_max = tl.zeros([], dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        cols = k * BLOCK_K + offs_k
        mask = cols < K
        vals = tl.load(x_base + cols, mask=mask, other=0.0).to(tl.float32)
        abs_vals = tl.abs(vals)
        block_max = tl.max(abs_vals, axis=0)
        row_max = tl.maximum(row_max, block_max)

    scale = tl.maximum(row_max, 1e-8) / 127.0
    tl.store(s_ptr + pid, scale)

    # Pass 2: quantize using the computed scale
    out_base = out_ptr + pid * stride_outm
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        cols = k * BLOCK_K + offs_k
        mask = cols < K
        vals = tl.load(x_base + cols, mask=mask, other=0.0).to(tl.float32)
        quantized = vals / scale
        rounded = tl.where(quantized >= 0, quantized + 0.5, quantized - 0.5)
        rounded = rounded.to(tl.int32)
        clamped = tl.minimum(tl.maximum(rounded, -128), 127)
        tl.store(out_base + cols, clamped.to(tl.int8), mask=mask)


def dynamic_quantize_activation(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize an activation tensor to symmetric INT8 with *per-token* dynamic scales.

    Parameters
    ----------
    x : torch.Tensor
        Activation tensor of shape ``[M, K]`` (or any shape where the last dim
        is the feature dimension).  dtype is typically bf16 or fp16.

    Returns
    -------
    x_int8 : torch.Tensor
        Quantized tensor of the same shape as ``x``, dtype ``torch.int8``.
    s_a : torch.Tensor
        Per-token (per-row) activation scales, shape ``[M]``.
        ``s_a[i] = max(abs(x[i, :])) / 127`` (the 1/127 factor is absorbed
        so the GEMM kernel only needs to multiply).
    """
    orig_shape = x.shape
    if x.dim() > 2:
        x = x.reshape(-1, x.shape[-1])

    M, K = x.shape

    # CPU fallback: use pure PyTorch when CUDA/Triton is unavailable.
    if not x.is_cuda:
        x_f = x.to(torch.float32)
        s_a = x_f.abs().amax(dim=1).clamp(min=1e-8) / 127.0
        x_int8 = (x_f / s_a.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
        if orig_shape != x.shape:
            x_int8 = x_int8.reshape(orig_shape)
            s_a = s_a.reshape(*orig_shape[:-1])
        return x_int8, s_a

    x_contig = x.contiguous()

    # Pick the largest power-of-2 <= 2048 that divides K evenly.
    BLOCK_K = 1
    max_block = 2048
    while BLOCK_K < min(K, max_block) and K % (BLOCK_K * 2) == 0:
        BLOCK_K *= 2

    s_a = torch.empty(M, dtype=torch.float32, device=x.device)
    x_int8 = torch.empty(M, K, dtype=torch.int8, device=x.device)

    _fused_quantize_kernel[(M,)](
        x_contig,
        s_a,
        x_int8,
        x_contig.stride(0),
        x_int8.stride(0),
        K,
        BLOCK_K=BLOCK_K,
    )

    if orig_shape != x.shape:
        x_int8 = x_int8.reshape(orig_shape)
        s_a = s_a.reshape(*orig_shape[:-1])

    return x_int8, s_a
