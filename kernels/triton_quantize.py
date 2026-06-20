"""
Dynamic per-token INT8 quantization — fused Triton implementation.

  Input:  x_bf16 [M, K]
  Output: x_int8 [M, K], s_a [M]

Single fused kernel: computes per-token abs-max scale and quantizes in
two internal loops over K within the same block launch.

Also provides fused norm+quantize kernels (LayerNorm, RMSNorm) and a
fused softmax+quantize kernel that collapse their respective pipelines
into a single kernel, avoiding the intermediate fp round-trip.
"""

import torch
import triton
import triton.language as tl


def _pick_block_k(K: int, max_block: int = 2048) -> int:
    """Largest power-of-2 ≤ max_block that divides K, for unrolled tile loops."""
    BLOCK_K = 1
    while BLOCK_K < min(K, max_block) and K % (BLOCK_K * 2) == 0:
        BLOCK_K *= 2
    return BLOCK_K


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


def dynamic_quantize_activation(
    x: torch.Tensor,
    s_a_out: torch.Tensor | None = None,
    x_int8_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize activations to symmetric INT8 with per-token dynamic scales.

    Parameters
    ----------
    x : [M, K] activation tensor (bf16/fp16)
    s_a_out : optional pre-allocated [M] float32 buffer for scales
    x_int8_out : optional pre-allocated [M, K] int8 buffer for output

    Returns
    -------
    x_int8 : [M, K] int8 quantized tensor
    s_a : [M] per-token scales (max-abs / 127, absorbed for GEMM)
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

    x_contig = x if x.is_contiguous() else x.contiguous()

    # Pick the largest power-of-2 <= 2048 that divides K evenly.
    BLOCK_K = _pick_block_k(K)

    if s_a_out is not None and s_a_out.shape == (M,) and s_a_out.dtype == torch.float32:
        s_a = s_a_out
    else:
        s_a = torch.empty(M, dtype=torch.float32, device=x.device)
    if x_int8_out is not None and x_int8_out.shape == (M, K) and x_int8_out.dtype == torch.int8:
        x_int8 = x_int8_out
    else:
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


# ── Fused LayerNorm + INT8 quantize ─────────────────────────────────────────
#
# 4 passes over K: mean, variance, abs-max of normalized values, quantize.
# The separate path writes the normalized fp tensor to global memory and
# immediately re-reads it for quantization; this kernel avoids that.


@triton.jit
def _fused_layernorm_quantize_kernel(
    x_ptr, gamma_ptr, beta_ptr, out_ptr, s_ptr,
    stride_xm, stride_outm,
    K, eps,
    BLOCK_K: tl.constexpr,
    HAS_BETA: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs_k = tl.arange(0, BLOCK_K)
    x_base = x_ptr + pid * stride_xm

    # Pass 1: compute mean (sum all elements, count = K)
    sum_val = tl.zeros([], dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        cols = k * BLOCK_K + offs_k
        mask = cols < K
        vals = tl.load(x_base + cols, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(vals, axis=0)
    mean = sum_val / K

    # Pass 2: compute variance
    var_val = tl.zeros([], dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        cols = k * BLOCK_K + offs_k
        mask = cols < K
        vals = tl.load(x_base + cols, mask=mask, other=0.0).to(tl.float32)
        diff = vals - mean
        var_val += tl.sum(diff * diff, axis=0)
    var_val = var_val / K
    rstd = 1.0 / tl.sqrt(var_val + eps)

    # Pass 3: normalize, apply affine, and compute per-row abs-max for scale
    row_max = tl.zeros([], dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        cols = k * BLOCK_K + offs_k
        mask = cols < K
        vals = tl.load(x_base + cols, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        normed = (vals - mean) * rstd
        out_val = normed * gamma
        if HAS_BETA:
            beta = tl.load(beta_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            out_val = out_val + beta
        abs_val = tl.abs(out_val)
        block_max = tl.max(abs_val, axis=0)
        row_max = tl.maximum(row_max, block_max)

    scale = tl.maximum(row_max, 1e-8) / 127.0
    tl.store(s_ptr + pid, scale)

    # Pass 4: normalize + quantize + store
    out_base = out_ptr + pid * stride_outm
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        cols = k * BLOCK_K + offs_k
        mask = cols < K
        vals = tl.load(x_base + cols, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        normed = (vals - mean) * rstd
        out_val = normed * gamma
        if HAS_BETA:
            beta = tl.load(beta_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            out_val = out_val + beta
        quantized = out_val / scale
        rounded = tl.where(quantized >= 0, quantized + 0.5, quantized - 0.5)
        clamped = tl.minimum(tl.maximum(rounded.to(tl.int32), -128), 127)
        tl.store(out_base + cols, clamped.to(tl.int8), mask=mask)


def fused_layernorm_quantize(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor | None = None,
    eps: float = 1e-5,
    s_a_out: torch.Tensor | None = None,
    x_int8_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused LayerNorm + per-row symmetric INT8 quantization.

    Equivalent to ``F.layer_norm(x, (K,), gamma, beta, eps)`` followed by
    ``dynamic_quantize_activation()``, but avoids the intermediate fp
    round-trip through global memory by performing both in one kernel.

    Parameters
    ----------
    x : [..., K] fp16/bf16/fp32 activation tensor
    gamma : [K] LayerNorm weight
    beta  : [K] LayerNorm bias, or None
    eps   : variance epsilon
    s_a_out   : optional pre-allocated [M] float32 output for scales
    x_int8_out: optional pre-allocated [M, K] int8 output for quantized values

    Returns
    -------
    x_int8 : [..., K] int8 quantized activations
    s_a    : [...]    float32 per-row scales
    """
    orig_shape = x.shape
    if x.dim() > 2:
        x = x.reshape(-1, x.shape[-1])
    M, K = x.shape

    # CPU fallback (also used when CUDA/Triton unavailable).
    if not x.is_cuda:
        normed = torch.nn.functional.layer_norm(
            x.to(torch.float32), (K,), gamma.to(torch.float32),
            beta.to(torch.float32) if beta is not None else None, eps,
        ).to(x.dtype)
        s_a = normed.float().abs().amax(dim=1).clamp(min=1e-8) / 127.0
        x_int8 = (normed.float() / s_a.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
        if orig_shape != x.shape:
            x_int8 = x_int8.reshape(orig_shape)
            s_a = s_a.reshape(*orig_shape[:-1])
        return x_int8, s_a

    x_contig = x if x.is_contiguous() else x.contiguous()
    gamma_c = gamma if gamma.is_contiguous() else gamma.contiguous()
    beta_c = beta if beta is not None and beta.is_contiguous() else beta

    BLOCK_K = _pick_block_k(K)
    has_beta = beta is not None

    if s_a_out is not None and s_a_out.shape == (M,) and s_a_out.dtype == torch.float32:
        s_a = s_a_out
    else:
        s_a = torch.empty(M, dtype=torch.float32, device=x.device)
    if x_int8_out is not None and x_int8_out.shape == (M, K) and x_int8_out.dtype == torch.int8:
        x_int8 = x_int8_out
    else:
        x_int8 = torch.empty(M, K, dtype=torch.int8, device=x.device)

    _fused_layernorm_quantize_kernel[(M,)](
        x_contig, gamma_c, beta_c if beta_c is not None else x_contig,
        x_int8, s_a,
        x_contig.stride(0), x_int8.stride(0),
        K, eps,
        BLOCK_K=BLOCK_K,
        HAS_BETA=has_beta,
    )

    if orig_shape != x.shape:
        x_int8 = x_int8.reshape(orig_shape)
        s_a = s_a.reshape(*orig_shape[:-1])

    return x_int8, s_a


# ── Fused RMSNorm + INT8 quantize ────────────────────────────────────────────
#
# 3 passes over K (sum-of-squares, abs-max, quantize).  One fewer pass than
# LayerNorm because RMSNorm has no mean subtraction.


@triton.jit
def _fused_rmsnorm_quantize_kernel(
    x_ptr, gamma_ptr, out_ptr, s_ptr,
    stride_xm, stride_outm,
    K, eps,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs_k = tl.arange(0, BLOCK_K)
    x_base = x_ptr + pid * stride_xm

    # Pass 1: compute sum of squares
    sum_sq = tl.zeros([], dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        cols = k * BLOCK_K + offs_k
        mask = cols < K
        vals = tl.load(x_base + cols, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(vals * vals, axis=0)
    rms = tl.sqrt(sum_sq / K + eps)
    rstd = 1.0 / rms

    # Pass 2: normalize and compute abs-max for the int8 scale
    row_max = tl.zeros([], dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        cols = k * BLOCK_K + offs_k
        mask = cols < K
        vals = tl.load(x_base + cols, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        normed = vals * rstd * gamma
        abs_val = tl.abs(normed)
        block_max = tl.max(abs_val, axis=0)
        row_max = tl.maximum(row_max, block_max)

    scale = tl.maximum(row_max, 1e-8) / 127.0
    tl.store(s_ptr + pid, scale)

    # Pass 3: normalize + quantize + store
    out_base = out_ptr + pid * stride_outm
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        cols = k * BLOCK_K + offs_k
        mask = cols < K
        vals = tl.load(x_base + cols, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        normed = vals * rstd * gamma
        quantized = normed / scale
        rounded = tl.where(quantized >= 0, quantized + 0.5, quantized - 0.5)
        clamped = tl.minimum(tl.maximum(rounded.to(tl.int32), -128), 127)
        tl.store(out_base + cols, clamped.to(tl.int8), mask=mask)


def fused_rmsnorm_quantize(
    x: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1e-6,
    s_a_out: torch.Tensor | None = None,
    x_int8_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused RMSNorm + per-row symmetric INT8 quantization.

    Equivalent to ``x / sqrt(mean(x^2) + eps) * gamma`` followed by
    ``dynamic_quantize_activation()``, but avoids the intermediate fp
    round-trip through global memory by performing both in one kernel.

    Parameters
    ----------
    x     : [..., K] fp16/bf16/fp32 activation tensor
    gamma : [K] RMSNorm weight
    eps   : variance epsilon
    s_a_out   : optional pre-allocated [M] float32 output for scales
    x_int8_out: optional pre-allocated [M, K] int8 output for quantized values

    Returns
    -------
    x_int8 : [..., K] int8 quantized activations
    s_a    : [...]    float32 per-row scales
    """
    orig_shape = x.shape
    if x.dim() > 2:
        x = x.reshape(-1, x.shape[-1])
    M, K = x.shape

    # CPU fallback.
    if not x.is_cuda:
        xf = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + eps)
        normed = (xf / rms * gamma.to(torch.float32)).to(x.dtype)
        s_a = normed.float().abs().amax(dim=1).clamp(min=1e-8) / 127.0
        x_int8 = (normed.float() / s_a.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
        if orig_shape != x.shape:
            x_int8 = x_int8.reshape(orig_shape)
            s_a = s_a.reshape(*orig_shape[:-1])
        return x_int8, s_a

    x_contig = x if x.is_contiguous() else x.contiguous()
    gamma_c = gamma if gamma.is_contiguous() else gamma.contiguous()

    BLOCK_K = _pick_block_k(K)

    if s_a_out is not None and s_a_out.shape == (M,) and s_a_out.dtype == torch.float32:
        s_a = s_a_out
    else:
        s_a = torch.empty(M, dtype=torch.float32, device=x.device)
    if x_int8_out is not None and x_int8_out.shape == (M, K) and x_int8_out.dtype == torch.int8:
        x_int8 = x_int8_out
    else:
        x_int8 = torch.empty(M, K, dtype=torch.int8, device=x.device)

    _fused_rmsnorm_quantize_kernel[(M,)](
        x_contig, gamma_c, x_int8, s_a,
        x_contig.stride(0), x_int8.stride(0),
        K, eps,
        BLOCK_K=BLOCK_K,
    )

    if orig_shape != x.shape:
        x_int8 = x_int8.reshape(orig_shape)
        s_a = s_a.reshape(*orig_shape[:-1])

    return x_int8, s_a


# ── Fused softmax + INT8 quantize ─────────────────────────────────────────────
#
# 4 passes over K: row-max (numerically stable), sum-of-exp, abs-max for
# quantization scale, quantize+store.  Uses the shift-invariance of softmax
# (subtract row-max before exp) to avoid overflow.


@triton.jit
def _fused_softmax_quantize_kernel(
    x_ptr, out_ptr, s_ptr,
    stride_xm, stride_outm,
    K,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs_k = tl.arange(0, BLOCK_K)
    x_base = x_ptr + pid * stride_xm

    # Pass 1: find global row max (numerical stability for exp)
    global_max = tl.zeros([], dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        cols = k * BLOCK_K + offs_k
        mask = cols < K
        vals = tl.load(x_base + cols, mask=mask, other=-1e30).to(tl.float32)
        block_max = tl.max(vals, axis=0)
        global_max = tl.maximum(global_max, block_max)

    # Pass 2: compute sum of exp(x - global_max)
    sum_exp = tl.zeros([], dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        cols = k * BLOCK_K + offs_k
        mask = cols < K
        vals = tl.load(x_base + cols, mask=mask, other=-1e30).to(tl.float32)
        exp_vals = tl.exp(vals - global_max)
        sum_exp += tl.sum(exp_vals, axis=0)
    inv_sum = 1.0 / sum_exp

    # Pass 3: softmax + abs-max for scale
    row_max = tl.zeros([], dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        cols = k * BLOCK_K + offs_k
        mask = cols < K
        vals = tl.load(x_base + cols, mask=mask, other=-1e30).to(tl.float32)
        softmax_val = tl.exp(vals - global_max) * inv_sum
        abs_val = tl.abs(softmax_val)
        block_max = tl.max(abs_val, axis=0)
        row_max = tl.maximum(row_max, block_max)

    scale = tl.maximum(row_max, 1e-8) / 127.0
    tl.store(s_ptr + pid, scale)

    # Pass 4: quantize and store
    out_base = out_ptr + pid * stride_outm
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        cols = k * BLOCK_K + offs_k
        mask = cols < K
        vals = tl.load(x_base + cols, mask=mask, other=-1e30).to(tl.float32)
        softmax_val = tl.exp(vals - global_max) * inv_sum
        quantized = softmax_val / scale
        rounded = tl.where(quantized >= 0, quantized + 0.5, quantized - 0.5)
        clamped = tl.minimum(tl.maximum(rounded.to(tl.int32), -128), 127)
        tl.store(out_base + cols, clamped.to(tl.int8), mask=mask)


def fused_softmax_quantize(
    x: torch.Tensor,
    s_a_out: torch.Tensor | None = None,
    x_int8_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused softmax + per-row symmetric INT8 quantization.

    Equivalent to ``F.softmax(x, dim=-1)`` followed by
    ``dynamic_quantize_activation()``, but avoids the intermediate fp
    round-trip through global memory by performing both in one kernel.
    Uses the shift-invariant form of softmax (subtract max before exp)
    for numerical stability.

    Parameters
    ----------
    x : [..., K] fp16/bf16/fp32 logits tensor (softmax over last dim)
    s_a_out   : optional pre-allocated [M] float32 output for scales
    x_int8_out: optional pre-allocated [M, K] int8 output for quantized values

    Returns
    -------
    x_int8 : [..., K] int8 quantized softmax probabilities
    s_a    : [...]    float32 per-row scales
    """
    orig_shape = x.shape
    if x.dim() > 2:
        x = x.reshape(-1, x.shape[-1])
    M, K = x.shape

    # CPU fallback.
    if not x.is_cuda:
        sm = torch.softmax(x.to(torch.float32), dim=-1)
        s_a = sm.abs().amax(dim=1).clamp(min=1e-8) / 127.0
        x_int8 = (sm / s_a.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
        if orig_shape != x.shape:
            x_int8 = x_int8.reshape(orig_shape)
            s_a = s_a.reshape(*orig_shape[:-1])
        return x_int8, s_a

    x_contig = x if x.is_contiguous() else x.contiguous()

    BLOCK_K = _pick_block_k(K)

    if s_a_out is not None and s_a_out.shape == (M,) and s_a_out.dtype == torch.float32:
        s_a = s_a_out
    else:
        s_a = torch.empty(M, dtype=torch.float32, device=x.device)
    if x_int8_out is not None and x_int8_out.shape == (M, K) and x_int8_out.dtype == torch.int8:
        x_int8 = x_int8_out
    else:
        x_int8 = torch.empty(M, K, dtype=torch.int8, device=x.device)

    _fused_softmax_quantize_kernel[(M,)](
        x_contig, x_int8, s_a,
        x_contig.stride(0), x_int8.stride(0),
        K,
        BLOCK_K=BLOCK_K,
    )

    if orig_shape != x.shape:
        x_int8 = x_int8.reshape(orig_shape)
        s_a = s_a.reshape(*orig_shape[:-1])

    return x_int8, s_a
