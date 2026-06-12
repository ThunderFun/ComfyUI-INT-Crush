"""
Fused INT8 GEMM + per-channel/per-token scale + bias — Triton kernel.

  y = GEMM_int32(x_int8, w_int8^T) * s_a * s_w + bias

s_a and s_w already have the 1/127 factor absorbed during quantization.

Also provides:
  - Fused quantize+GEMM+dequant kernel (single launch for small M)
  - Fused SiLU/GELU activation epilogue
  - Fused residual add epilogue
"""

import torch
import triton
import triton.language as tl


_configs_gemm = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128}, num_stages=3, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=4, num_warps=8),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128}, num_stages=4, num_warps=8),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 256}, num_stages=3, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 128}, num_stages=4, num_warps=8),
]


@triton.autotune(configs=_configs_gemm, key=["M", "N", "K"])
@triton.jit
def _fused_int8_gemm_dequant_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Pointers to scales & bias
    s_a_ptr, s_w_ptr, bias_ptr,
    # Optional residual pointer
    residual_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_rm, stride_rn,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    # Runtime parameters (single kernel specialization)
    ACTIVATION: tl.int32,   # 0=none, 1=silu, 2=gelu
    HAS_RESIDUAL: tl.int32,
):
    # Map program id to the block of C it should compute.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Offsets for this block.
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Accumulator (INT32).
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    # Pointer increments for A and B.
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Loop over K dimension.
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0)

        acc += tl.dot(a, b, allow_tf32=False)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # --- Dequantisation ---
    s_a = tl.load(s_a_ptr + offs_m, mask=offs_m < M, other=1.0)
    s_w = tl.load(s_w_ptr + offs_n, mask=offs_n < N, other=1.0)

    acc_f32 = acc.to(tl.float32)

    # 1/127 has already been absorbed into s_a and s_w during pre-processing.
    scale = s_a[:, None] * s_w[None, :]
    acc_f32 = acc_f32 * scale

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc_f32 = acc_f32 + bias[None, :]

    # Fused activation epilogue
    if ACTIVATION == 1:  # SiLU: x * sigmoid(x)
        acc_f32 = acc_f32 * tl.sigmoid(acc_f32)
    elif ACTIVATION == 2:  # GELU (approximate)
        acc_f32 = acc_f32 * 0.5 * (1.0 + tl.erf(acc_f32 * 0.7071067811865476))

    # Fused residual add epilogue
    if HAS_RESIDUAL:
        r_ptrs = residual_ptr + (offs_m[:, None] * stride_rm + offs_n[None, :] * stride_rn)
        residual = tl.load(r_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)
        acc_f32 = acc_f32 + residual

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc_f32, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def fused_int8_gemm_dequant(
    x_int8: torch.Tensor,
    w_int8: torch.Tensor,
    s_w: torch.Tensor,
    s_a: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    activation: int = 0,
    residual: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Fused INT8 GEMM with per-channel and per-token dequantisation.

    Parameters
    ----------
    x_int8 : [M, K] int8 quantized activations
    w_int8 : [N, K] int8 quantized weights (transposed for GEMM)
    s_w : [N] per-channel weight scales
    s_a : [M] per-token activation scales
    bias : [N] optional bias
    out_dtype : output dtype, default bfloat16
    activation : fused activation: 0=none, 1=silu, 2=gelu
    residual : [M, N] optional residual to add after activation
    out : [M, N] optional pre-allocated output buffer

    Returns
    -------
    [M, N] tensor in out_dtype
    """
    assert x_int8.dtype == torch.int8, f"x_int8 must be int8, got {x_int8.dtype}"
    assert w_int8.dtype == torch.int8, f"w_int8 must be int8, got {w_int8.dtype}"

    M, K = x_int8.shape
    N, K_w = w_int8.shape
    assert K == K_w, f"K mismatch: {K} vs {K_w}"

    if out is not None:
        assert out.shape == (M, N), f"out shape mismatch: {out.shape} vs ({M}, {N})"
        c = out
    else:
        c = torch.empty((M, N), dtype=out_dtype, device=x_int8.device)

    GROUP_M = 8
    has_bias = bias is not None
    if not has_bias:
        bias = x_int8.new_empty(0, dtype=x_int8.dtype)  # dummy; kernel won't read it

    has_residual = residual is not None
    if not has_residual:
        residual = x_int8.new_empty(0, dtype=x_int8.dtype)  # dummy

    def grid(META):
        return (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

    _fused_int8_gemm_dequant_kernel[grid](
        x_int8, w_int8, c,
        s_a, s_w, bias,
        residual,
        M, N, K,
        x_int8.stride(0), x_int8.stride(1),
        w_int8.stride(1), w_int8.stride(0),
        c.stride(0), c.stride(1),
        residual.stride(0) if has_residual else 0,
        residual.stride(1) if has_residual else 0,
        GROUP_M=GROUP_M,
        HAS_BIAS=has_bias,
        ACTIVATION=activation,
        HAS_RESIDUAL=has_residual,
    )

    return c


# ── Fused quantize + GEMM + dequant (single kernel for small M) ────────────

_configs_fused_qgd = [
    # These configs are for the fused quantize+GEMM+dequant kernel.
    # BLOCK_K must be >= K for the single-pass quantize approach,
    # so we use large BLOCK_K values.
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128}, num_stages=3, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128}, num_stages=3, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 128}, num_stages=3, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_stages=4, num_warps=8),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 256}, num_stages=3, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 256}, num_stages=3, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 256}, num_stages=3, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 256}, num_stages=3, num_warps=8),
]


@triton.autotune(configs=_configs_fused_qgd, key=["M", "N", "K"])
@triton.jit
def _fused_quant_int8_gemm_dequant_kernel(
    # Pointers to matrices
    x_ptr, b_ptr, c_ptr,
    # Pointers to scales & bias
    s_w_ptr, bias_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_xm, stride_xk,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Single kernel: quantize fp16→int8 + INT8 GEMM + dequant.

    Phase 1: Loop over K tiles to compute per-row abs-max (in fp32).
    Phase 2: Compute scale = max / 127.0.
    Phase 3: Loop over K tiles, quantize fp16→int8 in registers, accumulate GEMM.
    Phase 4: Dequant epilogue (scale × weight_scale + bias).
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M

    # Phase 1: Compute per-row abs-max across K dimension.
    row_max = tl.zeros([BLOCK_M], dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        cols = k * BLOCK_K + offs_k
        k_mask = cols < K
        x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + cols[None, :] * stride_xk)
        vals = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
        abs_vals = tl.abs(vals)
        block_max = tl.max(abs_vals, axis=1)
        row_max = tl.maximum(row_max, block_max)

    # Phase 2: Compute per-token scale.
    s_a = tl.maximum(row_max, 1e-8) / 127.0  # [BLOCK_M]

    # Phase 3: Quantize + GEMM accumulation.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    n_mask = offs_n < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        cols = k * BLOCK_K + offs_k
        k_mask = cols < K
        x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + cols[None, :] * stride_xk)
        vals = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
        # Quantize: round(vals / scale) clamped to [-128, 127]
        quantized = vals / s_a[:, None]
        rounded = tl.where(quantized >= 0, quantized + 0.5, quantized - 0.5)
        rounded = tl.minimum(tl.maximum(rounded.to(tl.int32), -128), 127)
        x_int8 = rounded.to(tl.int8)

        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0)
        acc += tl.dot(x_int8, b, allow_tf32=False)
        b_ptrs += BLOCK_K * stride_bk

    # Phase 4: Dequant epilogue.
    s_w = tl.load(s_w_ptr + offs_n, mask=n_mask, other=1.0)
    acc_f32 = acc.to(tl.float32)
    scale = s_a[:, None] * s_w[None, :]
    acc_f32 = acc_f32 * scale

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
        acc_f32 = acc_f32 + bias[None, :]

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc_f32, mask=m_mask[:, None] & n_mask[None, :])


def fused_quant_int8_gemm_dequant(
    x_fp16: torch.Tensor,
    w_int8: torch.Tensor,
    s_w: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Fused quantize + INT8 GEMM + dequant in a single kernel launch.

    Eliminates the separate quantize kernel and its global memory round-trip.
    Optimal for small M (≤ 128) where kernel launch overhead dominates.

    Parameters
    ----------
    x_fp16 : [M, K] fp16/bf16 activations
    w_int8 : [N, K] int8 quantized weights
    s_w : [N] per-channel weight scales
    bias : [N] optional bias
    out_dtype : output dtype, default bfloat16

    Returns
    -------
    [M, N] tensor in out_dtype
    """
    assert x_fp16.dtype in (torch.float16, torch.bfloat16), f"x must be fp16/bf16, got {x_fp16.dtype}"
    assert w_int8.dtype == torch.int8, f"w_int8 must be int8, got {w_int8.dtype}"

    M, K = x_fp16.shape
    N, K_w = w_int8.shape
    assert K == K_w, f"K mismatch: {K} vs {K_w}"

    if out is not None and out.shape == (M, N):
        c = out
    else:
        c = torch.empty((M, N), dtype=out_dtype, device=x_fp16.device)

    GROUP_M = 8
    has_bias = bias is not None
    if not has_bias:
        bias = x_fp16.new_empty(0, dtype=x_fp16.dtype)  # dummy

    x_contig = x_fp16 if x_fp16.is_contiguous() else x_fp16.contiguous()

    def grid(META):
        return (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

    _fused_quant_int8_gemm_dequant_kernel[grid](
        x_contig, w_int8, c,
        s_w, bias,
        M, N, K,
        x_contig.stride(0), x_contig.stride(1),
        w_int8.stride(1), w_int8.stride(0),
        c.stride(0), c.stride(1),
        GROUP_M=GROUP_M,
        HAS_BIAS=has_bias,
    )

    return c
