"""
Fused INT8 GEMM + per-channel/per-token scale + bias — W4A8-optimised Triton kernel.

Purpose-built for the W4A8 inference path where INT4 weights are unpacked
to INT8 before GEMM.  Compared to the generic W8A8 kernel:

  - Autotune configs favour BLOCK_K=32 and BLOCK_K=64 (better register
    utilisation and L1 behaviour with INT4-range values [-8, 7]).
  - No fused SiLU/GELU activation epilogue (unused by W4A8 caller).
  - No fused residual add (unused by W4A8 caller).
  - Fewer kernel parameters → lower register pressure per launch.

  y = GEMM_int32(x_int8, w_int8^T) * s_a * s_w + bias

s_a and s_w already have the 1/127 factor absorbed during quantization.
"""

import torch
import triton
import triton.language as tl


_configs_w4a8 = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=3, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=4, num_warps=8),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=3, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=4, num_warps=8),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=4, num_warps=8),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32}, num_stages=3, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32}, num_stages=4, num_warps=8),
]


@triton.autotune(configs=_configs_w4a8, key=["M", "N", "K"])
@triton.jit
def _w4a8_gemm_dequant_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Pointers to scales & bias
    s_a_ptr, s_w_ptr, bias_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
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

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc_f32, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def fused_w4a8_gemm_dequant(
    x_int8: torch.Tensor,
    w_int8: torch.Tensor,
    s_w: torch.Tensor,
    s_a: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Fused INT8 GEMM with per-channel and per-token dequantisation,
    optimised for W4A8 workloads (INT4 weights unpacked to INT8).

    Parameters
    ----------
    x_int8 : [M, K] int8 quantized activations
    w_int8 : [N, K] int8 quantized weights (transposed for GEMM)
    s_w : [N] per-channel weight scales
    s_a : [M] per-token activation scales
    bias : [N] optional bias
    out_dtype : output dtype, default bfloat16
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

    def grid(META):
        return (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

    _w4a8_gemm_dequant_kernel[grid](
        x_int8, w_int8, c,
        s_a, s_w, bias,
        M, N, K,
        x_int8.stride(0), x_int8.stride(1),
        w_int8.stride(1), w_int8.stride(0),
        c.stride(0), c.stride(1),
        GROUP_M=GROUP_M,
        HAS_BIAS=has_bias,
    )

    return c
