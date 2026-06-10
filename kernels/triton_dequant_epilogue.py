"""
Fused dequant epilogue kernel for _int_mm output.

Takes the int32 accumulator from torch._int_mm, applies per-token
activation scale * per-row weight scale + bias, and writes directly
to fp16/bf16 — eliminating the int32 write + float32 read round-trip.

  out[m, n] = (acc_int32[m, n] * s_a[m] * s_w[n] + bias[n]).to(out_dtype)
"""

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _dequant_epilogue_kernel(
        acc_ptr,
        s_a_ptr,
        s_w_ptr,
        out_ptr,
        bias_ptr,
        stride_acc_m,
        stride_acc_n,
        stride_out_m,
        stride_out_n,
        M,
        N,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)

        m_off = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        m_mask = m_off < M
        n_mask = n_off < N

        acc = tl.load(
            acc_ptr + m_off[:, None] * stride_acc_m + n_off[None, :] * stride_acc_n,
            mask=m_mask[:, None] & n_mask[None, :],
            other=0,
        ).to(tl.float32)

        s_a = tl.load(s_a_ptr + m_off, mask=m_mask, other=1.0).to(tl.float32)
        s_w = tl.load(s_w_ptr + n_off, mask=n_mask, other=1.0).to(tl.float32)

        result = acc * s_a[:, None] * s_w[None, :]

        if HAS_BIAS:
            bias = tl.load(bias_ptr + n_off, mask=n_mask, other=0.0).to(tl.float32)
            result = result + bias[None, :]

        tl.store(
            out_ptr + m_off[:, None] * stride_out_m + n_off[None, :] * stride_out_n,
            result,
            mask=m_mask[:, None] & n_mask[None, :],
        )


def fused_dequant_epilogue(
    acc: torch.Tensor,
    s_a: torch.Tensor,
    s_w: torch.Tensor,
    out_dtype: torch.dtype = torch.float16,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused dequant epilogue: int32 acc → scaled float output.

    Args:
        acc: [M, N] int32 accumulator from torch._int_mm
        s_a: [M] float32 per-token activation scales
        s_w: [N] float32 per-row weight scales
        out_dtype: output dtype (float16 or bfloat16)
        bias: [N] optional bias

    Returns:
        [M, N] tensor in out_dtype
    """
    if not _HAS_TRITON:
        out = acc.float() * s_a[:, None] * s_w[None, :]
        if bias is not None:
            out = out + bias.float()
        return out.to(out_dtype)

    M, N = acc.shape
    out = torch.empty(M, N, dtype=out_dtype, device=acc.device)

    has_bias = bias is not None
    if not has_bias:
        bias = acc.new_empty(0, dtype=torch.float32)

    BLOCK_M = 16
    BLOCK_N = 64

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _dequant_epilogue_kernel[grid](
        acc, s_a, s_w, out, bias,
        acc.stride(0), acc.stride(1),
        out.stride(0), out.stride(1),
        M, N,
        HAS_BIAS=has_bias,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return out
