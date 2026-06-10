"""
INT4 -> INT8 unpack with transpose for W4A8 _int_mm GEMM.

Unpacks packed INT4 weights (uint8, 2 values per byte) to int8
in transposed [K, N] layout, suitable for direct use as the second
argument to torch._int_mm without an extra transpose copy.

Layout: low nibble = even index, high nibble = odd index (two's complement).
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
    def _unpack_int4_to_int8_transposed_kernel(
        packed_ptr,
        out_ptr,
        stride_packed_n,
        stride_packed_k,
        stride_out_k,
        stride_out_n,
        K,
        N,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_k = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)

        k_off = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        k_mask = k_off < K
        n_mask = n_off < N

        packed_col = k_off // 2
        is_high = (k_off % 2).to(tl.uint8)

        packed_ptrs = packed_ptr + n_off[None, :] * stride_packed_n + packed_col[:, None] * stride_packed_k
        packed_mask = k_mask[:, None] & n_mask[None, :]
        byte_vals = tl.load(packed_ptrs, mask=packed_mask, other=0).to(tl.uint8)

        low = byte_vals & 0x0F
        high = (byte_vals >> 4) & 0x0F
        nibble = tl.where(is_high[:, None] == 0, low, high).to(tl.int8)

        int4_vals = tl.where(nibble >= 8, nibble - 16, nibble).to(tl.int8)

        out_ptrs = out_ptr + k_off[:, None] * stride_out_k + n_off[None, :] * stride_out_n
        tl.store(out_ptrs, int4_vals, mask=packed_mask)


def unpack_int4_to_int8_transposed(packed: torch.Tensor, K: int) -> torch.Tensor:
    """Unpack INT4 packed weights to int8 in transposed [K, N] layout.

    Output layout is [K, N] which matches torch._int_mm's second argument
    requirement, avoiding a runtime transpose copy.

    Args:
        packed: [N, K_packed] uint8, 2 INT4 values per byte
        K:      original (unpacked) feature dimension

    Returns:
        weight_int8_t: [K, N] int8 (values in [-8, 7])
    """
    if not _HAS_TRITON:
        raise RuntimeError("Triton is required for unpack_int4_to_int8_transposed")

    N = packed.shape[0]
    out = torch.empty(K, N, dtype=torch.int8, device=packed.device)

    BLOCK_K = 128
    BLOCK_N = 64

    grid = (triton.cdiv(K, BLOCK_K), triton.cdiv(N, BLOCK_N))

    _unpack_int4_to_int8_transposed_kernel[grid](
        packed, out,
        packed.stride(0), packed.stride(1),
        out.stride(0), out.stride(1),
        K, N,
        BLOCK_K=BLOCK_K,
        BLOCK_N=BLOCK_N,
    )

    return out
