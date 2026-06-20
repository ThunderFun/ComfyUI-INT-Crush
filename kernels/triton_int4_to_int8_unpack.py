"""
INT4 -> INT8 unpack kernel for W4A8 GEMM.

Unpacks packed INT4 weights (uint8, 2 values per byte) to int8
WITHOUT scale application. The resulting int8 weights are suitable
for use with torch._int_mm (INT8 tensor core GEMM).

Layout: low nibble = even index, high nibble = odd index (two's complement).
"""

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

# Largest K-tile for the unpack kernel — matches the GEMM autotune BLOCK_K.
_DEFAULT_BLOCK_K: int = 2048


if _HAS_TRITON:

    @triton.jit
    def _unpack_int4_to_int8_kernel(
        packed_ptr,
        out_ptr,
        stride_packed_n,
        stride_out_n,
        K,
        BLOCK_K: tl.constexpr,
    ):
        pid_n = tl.program_id(axis=0)
        pid_k = tl.program_id(axis=1)

        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = offs_k < K

        packed_col = offs_k // 2
        is_high = (offs_k % 2).to(tl.uint8)

        packed_base = packed_ptr + pid_n * stride_packed_n
        byte_vals = tl.load(packed_base + packed_col, mask=mask, other=0).to(tl.uint8)

        nibble = tl.where(is_high == 0, byte_vals & 0x0F, (byte_vals >> 4) & 0x0F).to(tl.int8)

        int4_vals = tl.where(nibble >= 8, nibble - 16, nibble).to(tl.int8)

        out_base = out_ptr + pid_n * stride_out_n
        tl.store(out_base + offs_k, int4_vals, mask=mask)


def unpack_int4_to_int8(packed: torch.Tensor, K: int) -> torch.Tensor:
    """Unpack INT4 packed weights to int8 (no scale).

    Args:
        packed: [N, K_packed] uint8, 2 INT4 values per byte
        K:      original (unpacked) feature dimension

    Returns:
        weight_int8: [N, K] int8 (values in [-8, 7])
    """
    if not _HAS_TRITON:
        raise RuntimeError("Triton is required for unpack_int4_to_int8")

    N = packed.shape[0]
    out = torch.empty(N, K, dtype=torch.int8, device=packed.device)

    BLOCK_K = _DEFAULT_BLOCK_K
    grid = (N, triton.cdiv(K, BLOCK_K))

    _unpack_int4_to_int8_kernel[grid](
        packed, out,
        packed.stride(0), out.stride(0),
        K,
        BLOCK_K=BLOCK_K,
    )

    return out


__all__ = ["unpack_int4_to_int8"]
