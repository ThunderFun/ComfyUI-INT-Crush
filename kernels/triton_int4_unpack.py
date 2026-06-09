"""
INT4 → float16 unpack + scale kernel for W4A16 inference.

Unpacks packed INT4 weights (uint8, 2 values per byte) to float16 with
per-row scale applied, matching PyTorch quality on the CUDA path.

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
    def _unpack_int4_scaled_kernel(
        packed_ptr,     # [N, K_packed] uint8
        scale_ptr,      # [N] float16 per-row scale
        out_ptr,        # [N, K] float16
        stride_packed_n,
        stride_out_n,
        K,              # original (unpacked) last dim
        BLOCK_K: tl.constexpr,
    ):
        pid_n = tl.program_id(axis=0)
        pid_k = tl.program_id(axis=1)

        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = offs_k < K

        # Which packed byte and nibble for each output element
        packed_col = offs_k // 2
        is_high = (offs_k % 2).to(tl.uint8)

        packed_base = packed_ptr + pid_n * stride_packed_n
        byte_vals = tl.load(packed_base + packed_col, mask=mask, other=0).to(tl.uint8)

        # Extract nibble: low (even) or high (odd)
        nibble = tl.where(is_high == 0, byte_vals & 0x0F, (byte_vals >> 4) & 0x0F).to(tl.int8)

        # Sign extend: values >= 8 are negative in two's complement
        int4_vals = tl.where(nibble >= 8, nibble - 16, nibble).to(tl.float16)

        # Apply per-row scale
        row_scale = tl.load(scale_ptr + pid_n).to(tl.float16)
        result = int4_vals * row_scale

        out_base = out_ptr + pid_n * stride_out_n
        tl.store(out_base + offs_k, result, mask=mask)


def unpack_int4_to_float16(packed: torch.Tensor, scale: torch.Tensor, K: int) -> torch.Tensor:
    """Unpack INT4 packed weights to float16 with per-row scale applied.

    Args:
        packed: [N, K_packed] uint8, 2 INT4 values per byte
        scale:  [N] float16 per-row scales
        K:      original (unpacked) feature dimension

    Returns:
        weight_f16: [N, K] float16 with scale applied
    """
    N = packed.shape[0]
    out = torch.empty(N, K, dtype=torch.float16, device=packed.device)

    BLOCK_K = 2048
    grid = (N, triton.cdiv(K, BLOCK_K))

    _unpack_int4_scaled_kernel[grid](
        packed, scale, out,
        packed.stride(0), out.stride(0),
        K,
        BLOCK_K=BLOCK_K,
    )

    return out
