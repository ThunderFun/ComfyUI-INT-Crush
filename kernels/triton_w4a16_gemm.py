"""
Fused W4A16 GEMM — unpack INT4 weights on-the-fly in Triton.

Reads packed INT4 (uint8, 2 values per byte), unpacks to float16 in
registers, and computes the matmul without writing unpacked weights
to global memory.  Per-row weight scales are fused into the epilogue.

Same output quality as the PyTorch path (W4A16), much lower memory
bandwidth than unpack-then-GEMM.
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
    def _w4a16_gemm_kernel(
        # Activations — float16 [M, K]
        a_ptr,
        stride_am,
        stride_ak,
        # Packed weights — uint8 [N, K_packed]  (K_packed = K // 2)
        w_ptr,
        stride_wn,
        stride_wk,
        # Per-row weight scale — float16 [N]
        s_ptr,
        # Output — float16 [M, N]
        o_ptr,
        stride_om,
        stride_on,
        # Dimensions
        M,
        N,
        K,
        # Tile sizes
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)

        # Accumulator in float32 for precision
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        num_k_tiles = tl.cdiv(K, BLOCK_K)

        for k in range(num_k_tiles):
            k_off = k * BLOCK_K

            # ── Load activations tile [BLOCK_M, BLOCK_K] ──
            a_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            a_k = k_off + tl.arange(0, BLOCK_K)
            a_mask_m = a_m < M
            a_mask_k = a_k < K
            a_mask = a_mask_m[:, None] & a_mask_k[None, :]
            a = tl.load(a_ptr + a_m[:, None] * stride_am + a_k[None, :] * stride_ak,
                        mask=a_mask, other=0.0).to(tl.float32)

            # ── Load & unpack weight tile [BLOCK_K, BLOCK_N] ──
            w_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            w_k = k_off + tl.arange(0, BLOCK_K)
            w_mask_n = w_n < N
            w_mask_k = w_k < K
            w_mask = w_mask_k[:, None] & w_mask_n[None, :]

            # Each output col at w_k needs byte at w_k//2 from row w_n
            packed_col = w_k // 2
            is_high = (w_k % 2).to(tl.uint8)  # 0=low nibble, 1=high nibble

            # Load packed bytes: [BLOCK_K, BLOCK_N]
            w_byte = tl.load(
                w_ptr + w_n[None, :] * stride_wn + packed_col[:, None] * stride_wk,
                mask=w_mask, other=0,
            ).to(tl.uint8)

            # Extract nibble
            low = w_byte & 0x0F
            high = (w_byte >> 4) & 0x0F
            nibble = tl.where(is_high[:, None] == 0, low, high).to(tl.int8)

            # Sign extend: values >= 8 are negative
            int4 = tl.where(nibble >= 8, nibble - 16, nibble).to(tl.float32)

            # Apply per-row weight scale
            w_scale = tl.load(s_ptr + w_n, mask=w_mask_n, other=1.0).to(tl.float32)
            w_tile = int4 * w_scale[None, :]

            # ── Accumulate ──
            acc += tl.dot(a, w_tile)

        # ── Store output ──
        o_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        o_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        o_mask = (o_m[:, None] < M) & (o_n[None, :] < N)
        tl.store(o_ptr + o_m[:, None] * stride_om + o_n[None, :] * stride_on,
                 acc.to(tl.float16), mask=o_mask)


def fused_w4a16_gemm(
    activation: torch.Tensor,   # [..., K] float16/bfloat16
    weight_packed: torch.Tensor, # [N, K_packed] uint8
    weight_scale: torch.Tensor,  # [N] float16
) -> torch.Tensor:
    """Fused W4A16 GEMM: unpack INT4 on-the-fly, compute in float16.

    Returns [..., N] in the same dtype as activation.
    """
    orig_dtype = activation.dtype
    orig_shape = activation.shape
    K = orig_shape[-1]
    M = activation.numel() // K
    N = weight_packed.shape[0]
    K_packed = weight_packed.shape[1]
    assert K_packed == K // 2, f"Packed dim mismatch: K={K}, K_packed={K_packed}"

    # Flatten to 2D
    a_flat = activation.reshape(M, K)
    if not a_flat.is_contiguous():
        a_flat = a_flat.contiguous()
    if not weight_packed.is_contiguous():
        weight_packed = weight_packed.contiguous()
    if not weight_scale.is_contiguous():
        weight_scale = weight_scale.contiguous()

    # Triton works best with float16
    a_f16 = a_flat.to(torch.float16)

    out = torch.empty(M, N, dtype=torch.float16, device=activation.device)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64  # Must be even (2 values per byte)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _w4a16_gemm_kernel[grid](
        a_f16, a_f16.stride(0), a_f16.stride(1),
        weight_packed, weight_packed.stride(0), weight_packed.stride(1),
        weight_scale,
        out, out.stride(0), out.stride(1),
        M, N, K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return out.reshape(*orig_shape[:-1], N).to(orig_dtype)
