"""Fused FHT rotation + per-token INT8 quantization — Triton kernel.

Replaces the two-step path (matmul rotation → separate quantize) with a
single kernel that applies the Fast Hadamard Transform and quantizes in
one pass.  O(N log N) instead of O(N^2) for the rotation step.

Supports power-of-4 rot_sizes: 16, 64, 256, 1024, 4096.
"""

import math

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _fht_quantize_kernel(
        x_ptr, s_ptr, out_ptr, tmp_a_ptr, tmp_b_ptr,
        stride_xm, stride_outm, stride_tmpm,
        K: tl.constexpr, rot_size: tl.constexpr,
        NUM_STAGES: tl.constexpr,
        BLOCK_K: tl.constexpr,
        NUM_CHUNKS: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        x_base = x_ptr + pid * stride_xm
        out_base = out_ptr + pid * stride_outm
        tmp_a = tmp_a_ptr + pid * stride_tmpm
        tmp_b = tmp_b_ptr + pid * stride_tmpm

        # ── Pre-pass: FHT all chunks ──
        for chunk in range(NUM_CHUNKS):
            g: tl.constexpr = chunk * BLOCK_K
            offs = tl.arange(0, BLOCK_K)
            cols = g + offs
            mask = cols < K

            vals = tl.load(x_base + cols, mask=mask, other=0.0).to(tl.float32)
            tl.store(tmp_a + cols, vals, mask=mask)
            tl.debug_barrier()

            # Stage 0 (s=1): tmp_a → tmp_b
            sub = (offs // 1) % 4
            base = offs - sub
            a = tl.load(tmp_a + g + base, mask=mask, other=0.0)
            b = tl.load(tmp_a + g + base + 1, mask=mask, other=0.0)
            c = tl.load(tmp_a + g + base + 2, mask=mask, other=0.0)
            d = tl.load(tmp_a + g + base + 3, mask=mask, other=0.0)
            r0 = (a + b + c - d) * 0.5
            r1 = (a + b - c + d) * 0.5
            r2 = (a - b + c + d) * 0.5
            r3 = (-a + b + c + d) * 0.5
            result = tl.where(sub == 0, r0,
                     tl.where(sub == 1, r1,
                     tl.where(sub == 2, r2, r3)))
            tl.store(tmp_b + g + offs, result, mask=mask)
            tl.debug_barrier()

            if NUM_STAGES > 1:
                # Stage 1 (s=4): tmp_b → tmp_a
                sub = (offs // 4) % 4
                base = offs - sub * 4
                a = tl.load(tmp_b + g + base, mask=mask, other=0.0)
                b = tl.load(tmp_b + g + base + 4, mask=mask, other=0.0)
                c = tl.load(tmp_b + g + base + 8, mask=mask, other=0.0)
                d = tl.load(tmp_b + g + base + 12, mask=mask, other=0.0)
                r0 = (a + b + c - d) * 0.5
                r1 = (a + b - c + d) * 0.5
                r2 = (a - b + c + d) * 0.5
                r3 = (-a + b + c + d) * 0.5
                result = tl.where(sub == 0, r0,
                         tl.where(sub == 1, r1,
                         tl.where(sub == 2, r2, r3)))
                tl.store(tmp_a + g + offs, result, mask=mask)
                tl.debug_barrier()

            if NUM_STAGES > 2:
                # Stage 2 (s=16): tmp_a → tmp_b
                sub = (offs // 16) % 4
                base = offs - sub * 16
                a = tl.load(tmp_a + g + base, mask=mask, other=0.0)
                b = tl.load(tmp_a + g + base + 16, mask=mask, other=0.0)
                c = tl.load(tmp_a + g + base + 32, mask=mask, other=0.0)
                d = tl.load(tmp_a + g + base + 48, mask=mask, other=0.0)
                r0 = (a + b + c - d) * 0.5
                r1 = (a + b - c + d) * 0.5
                r2 = (a - b + c + d) * 0.5
                r3 = (-a + b + c + d) * 0.5
                result = tl.where(sub == 0, r0,
                         tl.where(sub == 1, r1,
                         tl.where(sub == 2, r2, r3)))
                tl.store(tmp_b + g + offs, result, mask=mask)
                tl.debug_barrier()

            if NUM_STAGES > 3:
                # Stage 3 (s=64): tmp_b → tmp_a
                sub = (offs // 64) % 4
                base = offs - sub * 64
                a = tl.load(tmp_b + g + base, mask=mask, other=0.0)
                b = tl.load(tmp_b + g + base + 64, mask=mask, other=0.0)
                c = tl.load(tmp_b + g + base + 128, mask=mask, other=0.0)
                d = tl.load(tmp_b + g + base + 192, mask=mask, other=0.0)
                r0 = (a + b + c - d) * 0.5
                r1 = (a + b - c + d) * 0.5
                r2 = (a - b + c + d) * 0.5
                r3 = (-a + b + c + d) * 0.5
                result = tl.where(sub == 0, r0,
                         tl.where(sub == 1, r1,
                         tl.where(sub == 2, r2, r3)))
                tl.store(tmp_a + g + offs, result, mask=mask)
                tl.debug_barrier()

            if NUM_STAGES > 4:
                # Stage 4 (s=256): tmp_a → tmp_b
                sub = (offs // 256) % 4
                base = offs - sub * 256
                a = tl.load(tmp_a + g + base, mask=mask, other=0.0)
                b = tl.load(tmp_a + g + base + 256, mask=mask, other=0.0)
                c = tl.load(tmp_a + g + base + 512, mask=mask, other=0.0)
                d = tl.load(tmp_a + g + base + 768, mask=mask, other=0.0)
                r0 = (a + b + c - d) * 0.5
                r1 = (a + b - c + d) * 0.5
                r2 = (a - b + c + d) * 0.5
                r3 = (-a + b + c + d) * 0.5
                result = tl.where(sub == 0, r0,
                         tl.where(sub == 1, r1,
                         tl.where(sub == 2, r2, r3)))
                tl.store(tmp_b + g + offs, result, mask=mask)
                tl.debug_barrier()

            if NUM_STAGES > 5:
                # Stage 5 (s=1024): tmp_b → tmp_a
                sub = (offs // 1024) % 4
                base = offs - sub * 1024
                a = tl.load(tmp_b + g + base, mask=mask, other=0.0)
                b = tl.load(tmp_b + g + base + 1024, mask=mask, other=0.0)
                c = tl.load(tmp_b + g + base + 2048, mask=mask, other=0.0)
                d = tl.load(tmp_b + g + base + 3072, mask=mask, other=0.0)
                r0 = (a + b + c - d) * 0.5
                r1 = (a + b - c + d) * 0.5
                r2 = (a - b + c + d) * 0.5
                r3 = (-a + b + c + d) * 0.5
                result = tl.where(sub == 0, r0,
                         tl.where(sub == 1, r1,
                         tl.where(sub == 2, r2, r3)))
                tl.store(tmp_a + g + offs, result, mask=mask)
                tl.debug_barrier()

        if NUM_STAGES % 2 == 0:
            final = tmp_a
        else:
            final = tmp_b

        # ── Pass 1: row max ──
        row_max = tl.zeros([], dtype=tl.float32)
        for chunk in range(NUM_CHUNKS):
            g: tl.constexpr = chunk * BLOCK_K
            offs = tl.arange(0, BLOCK_K)
            cols = g + offs
            mask = cols < K
            vals = tl.load(final + cols, mask=mask, other=0.0)
            block_max = tl.max(tl.abs(vals), axis=0)
            row_max = tl.maximum(row_max, block_max)

        scale = tl.maximum(row_max, 1e-8) / 127.0
        tl.store(s_ptr + pid, scale)

        # ── Pass 2: quantize ──
        for chunk in range(NUM_CHUNKS):
            g: tl.constexpr = chunk * BLOCK_K
            offs = tl.arange(0, BLOCK_K)
            cols = g + offs
            mask = cols < K
            vals = tl.load(final + cols, mask=mask, other=0.0)
            quantized = vals / scale
            rounded = tl.where(quantized >= 0, quantized + 0.5, quantized - 0.5)
            clamped = tl.minimum(tl.maximum(rounded, -128.0), 127.0)
            tl.store(out_base + g + cols, clamped.to(tl.int8), mask=mask)


def fht_quantize_activation(x: torch.Tensor, rot_size: int):
    """Fused FHT rotation + per-token INT8 quantization. O(N log N) instead of O(N^2).

    Parameters
    ----------
    x : [M, K] activation tensor (contiguous, CUDA)
    rot_size : Hadamard group size, power of 4 (16, 64, 256, 1024, 4096)

    Returns
    -------
    x_int8 : [M, K] int8 quantized activations
    s_a : [M] float32 per-token scales
    """
    # Max rows per kernel launch — keeps tmp buffer memory at ~64 MB
    # (2048 * 4096 * 4 * 2 = 64 MiB for BLOCK_K=4096).
    _CHUNK_ROWS = 2048

    orig_features = x.shape[-1]
    if orig_features % rot_size != 0:
        pad = rot_size - (orig_features % rot_size)
        x = F.pad(x, (0, pad))

    M, K = x.shape
    x_contig = x if x.is_contiguous() else x.contiguous()
    num_stages = int(math.log(rot_size) / math.log(4))

    # Pad K to next power of 2 for tl.arange compatibility
    BLOCK_K = 1
    while BLOCK_K < K:
        BLOCK_K *= 2

    s_a = torch.empty(M, dtype=torch.float32, device=x.device)
    x_int8 = torch.empty(M, K, dtype=torch.int8, device=x.device)

    # Allocate tmp buffers sized for one chunk (reused across all chunks)
    chunk = min(_CHUNK_ROWS, M)
    tmp_a = torch.empty(chunk, BLOCK_K, dtype=torch.float32, device=x.device)
    tmp_b = torch.empty(chunk, BLOCK_K, dtype=torch.float32, device=x.device)

    for start in range(0, M, _CHUNK_ROWS):
        end = min(start + _CHUNK_ROWS, M)
        n_rows = end - start

        x_chunk = x_contig[start:end]
        out_chunk = x_int8[start:end]
        s_chunk = s_a[start:end]

        _fht_quantize_kernel[(n_rows,)](
            x_chunk, s_chunk, out_chunk,
            tmp_a[:n_rows], tmp_b[:n_rows],
            x_chunk.stride(0), out_chunk.stride(0), tmp_a.stride(0),
            K, rot_size,
            NUM_STAGES=num_stages,
            BLOCK_K=BLOCK_K,
            NUM_CHUNKS=1,
        )

    if orig_features != K:
        x_int8 = x_int8[:, :orig_features]

    return x_int8, s_a
