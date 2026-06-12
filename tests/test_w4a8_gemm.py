"""Tests for W4A8 GEMM, _int_mm fast paths, and fused dequant epilogue."""

import importlib
import torch
import pytest

_qu = importlib.import_module("ComfyUI-INT-Crush._quant_utils")
unpack_int4 = _qu.unpack_int4
pack_int4 = _qu.pack_int4
rotate_activations = _qu.rotate_activations
INT4_SCALE_DIVISOR = _qu.INT4_SCALE_DIVISOR

_HAS_CUDA = torch.cuda.is_available()
_HAS_INT_MM = hasattr(torch, '_int_mm')

try:
    _dynquant = importlib.import_module("ComfyUI-INT-Crush.kernels.triton_quantize")
    dynamic_quantize_activation = _dynquant.dynamic_quantize_activation
    _HAS_DYNQUANT = True
except Exception:
    dynamic_quantize_activation = None
    _HAS_DYNQUANT = False

try:
    _int4_int8 = importlib.import_module("ComfyUI-INT-Crush.kernels.triton_int4_to_int8_unpack")
    unpack_int4_to_int8 = _int4_int8.unpack_int4_to_int8
    _HAS_INT4_INT8_UNPACK = True
except Exception:
    unpack_int4_to_int8 = None
    _HAS_INT4_INT8_UNPACK = False

try:
    _int4_int8_t = importlib.import_module("ComfyUI-INT-Crush.kernels.triton_int4_to_int8_unpack_transpose")
    unpack_int4_to_int8_transposed = _int4_int8_t.unpack_int4_to_int8_transposed
    _HAS_INT4_INT8_UNPACK_T = True
except Exception:
    unpack_int4_to_int8_transposed = None
    _HAS_INT4_INT8_UNPACK_T = False

try:
    _epilogue = importlib.import_module("ComfyUI-INT-Crush.kernels.triton_dequant_epilogue")
    fused_dequant_epilogue = _epilogue.fused_dequant_epilogue
    _HAS_DEQUANT_EPILOGUE = True
except Exception:
    fused_dequant_epilogue = None
    _HAS_DEQUANT_EPILOGUE = False


def _make_int4_layer(N, K, device="cpu"):
    W = torch.randn(N, K, dtype=torch.float32, device=device)
    max_vals = W.abs().amax(dim=1)
    scales = (max_vals / INT4_SCALE_DIVISOR).clamp(min=1e-8).to(torch.float16)
    W_scaled = W / scales.unsqueeze(1).to(W.dtype)
    int_rounded = W_scaled.round().clamp(-8, 7).to(torch.int8)
    packed = pack_int4(int_rounded).to(torch.uint8)
    return packed, scales.reshape(-1), K


class TestUnpackInt4ToInt8:
    def test_pytorch_unpack_returns_int8(self):
        values = torch.tensor([-8, -1, 0, 1, 7, -3, 5, 0], dtype=torch.int8)
        packed = pack_int4(values)
        unpacked = unpack_int4(packed, K=values.shape[0])
        assert unpacked.dtype == torch.int8
        assert torch.equal(values, unpacked)

    def test_pytorch_unpack_2d(self):
        values = torch.randint(-8, 8, (4, 32), dtype=torch.int8)
        packed = pack_int4(values)
        unpacked = unpack_int4(packed, K=32)
        assert unpacked.dtype == torch.int8
        assert torch.equal(values, unpacked)

    @pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")
    @pytest.mark.skipif(not _HAS_INT4_INT8_UNPACK, reason="Triton INT4->INT8 unpack not available")
    def test_triton_roundtrip_cuda(self):
        values = torch.randint(-8, 8, (16, 256), dtype=torch.int8, device="cuda")
        packed = pack_int4(values).to(torch.uint8).cuda()
        result = unpack_int4_to_int8(packed, K=256)
        assert result.dtype == torch.int8
        assert result.shape == (16, 256)
        assert torch.equal(values, result)

    @pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")
    @pytest.mark.skipif(not _HAS_INT4_INT8_UNPACK_T, reason="Triton transposed unpack not available")
    def test_triton_transposed_unpack_cuda(self):
        values = torch.randint(-8, 8, (16, 256), dtype=torch.int8, device="cuda")
        packed = pack_int4(values).to(torch.uint8).cuda()
        result = unpack_int4_to_int8_transposed(packed, K=256)
        assert result.dtype == torch.int8
        assert result.shape == (256, 16)
        assert torch.equal(values.t(), result)

    @pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")
    @pytest.mark.skipif(not _HAS_INT4_INT8_UNPACK_T, reason="Triton transposed unpack not available")
    def test_transposed_matches_regular_unpack(self):
        values = torch.randint(-8, 8, (32, 128), dtype=torch.int8, device="cuda")
        packed = pack_int4(values).to(torch.uint8).cuda()
        regular = unpack_int4_to_int8(packed, K=128) if _HAS_INT4_INT8_UNPACK else unpack_int4(packed, K=128).cuda()
        transposed = unpack_int4_to_int8_transposed(packed, K=128)
        assert torch.equal(regular.t(), transposed)


class TestIntMmDequant:
    @pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")
    @pytest.mark.skipif(not _HAS_INT_MM, reason="torch._int_mm required")
    def test_int_mm_basic(self):
        a = torch.randint(-128, 127, (32, 64), dtype=torch.int8, device="cuda")
        b = torch.randint(-128, 127, (16, 64), dtype=torch.int8, device="cuda")
        result = torch._int_mm(a, b.t().contiguous())
        ref = a.float() @ b.t().float()
        assert torch.allclose(result.float(), ref, atol=1)

    @pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")
    @pytest.mark.skipif(not _HAS_INT_MM, reason="torch._int_mm required")
    def test_int_mm_pretransposed(self):
        a = torch.randint(-128, 127, (32, 64), dtype=torch.int8, device="cuda")
        b = torch.randint(-128, 127, (16, 64), dtype=torch.int8, device="cuda")
        b_t = b.t().contiguous()
        result = torch._int_mm(a, b_t)
        ref = a.float() @ b.t().float()
        assert torch.allclose(result.float(), ref, atol=1)


class TestDequantEpilogue:
    @pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")
    @pytest.mark.skipif(not _HAS_INT_MM, reason="torch._int_mm required")
    def test_dequant_epilogue_vs_pytorch(self):
        M, N = 32, 64
        acc = torch.randint(-10000, 10000, (M, N), dtype=torch.int32, device="cuda")
        s_a = torch.randn(M, dtype=torch.float32, device="cuda").abs().clamp(min=0.01)
        s_w = torch.randn(N, dtype=torch.float32, device="cuda").abs().clamp(min=0.01)
        bias = torch.randn(N, dtype=torch.float32, device="cuda")

        ref = (acc.float() * s_a[:, None] * s_w[None, :] + bias).to(torch.float16)

        if _HAS_DEQUANT_EPILOGUE:
            out = fused_dequant_epilogue(acc, s_a, s_w, out_dtype=torch.float16, bias=bias)
        else:
            out = (acc.float() * s_a[:, None] * s_w[None, :] + bias).to(torch.float16)

        assert out.shape == ref.shape
        assert out.dtype == torch.float16
        assert torch.allclose(out, ref, atol=1e-2, rtol=1e-2)

    @pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")
    @pytest.mark.skipif(not _HAS_INT_MM, reason="torch._int_mm required")
    def test_dequant_epilogue_no_bias(self):
        M, N = 16, 32
        acc = torch.randint(-10000, 10000, (M, N), dtype=torch.int32, device="cuda")
        s_a = torch.randn(M, dtype=torch.float32, device="cuda").abs().clamp(min=0.01)
        s_w = torch.randn(N, dtype=torch.float32, device="cuda").abs().clamp(min=0.01)

        ref = (acc.float() * s_a[:, None] * s_w[None, :]).to(torch.float16)

        if _HAS_DEQUANT_EPILOGUE:
            out = fused_dequant_epilogue(acc, s_a, s_w, out_dtype=torch.float16, bias=None)
        else:
            out = (acc.float() * s_a[:, None] * s_w[None, :]).to(torch.float16)

        assert torch.allclose(out, ref, atol=1e-2, rtol=1e-2)


class TestW4A8Path:
    @pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")
    @pytest.mark.skipif(not _HAS_INT_MM, reason="torch._int_mm required")
    @pytest.mark.skipif(not _HAS_DYNQUANT, reason="Triton dynamic_quantize not available")
    def test_w4a8_vs_w4a16_reference(self):
        N, K, M = 32, 128, 32
        packed, scale_flat, K_orig = _make_int4_layer(N, K, device="cuda")
        x = torch.randn(M, K, dtype=torch.float16, device="cuda")
        x_rot = rotate_activations(x, 64)

        unpacked = unpack_int4(packed.cuda(), K_orig)
        weight_f16 = (unpacked.float() * scale_flat.cuda().float().unsqueeze(1)).to(torch.float16)
        out_ref = torch.nn.functional.linear(x_rot, weight_f16)

        x_2d = x_rot.reshape(-1, K)
        x_int8, s_a = dynamic_quantize_activation(x_2d)
        if _HAS_INT4_INT8_UNPACK_T:
            w_int8_t = unpack_int4_to_int8_transposed(packed.cuda(), K_orig)
        else:
            w_int8_t = unpack_int4(packed.cuda(), K_orig).t().contiguous()
        acc = torch._int_mm(x_int8, w_int8_t)

        if _HAS_DEQUANT_EPILOGUE:
            out_w4a8 = fused_dequant_epilogue(acc, s_a, scale_flat.cuda(), out_dtype=torch.float16)
        else:
            out_w4a8 = (acc.float() * s_a[:, None] * scale_flat.cuda()[None, :]).to(torch.float16)

        assert out_w4a8.shape == out_ref.shape
        rel_err = (out_w4a8 - out_ref).abs().mean() / out_ref.abs().mean().clamp(min=1e-6)
        assert rel_err < 0.15, f"W4A8 relative error {rel_err:.4f} too high"

    @pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")
    @pytest.mark.skipif(not _HAS_INT_MM, reason="torch._int_mm required")
    @pytest.mark.skipif(not _HAS_DYNQUANT, reason="Triton dynamic_quantize not available")
    def test_w4a8_shape(self):
        N, K, M = 32, 64, 32
        packed, scale_flat, K_orig = _make_int4_layer(N, K, device="cuda")
        x = torch.randn(M, K, dtype=torch.float16, device="cuda")
        x_rot = rotate_activations(x, 64)

        x_2d = x_rot.reshape(-1, K)
        x_int8, s_a = dynamic_quantize_activation(x_2d)
        w_int8_t = unpack_int4(packed.cuda(), K_orig).t().contiguous()
        acc = torch._int_mm(x_int8, w_int8_t)
        out = acc.float() * s_a[:, None] * scale_flat.cuda()[None, :]
        assert out.shape == (M, N)
