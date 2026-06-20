"""Tests for W4A8 GEMM and _int_mm fast paths."""

import torch
import pytest

from _intcrush import load as _load
from _fixtures import make_int4_layer

_qu = _load("_quant_utils")
unpack_int4 = _qu.unpack_int4
pack_int4 = _qu.pack_int4
rotate_activations = _qu.rotate_activations

_HAS_CUDA = torch.cuda.is_available()
_HAS_INT_MM = hasattr(torch, '_int_mm')

try:
    _dynquant = _load("kernels.triton_quantize")
    dynamic_quantize_activation = _dynquant.dynamic_quantize_activation
    _HAS_DYNQUANT = True
except Exception:
    dynamic_quantize_activation = None
    _HAS_DYNQUANT = False

try:
    _int4_int8 = _load("kernels.triton_int4_to_int8_unpack")
    unpack_int4_to_int8 = _int4_int8.unpack_int4_to_int8
    _HAS_INT4_INT8_UNPACK = True
except Exception:
    unpack_int4_to_int8 = None
    _HAS_INT4_INT8_UNPACK = False


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


class TestW4A8Path:
    @pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")
    @pytest.mark.skipif(not _HAS_INT_MM, reason="torch._int_mm required")
    @pytest.mark.skipif(not _HAS_DYNQUANT, reason="Triton dynamic_quantize not available")
    def test_w4a8_vs_w4a16_reference(self):
        N, K, M = 32, 128, 32
        packed, scale_flat, K_orig = make_int4_layer(N, K, device="cuda")
        x = torch.randn(M, K, dtype=torch.float16, device="cuda")
        x_rot = rotate_activations(x, 64)

        unpacked = unpack_int4(packed.cuda(), K_orig)
        weight_f16 = (unpacked.float() * scale_flat.cuda().float().unsqueeze(1)).to(torch.float16)
        out_ref = torch.nn.functional.linear(x_rot, weight_f16)

        x_2d = x_rot.reshape(-1, K)
        x_int8, s_a = dynamic_quantize_activation(x_2d)
        w_int8_t = unpack_int4(packed.cuda(), K_orig).t().contiguous()
        acc = torch._int_mm(x_int8, w_int8_t)

        out_w4a8 = (acc.float() * s_a[:, None] * scale_flat.cuda()[None, :]).to(torch.float16)

        assert out_w4a8.shape == out_ref.shape
        rel_err = (out_w4a8 - out_ref).abs().mean() / out_ref.abs().mean().clamp(min=1e-6)
        assert rel_err < 0.15, f"W4A8 relative error {rel_err:.4f} too high"

    @pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")
    @pytest.mark.skipif(not _HAS_INT_MM, reason="torch._int_mm required")
    @pytest.mark.skipif(not _HAS_DYNQUANT, reason="Triton dynamic_quantize not available")
    def test_w4a8_shape(self):
        N, K, M = 32, 64, 32
        packed, scale_flat, K_orig = make_int4_layer(N, K, device="cuda")
        x = torch.randn(M, K, dtype=torch.float16, device="cuda")
        x_rot = rotate_activations(x, 64)

        x_2d = x_rot.reshape(-1, K)
        x_int8, s_a = dynamic_quantize_activation(x_2d)
        w_int8_t = unpack_int4(packed.cuda(), K_orig).t().contiguous()
        acc = torch._int_mm(x_int8, w_int8_t)
        out = acc.float() * s_a[:, None] * scale_flat.cuda()[None, :]
        assert out.shape == (M, N)
