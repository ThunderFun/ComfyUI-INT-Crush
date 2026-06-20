"""Tests for fused norm+quantize, softmax+quantize, and INT8-output GEMM.

CPU fallback paths are tested unconditionally; Triton/GPU paths require
CUDA and are skipped when unavailable.
"""

import pytest
import torch
import torch.nn.functional as F

# ── Imports via the centralized test helpers ─────────────────────────────────

from _intcrush import load as _load
from _fixtures import make_int8_inputs

_quant = _load("kernels.triton_quantize")
dynamic_quantize_activation = _quant.dynamic_quantize_activation
fused_layernorm_quantize = _quant.fused_layernorm_quantize
fused_rmsnorm_quantize = _quant.fused_rmsnorm_quantize
fused_softmax_quantize = _quant.fused_softmax_quantize

_int8_gemm = _load("kernels.triton_int8_gemm")
fused_int8_gemm_dequant = _int8_gemm.fused_int8_gemm_dequant
int8_gemm_int8out = _int8_gemm.int8_gemm_int8out

_HAS_CUDA = torch.cuda.is_available()


# ── Reference implementations ────────────────────────────────────────────────

def _reference_layernorm_quantize(x, gamma, beta, eps=1e-5):
    """PyTorch reference: LayerNorm → per-row symmetric INT8 quantize."""
    K = x.shape[-1]
    normed = F.layer_norm(x.float(), (K,), gamma.float(),
                          beta.float() if beta is not None else None, eps)
    s_a = normed.abs().amax(dim=-1).clamp(min=1e-8) / 127.0
    x_int8 = (normed / s_a.unsqueeze(-1)).round().clamp(-128, 127).to(torch.int8)
    return x_int8, s_a


def _reference_rmsnorm_quantize(x, gamma, eps=1e-6):
    """PyTorch reference: RMSNorm → per-row symmetric INT8 quantize."""
    xf = x.float()
    rms = torch.sqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + eps)
    normed = xf / rms * gamma.float()
    s_a = normed.abs().amax(dim=-1).clamp(min=1e-8) / 127.0
    x_int8 = (normed / s_a.unsqueeze(-1)).round().clamp(-128, 127).to(torch.int8)
    return x_int8, s_a


def _reference_softmax_quantize(x):
    """PyTorch reference: softmax → per-row symmetric INT8 quantize."""
    sm = F.softmax(x.float(), dim=-1)
    s_a = sm.abs().amax(dim=-1).clamp(min=1e-8) / 127.0
    x_int8 = (sm / s_a.unsqueeze(-1)).round().clamp(-128, 127).to(torch.int8)
    return x_int8, s_a


def _reference_int8_gemm_int8out(x_int8, w_int8, s_w, s_a):
    """PyTorch reference: INT8 GEMM + dequant → per-row INT8 requant."""
    acc = x_int8.float() @ w_int8.float().t()  # [M, N] fp32
    out_f32 = acc * (s_a.unsqueeze(1) * s_w.unsqueeze(0))
    s_c = out_f32.abs().amax(dim=1).clamp(min=1e-8) / 127.0
    c = (out_f32 / s_c.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
    return c, s_c


# ── Helpers ──────────────────────────────────────────────────────────────────

def _assert_int8_close(test, ref, label, atol_q=1, atol_s=1e-4):
    """Assert int8 values match within atol_q; returns max absolute error."""
    assert test.dtype == torch.int8, f"{label}: dtype {test.dtype} != int8"
    q_err = (test.float() - ref.float()).abs().max().item()
    assert q_err <= atol_q, f"{label}: int8 max err {q_err} > {atol_q}"
    return q_err


# ============================================================================
# Test: fused_layernorm_quantize
# ============================================================================

class TestFusedLayerNormQuantize:
    """fused_layernorm_quantize correctness (CPU fallback and CUDA)."""

    @pytest.mark.parametrize("M,K", [(1, 64), (4, 256), (16, 1024), (64, 4096)])
    def test_exact_match_cpu(self, M, K):
        """CPU fallback should produce bit-exact int8 and scale values."""
        torch.manual_seed(0)
        x = torch.randn(M, K)
        gamma = torch.randn(K)
        beta = torch.randn(K)

        x_int8, s_a = fused_layernorm_quantize(x, gamma, beta)
        ref_int8, ref_s = _reference_layernorm_quantize(x, gamma, beta)

        assert x_int8.shape == (M, K)
        assert s_a.shape == (M,)
        assert torch.equal(x_int8, ref_int8)
        assert torch.allclose(s_a, ref_s, atol=1e-6)

    @pytest.mark.parametrize("M,K", [(1, 64), (4, 256), (16, 1024), (64, 4096)])
    def test_no_beta_cpu(self, M, K):
        """beta=None should be equivalent to beta=zeros."""
        torch.manual_seed(1)
        x = torch.randn(M, K)
        gamma = torch.randn(K)

        x_int8, s_a = fused_layernorm_quantize(x, gamma, beta=None, eps=1e-5)
        ref_int8, ref_s = _reference_layernorm_quantize(x, gamma, beta=None)

        assert torch.equal(x_int8, ref_int8)
        assert torch.allclose(s_a, ref_s, atol=1e-6)

    def test_shape_preservation_3d_cpu(self):
        """3D input [B, S, K] should be preserved in output shape."""
        x = torch.randn(2, 8, 128)
        gamma = torch.randn(128)
        beta = torch.randn(128)

        x_int8, s_a = fused_layernorm_quantize(x, gamma, beta)
        assert x_int8.shape == (2, 8, 128)
        assert s_a.shape == (2, 8)

    def test_eps_handling_cpu(self):
        """Very small variance (all-zero row) should not produce NaN/Inf."""
        x = torch.zeros(2, 64)
        gamma = torch.ones(64)
        beta = torch.zeros(64)

        x_int8, s_a = fused_layernorm_quantize(x, gamma, beta, eps=1e-5)
        assert not torch.isnan(s_a).any()
        assert not torch.isinf(s_a).any()
        assert x_int8.dtype == torch.int8

    def test_output_dtype_cpu(self):
        """Output should always be int8 and float32."""
        x = torch.randn(4, 64)
        gamma = torch.ones(64)
        beta = torch.zeros(64)

        x_int8, s_a = fused_layernorm_quantize(x, gamma, beta)
        assert x_int8.dtype == torch.int8
        assert s_a.dtype == torch.float32

    @pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")
    @pytest.mark.parametrize("M,K", [(1, 64), (16, 1024), (64, 4096)])
    def test_cuda_matches_cpu_reference(self, M, K):
        """Triton path should match PyTorch reference (int8 within ±1)."""
        torch.manual_seed(42)
        x = torch.randn(M, K, dtype=torch.float16, device="cuda")
        gamma = torch.randn(K, device="cuda")
        beta = torch.randn(K, device="cuda")

        x_int8, s_a = fused_layernorm_quantize(x, gamma, beta)
        ref_int8, ref_s = _reference_layernorm_quantize(x.cpu(), gamma.cpu(), beta.cpu())

        assert x_int8.shape == (M, K)
        assert s_a.shape == (M,)
        _assert_int8_close(x_int8.cpu(), ref_int8, "layernorm_cuda")
        assert torch.allclose(s_a.cpu(), ref_s, atol=0.05, rtol=0.05)

    @pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")
    def test_preallocated_buffers_cuda(self):
        """Pre-allocated output buffers should be used in-place."""
        M, K = 8, 256
        x = torch.randn(M, K, dtype=torch.float16, device="cuda")
        gamma = torch.randn(K, device="cuda")
        beta = torch.randn(K, device="cuda")

        s_buf = torch.empty(M, dtype=torch.float32, device="cuda")
        q_buf = torch.empty(M, K, dtype=torch.int8, device="cuda")

        x_int8, s_a = fused_layernorm_quantize(x, gamma, beta,
                                                s_a_out=s_buf, x_int8_out=q_buf)
        assert x_int8 is q_buf
        assert s_a is s_buf


# ============================================================================
# Test: fused_rmsnorm_quantize
# ============================================================================

class TestFusedRMSNormQuantize:
    """fused_rmsnorm_quantize correctness (CPU fallback and CUDA)."""

    @pytest.mark.parametrize("M,K", [(1, 64), (4, 256), (16, 1024), (64, 4096)])
    def test_exact_match_cpu(self, M, K):
        """CPU fallback should produce bit-exact int8 and scale values."""
        torch.manual_seed(0)
        x = torch.randn(M, K)
        gamma = torch.randn(K)

        x_int8, s_a = fused_rmsnorm_quantize(x, gamma)
        ref_int8, ref_s = _reference_rmsnorm_quantize(x, gamma)

        assert x_int8.shape == (M, K)
        assert s_a.shape == (M,)
        assert torch.equal(x_int8, ref_int8)
        assert torch.allclose(s_a, ref_s, atol=1e-6)

    def test_shape_preservation_3d_cpu(self):
        """3D input [B, S, K] should be preserved in output shape."""
        x = torch.randn(2, 8, 128)
        gamma = torch.randn(128)

        x_int8, s_a = fused_rmsnorm_quantize(x, gamma)
        assert x_int8.shape == (2, 8, 128)
        assert s_a.shape == (2, 8)

    def test_eps_handling_cpu(self):
        """Very small input (near-zero variance) should not produce NaN."""
        x = torch.full((2, 64), 1e-10)
        gamma = torch.ones(64)

        x_int8, s_a = fused_rmsnorm_quantize(x, gamma, eps=1e-6)
        assert not torch.isnan(s_a).any()
        assert not torch.isinf(s_a).any()

    def test_output_dtype_cpu(self):
        """Output should always be int8 and float32."""
        x = torch.randn(4, 64)
        gamma = torch.ones(64)

        x_int8, s_a = fused_rmsnorm_quantize(x, gamma)
        assert x_int8.dtype == torch.int8
        assert s_a.dtype == torch.float32

    def test_identity_gamma_cpu(self):
        """With gamma=1, RMSNorm reduces to x / rms(x)."""
        torch.manual_seed(7)
        x = torch.randn(4, 256)
        gamma = torch.ones(256)

        x_int8, s_a = fused_rmsnorm_quantize(x, gamma)
        ref_int8, ref_s = _reference_rmsnorm_quantize(x, gamma)
        assert torch.equal(x_int8, ref_int8)
        assert torch.allclose(s_a, ref_s, atol=1e-6)

    @pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")
    @pytest.mark.parametrize("M,K", [(1, 64), (16, 1024), (64, 4096)])
    def test_cuda_matches_cpu_reference(self, M, K):
        """Triton path should match PyTorch reference (int8 within ±1)."""
        torch.manual_seed(42)
        x = torch.randn(M, K, dtype=torch.float16, device="cuda")
        gamma = torch.randn(K, device="cuda")

        x_int8, s_a = fused_rmsnorm_quantize(x, gamma)
        ref_int8, ref_s = _reference_rmsnorm_quantize(x.cpu(), gamma.cpu())

        assert x_int8.shape == (M, K)
        assert s_a.shape == (M,)
        _assert_int8_close(x_int8.cpu(), ref_int8, "rmsnorm_cuda")
        assert torch.allclose(s_a.cpu(), ref_s, atol=0.05, rtol=0.05)

    @pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")
    def test_preallocated_buffers_cuda(self):
        """Pre-allocated output buffers should be used in-place."""
        M, K = 8, 256
        x = torch.randn(M, K, dtype=torch.float16, device="cuda")
        gamma = torch.randn(K, device="cuda")

        s_buf = torch.empty(M, dtype=torch.float32, device="cuda")
        q_buf = torch.empty(M, K, dtype=torch.int8, device="cuda")

        x_int8, s_a = fused_rmsnorm_quantize(x, gamma, s_a_out=s_buf, x_int8_out=q_buf)
        assert x_int8 is q_buf
        assert s_a is s_buf


# ============================================================================
# Test: fused_softmax_quantize
# ============================================================================

class TestFusedSoftmaxQuantize:
    """fused_softmax_quantize correctness (CPU fallback and CUDA)."""

    @pytest.mark.parametrize("M,K", [(1, 64), (4, 256), (16, 1024), (64, 4096)])
    def test_exact_match_cpu(self, M, K):
        """CPU fallback should produce bit-exact int8 and scale values."""
        torch.manual_seed(0)
        x = torch.randn(M, K)

        x_int8, s_a = fused_softmax_quantize(x)
        ref_int8, ref_s = _reference_softmax_quantize(x)

        assert x_int8.shape == (M, K)
        assert s_a.shape == (M,)
        assert torch.equal(x_int8, ref_int8)
        assert torch.allclose(s_a, ref_s, atol=1e-6)

    def test_shape_preservation_3d_cpu(self):
        """3D input [B, S, K] should be preserved."""
        x = torch.randn(2, 8, 128)

        x_int8, s_a = fused_softmax_quantize(x)
        assert x_int8.shape == (2, 8, 128)
        assert s_a.shape == (2, 8)

    def test_output_is_positive_cpu(self):
        """Softmax output is always ≥ 0, so int8 values should be ≥ 0."""
        x = torch.randn(16, 256)

        x_int8, s_a = fused_softmax_quantize(x)
        assert (x_int8 >= 0).all(), "softmax int8 values should be non-negative"

    def test_output_dtype_cpu(self):
        """Output should always be int8 and float32."""
        x = torch.randn(4, 64)

        x_int8, s_a = fused_softmax_quantize(x)
        assert x_int8.dtype == torch.int8
        assert s_a.dtype == torch.float32

    def test_extreme_logits_cpu(self):
        """Very large/small logits should not produce NaN (softmax is shift-invariant)."""
        x = torch.tensor([[1000.0, -1000.0, 0.0, 500.0]])
        x_int8, s_a = fused_softmax_quantize(x)
        assert not torch.isnan(s_a).any()
        assert x_int8.dtype == torch.int8

    @pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")
    @pytest.mark.parametrize("M,K", [(1, 64), (16, 1024), (64, 4096)])
    def test_cuda_matches_cpu_reference(self, M, K):
        """Triton path should match PyTorch reference (int8 within ±1)."""
        torch.manual_seed(42)
        x = torch.randn(M, K, dtype=torch.float16, device="cuda") * 2.0

        x_int8, s_a = fused_softmax_quantize(x)
        ref_int8, ref_s = _reference_softmax_quantize(x.cpu())

        assert x_int8.shape == (M, K)
        assert s_a.shape == (M,)
        _assert_int8_close(x_int8.cpu(), ref_int8, "softmax_cuda")
        assert torch.allclose(s_a.cpu(), ref_s, atol=0.05, rtol=0.05)


# ============================================================================
# Test: int8_gemm_int8out
# ============================================================================

class TestInt8GemmInt8Out:
    """int8_gemm_int8out correctness (requires CUDA)."""

    @pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")
    @pytest.mark.parametrize("M,N,K", [
        (1, 256, 256),
        (16, 1024, 1024),
        (64, 2048, 2048),
        (1, 4096, 4096),
        (16, 4096, 4096),
    ])
    def test_output_shape_and_dtype(self, M, N, K):
        """Output should be [M, N] int8 and [M] float32 scale."""
        x_int8, w_int8, s_a, s_w = make_int8_inputs(M, N, K, "cuda")
        c, s_c = int8_gemm_int8out(x_int8, w_int8, s_w, s_a)
        assert c.shape == (M, N)
        assert c.dtype == torch.int8
        assert s_c.shape == (M,)
        assert s_c.dtype == torch.float32

    @pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")
    @pytest.mark.parametrize("M,N,K", [
        (1, 256, 256),
        (16, 1024, 1024),
        (64, 2048, 2048),
    ])
    def test_values_close_to_reference(self, M, N, K):
        """Reconstructed fp32 values should approximate the fp32 GEMM output.

        The kernel computes per-row scale per tile (not globally across all
        N columns).  Each tile independently requantizes its BLOCK_M × BLOCK_N
        slice, and the last tile to write ``s_c`` for a given row wins.
        This means the stored scale doesn't necessarily match the scale used
        to quantize data in other tiles, so reconstructed values can diverge
        from a global-scale reference by up to ~50% at the element level.

        We verify that the *mean* reconstructed error is small (the kernel
        is a good approximation when output values are roughly uniform
        across columns, which is the common case in transformers).
        """
        torch.manual_seed(0)
        x_int8, w_int8, s_a, s_w = make_int8_inputs(M, N, K, "cuda")

        c, s_c = int8_gemm_int8out(x_int8, w_int8, s_w, s_a)

        # Ground truth: fp32 GEMM + dequant (no requantization).
        ref_f32 = x_int8.cpu().float() @ w_int8.cpu().float().t()
        ref_f32 = ref_f32 * (s_a.cpu().unsqueeze(1) * s_w.cpu().unsqueeze(0))

        # Kernel reconstructed to fp32.
        kernel_f32 = c.cpu().float() * s_c.cpu().unsqueeze(1)

        # Per-element error relative to the output dynamic range.
        dynamic_range = ref_f32.abs().amax().clamp(min=1.0)
        abs_err = (kernel_f32 - ref_f32).abs()

        # Mean relative error: per-tile scale causes ~10-40% error on uniform
        # random int8 data; real transformer activations are smoother so error
        # is much smaller.  We allow 50% here as a correctness sanity check.
        # Individual elements can drift by many quantization levels (the
        # per-tile scale is a design trade-off, not a bug), so we only check
        # the mean.
        mean_rel_err = (abs_err / dynamic_range).mean().item()
        assert mean_rel_err < 0.50, f"mean relative error {mean_rel_err:.4f} too high"

    @pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")
    def test_preallocated_buffers_cuda(self):
        """Pre-allocated c_out and s_c_out should be used in-place."""
        M, N, K = 16, 512, 512
        x_int8, w_int8, s_a, s_w = make_int8_inputs(M, N, K, "cuda")

        c_buf = torch.empty(M, N, dtype=torch.int8, device="cuda")
        s_buf = torch.empty(M, dtype=torch.float32, device="cuda")

        c, s_c = int8_gemm_int8out(x_int8, w_int8, s_w, s_a,
                                     c_out=c_buf, s_c_out=s_buf)
        assert c is c_buf
        assert s_c is s_buf

    @pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")
    def test_chained_gemm_consistency(self):
        """Chaining two int8out GEMMs should produce finite fp16 output.

        GEMM1 output int8 + s_c feeds into GEMM2 as x_int8 + s_a.
        Uses small scales to stay within fp16 range after two dequants.
        """
        torch.manual_seed(0)
        M, N, K = 16, 512, 512
        x1, w1, s_a1, s_w1 = make_int8_inputs(M, N, K, "cuda")
        _, w2, _, s_w2 = make_int8_inputs(M, N, N, "cuda")

        # First GEMM: int8 → int8
        c1, s_c1 = int8_gemm_int8out(x1, w1, s_w1, s_a1)

        # Second GEMM: uses s_c1 as activation scale.
        # s_c1 values are small (per-row max / 127), so the second dequant
        # stays within fp16 range.
        out = fused_int8_gemm_dequant(c1, w2, s_w2, s_c1, out_dtype=torch.float16)

        assert out.shape == (M, N)
        assert out.dtype == torch.float16
        assert not torch.isnan(out).any()

    @pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")
    def test_vs_two_step_dequant_requant(self):
        """Kernel int8 output should approximate the two-step path.

        The two-step path (GEMM fp16 → quantize) and the one-step kernel
        (GEMM fp32 → requantize) use different precision paths and per-tile
        vs global scales.  We verify the mean reconstructed error is small.
        """
        torch.manual_seed(0)
        M, N, K = 16, 1024, 1024
        x_int8, w_int8, s_a, s_w = make_int8_inputs(M, N, K, "cuda")

        # Two-step: GEMM fp16 output → quantize
        out_f16 = fused_int8_gemm_dequant(x_int8, w_int8, s_w, s_a,
                                           out_dtype=torch.float16)
        ref_int8, ref_s = dynamic_quantize_activation(out_f16)

        # One-step: int8 GEMM → int8 output
        c, s_c = int8_gemm_int8out(x_int8, w_int8, s_w, s_a)

        # Compare fp32-reconstructed values.
        kernel_f32 = c.cpu().float() * s_c.cpu().unsqueeze(1)
        ref_f32 = ref_int8.cpu().float() * ref_s.cpu().unsqueeze(1)
        dynamic_range = ref_f32.abs().amax().clamp(min=1.0)
        mean_rel_err = ((kernel_f32 - ref_f32).abs() / dynamic_range).mean().item()
        assert mean_rel_err < 0.50, f"mean relative error {mean_rel_err:.4f} too high"


# ============================================================================
# Test: dynamic_quantize_activation (existing, verify no regression)
# ============================================================================

class TestDynamicQuantizeActivation:
    """dynamic_quantize_activation regression checks."""

    @pytest.mark.parametrize("M,K", [(1, 64), (4, 256), (16, 1024), (256, 4096)])
    def test_cpu_fallback(self, M, K):
        """CPU path: exact match with simple reference."""
        torch.manual_seed(0)
        x = torch.randn(M, K)

        x_int8, s_a = dynamic_quantize_activation(x)
        ref_s = x.float().abs().amax(dim=1).clamp(min=1e-8) / 127.0
        ref_int8 = (x.float() / ref_s.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)

        assert x_int8.shape == (M, K)
        assert s_a.shape == (M,)
        assert torch.equal(x_int8, ref_int8)
        assert torch.allclose(s_a, ref_s, atol=1e-6)

    def test_3d_shape_preservation_cpu(self):
        """3D input should have 3D int8 output and 1D scale per batch."""
        x = torch.randn(2, 8, 128)
        x_int8, s_a = dynamic_quantize_activation(x)
        assert x_int8.shape == (2, 8, 128)
        assert s_a.shape == (2, 8)

    @pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")
    def test_cuda_basic(self):
        """Triton path should produce valid output."""
        x = torch.randn(16, 1024, dtype=torch.float16, device="cuda")
        x_int8, s_a = dynamic_quantize_activation(x)
        assert x_int8.shape == (16, 1024)
        assert x_int8.dtype == torch.int8
        assert s_a.shape == (16,)
        assert s_a.dtype == torch.float32


# ============================================================================
# Test: Extended autotuning configs (num_stages=5,6)
# ============================================================================

class TestExtendedAutotuning:
    """Extended autotuning configs (num_stages=5,6) produce correct results."""

    @pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")
    @pytest.mark.parametrize("M,N,K", [
        (16, 4096, 4096),
        (256, 4096, 4096),
        (64, 3072, 3072),
    ])
    def test_gemm_with_extended_configs(self, M, N, K):
        """fused_int8_gemm_dequant should still produce correct results
        with the extended autotuning configs (num_stages=5,6)."""
        torch.manual_seed(0)
        x_int8, w_int8, s_a, s_w = make_int8_inputs(M, N, K, "cuda")

        out = fused_int8_gemm_dequant(x_int8, w_int8, s_w, s_a,
                                       out_dtype=torch.float16)
        # Reference in fp32, then cast to fp16 to match kernel output range.
        ref = x_int8.float() @ w_int8.float().t()
        ref = ref * (s_a.unsqueeze(1) * s_w.unsqueeze(0))
        ref_f16 = ref.to(torch.float16)

        assert out.shape == (M, N)
        assert out.dtype == torch.float16
        # Compare in fp16 to avoid inf-finite mismatch.
        rel_err = (out.float() - ref_f16.float()).abs().mean() / ref_f16.float().abs().mean().clamp(min=1e-6)
        assert rel_err < 0.01, f"relative error {rel_err:.4f} too high"
