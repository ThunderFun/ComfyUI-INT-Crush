"""Tests for _quant_utils — INT-Crush quantization utilities."""

import math
import torch
import pytest

from _intcrush import load as _load
_qu = _load("_quant_utils")
_is_power_of_four = _qu._is_power_of_four
make_hadamard_regular = _qu.make_hadamard_regular
pack_int4 = _qu.pack_int4
unpack_int4 = _qu.unpack_int4
validate_int4_range = _qu.validate_int4_range
INT4_SCALE_DIVISOR = _qu.INT4_SCALE_DIVISOR
INT4_MIN = _qu.INT4_MIN
INT4_MAX = _qu.INT4_MAX


class TestIsPowerOfFour:
    def test_powers(self):
        for n in [4, 16, 64, 256, 1024, 4096]:
            assert _is_power_of_four(n)

    def test_not_powers(self):
        for n in [1, 2, 8, 32, 128, 512, 2048]:
            assert not _is_power_of_four(n)

    def test_zero(self):
        assert not _is_power_of_four(0)

    def test_negative(self):
        assert not _is_power_of_four(-4)


class TestMakeHadamardRegular:
    def test_shape(self):
        for n in [4, 16, 64, 256, 1024]:
            H = make_hadamard_regular(n)
            assert H.shape == (n, n)

    def test_orthogonality(self):
        """H @ H.T should equal identity."""
        for n in [4, 16]:
            H = make_hadamard_regular(n, dtype=torch.float32)
            product = H @ H.T
            identity = torch.eye(n, dtype=torch.float32)
            assert torch.allclose(product, identity, atol=1e-5)

    def test_values_in_range(self):
        """All values should be ±1/sqrt(n)."""
        for n in [4, 16]:
            H = make_hadamard_regular(n)
            val = 1.0 / math.sqrt(n)
            assert torch.allclose(H.abs(), torch.full_like(H, val), atol=1e-5)

    def test_invalid_size(self):
        with pytest.raises(ValueError):
            make_hadamard_regular(12)  # not power of 2

    def test_dtype(self):
        H = make_hadamard_regular(4, dtype=torch.float32)
        assert H.dtype == torch.float32


class TestPackUnpackRoundtrip:
    def test_roundtrip_1d(self):
        values = torch.tensor([-8, -1, 0, 1, 7, -3, 5, 0], dtype=torch.int8)
        packed = pack_int4(values)
        unpacked = unpack_int4(packed, K=values.shape[0])
        assert torch.equal(values, unpacked)

    def test_roundtrip_2d(self):
        values = torch.randint(-8, 8, (4, 32), dtype=torch.int8)
        packed = pack_int4(values)
        unpacked = unpack_int4(packed, K=32)
        assert torch.equal(values, unpacked)

    def test_roundtrip_odd_length(self):
        values = torch.tensor([-8, 0, 7], dtype=torch.int8)
        packed = pack_int4(values)
        unpacked = unpack_int4(packed, K=3)
        assert torch.equal(values, unpacked)

    def test_packed_shape(self):
        values = torch.randint(-8, 8, (4, 33), dtype=torch.int8)
        packed = pack_int4(values)
        assert packed.shape == (4, 17)

    def test_all_min_values(self):
        values = torch.full((2, 16), -8, dtype=torch.int8)
        packed = pack_int4(values)
        unpacked = unpack_int4(packed, K=16)
        assert torch.equal(values, unpacked)

    def test_all_max_values(self):
        values = torch.full((2, 16), 7, dtype=torch.int8)
        packed = pack_int4(values)
        unpacked = unpack_int4(packed, K=16)
        assert torch.equal(values, unpacked)


class TestValidateInt4Range:
    def test_valid(self):
        validate_int4_range(torch.tensor([-8, 0, 7]))

    def test_out_of_range_low(self):
        with pytest.raises(ValueError):
            validate_int4_range(torch.tensor([-9]))

    def test_out_of_range_high(self):
        with pytest.raises(ValueError):
            validate_int4_range(torch.tensor([8]))


class TestConstants:
    def test_scale_divisor(self):
        assert INT4_SCALE_DIVISOR == 7.0

    def test_int4_range(self):
        assert INT4_MIN == -8
        assert INT4_MAX == 7
