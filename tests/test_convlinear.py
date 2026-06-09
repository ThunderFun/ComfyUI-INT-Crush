"""Tests for loader.convlinear — ConvLinear4bit module."""

import importlib
import torch
import pytest

ConvLinear4bit = importlib.import_module("ComfyUI-INT-Crush.convlinear").ConvLinear4bit


class TestConvLinear4bit:
    def test_from_float_basic(self):
        """Test conversion from nn.Linear."""
        linear = torch.nn.Linear(64, 32, bias=True)
        convlinear = ConvLinear4bit.from_float(linear, rot_need=True, rot_size=64)
        assert convlinear.in_features == 64
        assert convlinear.out_features == 32
        assert convlinear.bias is not None

    def test_from_float_no_bias(self):
        linear = torch.nn.Linear(64, 32, bias=False)
        convlinear = ConvLinear4bit.from_float(linear, rot_need=True, rot_size=64)
        assert convlinear.bias is None

    def test_weight_shape(self):
        linear = torch.nn.Linear(64, 32)
        convlinear = ConvLinear4bit.from_float(linear, rot_need=True, rot_size=64)
        # Packed: [32, 32] (64/2 = 32 bytes per row)
        assert convlinear.weight.shape == (32, 32)

    def test_scale_shape(self):
        linear = torch.nn.Linear(64, 32)
        convlinear = ConvLinear4bit.from_float(linear, rot_need=True, rot_size=64)
        assert convlinear.scale.shape == (32, 1)

    def test_forward_shape(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        linear = torch.nn.Linear(64, 32).cuda()
        convlinear = ConvLinear4bit.from_float(linear, rot_need=True, rot_size=64)
        x = torch.randn(4, 64, dtype=torch.float16, device="cuda")
        out = convlinear(x)
        assert out.shape == (4, 32)

    def test_forward_no_rotation(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        linear = torch.nn.Linear(64, 32).cuda()
        convlinear = ConvLinear4bit.from_float(linear, rot_need=False, rot_size=64)
        x = torch.randn(4, 64, dtype=torch.float16, device="cuda")
        out = convlinear(x)
        assert out.shape == (4, 32)
