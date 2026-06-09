"""ComfyUI nodes for INT4/INT8 model loading with INT-Crush.

Provides:
  - SimpleINT4UNetLoader: loads a quantized model with INT4 weights in memory
  - SimpleINT8UNetLoader: loads a quantized model with INT8 weights in memory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._quant_utils import (
    INT4_SCALE_DIVISOR,
    rotate_activations,
    pack_int4,
)

_TRITON_AVAILABLE = False
_TRITON_INT4_UNPACK = False
_TRITON_W4A16_GEMM = False
try:
    from .kernels.triton_int4_unpack import unpack_int4_to_float16
    _TRITON_INT4_UNPACK = True
    print("[INT-Crush] Triton INT4 unpack kernel loaded")
except Exception as e:
    print(f"[INT-Crush] Triton INT4 unpack not available: {e}")

try:
    from .kernels.triton_w4a16_gemm import fused_w4a16_gemm
    _TRITON_W4A16_GEMM = True
    print("[INT-Crush] Triton fused W4A16 GEMM loaded")
except Exception as e:
    print(f"[INT-Crush] Triton W4A16 GEMM not available: {e}")

try:
    from comfy.ops import manual_cast
    _COMFY_OPS_AVAILABLE = True
except ImportError:
    _COMFY_OPS_AVAILABLE = False

try:
    from comfy.ops.manual_cast import cast_bias_weight, uncast_bias_weight
    _MANUAL_CAST_AVAILABLE = True
except ImportError:
    _MANUAL_CAST_AVAILABLE = False


def _requantize_int4(w_float, rot_need, rot_size):
    """Re-quantize a float weight tensor to packed INT4 with per-row scales.

    Used by set_weight() when a LoRA patch or other operation produces a
    float32 weight that needs to be stored back in INT4 format.  Applies
    rotation (if the layer requires it), computes per-row scales, rounds,
    and packs two INT4 values per uint8 byte.

    Args:
        w_float: [out_features, in_features] float weight tensor
        rot_need: whether to apply Hadamard rotation before quantizing
        rot_size: Hadamard group size (16, 64, or 256)

    Returns:
        packed: [out_features, in_features // 2] uint8 packed INT4 weights
        scales: [out_features, 1] float16 per-row scales
    """
    if rot_need:
        w_rot = rotate_activations(w_float, rot_size)
    else:
        w_rot = w_float
    max_vals = w_rot.abs().amax(dim=1, keepdim=True)
    scales = (max_vals / INT4_SCALE_DIVISOR).clamp(min=1e-8).to(torch.float16)
    w_scaled = w_rot / scales.to(w_rot.dtype)
    int_rounded = w_scaled.round().clamp(-8, 7).to(torch.int8)
    packed = pack_int4(int_rounded).to(torch.uint8)
    return packed, scales


if _COMFY_OPS_AVAILABLE:

    class IntCrushInt4Ops(manual_cast):
        """Custom ComfyUI operations for INT-Crush INT4 quantization."""

        _rot_size_override = 0
        _is_gptq = False
        _use_pytorch = True  # Default to PyTorch fallback path
        _perm_count = 0

        class Linear(manual_cast.Linear):

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._is_int4 = False
                self._rot_need = True
                self._rot_size = IntCrushInt4Ops._rot_size_override
                self._perm = None  # PermuQuant permutation indices
                self.register_buffer("weight_scale", None)
                self.weight_comfy_model_dtype = getattr(self.weight, "dtype", None)
                self.bias_comfy_model_dtype = getattr(self.bias, "dtype", None)

            def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys, error_msgs):
                """Intercept weight loading to detect INT-Crush INT4 tensors.

                When the state dict contains a uint8 weight + float16 weight_scale
                pair, this layer enters INT4 mode: the raw packed weights and scales
                are stored directly (bypassing normal nn.Linear loading), and the
                rotation flag is set based on the layer name.

                Also loads an optional PermuQuant permutation tensor (weight.perm).
                """
                weight_key = prefix + "weight"
                scale_key = prefix + "weight_scale"
                perm_key = prefix + "weight.perm"

                if weight_key in state_dict and scale_key in state_dict:
                    wt = state_dict[weight_key]
                    sc = state_dict[scale_key]

                    if wt.dtype == torch.uint8 and sc.dtype == torch.float16:
                        self._is_int4 = True
                        self.weight = nn.Parameter(wt, requires_grad=False)
                        self.register_buffer("weight_scale", sc)

                        # Load PermuQuant permutation if present
                        if perm_key in state_dict:
                            self._perm = state_dict[perm_key].to(torch.int64)
                            IntCrushInt4Ops._perm_count += 1

                        bias_key = prefix + "bias"
                        if bias_key in state_dict:
                            self.bias = nn.Parameter(state_dict[bias_key], requires_grad=False)

                        name_lower = prefix.lower()
                        if self._rot_size == 0:
                            self._rot_need = False
                        else:
                            self._rot_need = not any(
                                p in name_lower for p in ["embed", "norm", "modulation", "output", "lm_head", "proj_out"]
                            )

                        self.weight_comfy_model_dtype = torch.uint8
                        self.weight_scale_comfy_model_dtype = torch.float16
                        if self.bias is not None:
                            self.bias_comfy_model_dtype = self.bias.dtype

                        state_dict.pop(weight_key, None)
                        state_dict.pop(scale_key, None)
                        state_dict.pop(perm_key, None)
                        state_dict.pop(bias_key, None)
                        return

                super()._load_from_state_dict(
                    state_dict, prefix, local_metadata,
                    strict, missing_keys, unexpected_keys, error_msgs
                )

            def _get_weight_scale(self):
                return self.weight_scale

            def convert_weight(self, _weight, inplace=False):
                if not self._is_int4:
                    return _weight
                return self.weight

            def set_weight(self, out_weight, inplace_update=False, seed=0, return_weight=False, **kwargs):
                if not self._is_int4:
                    new_weight = out_weight.to(self.weight.dtype)
                    if return_weight:
                        return new_weight
                    if inplace_update:
                        self.weight.data.copy_(new_weight)
                    else:
                        self.weight = nn.Parameter(new_weight, requires_grad=False)
                    return

                if out_weight.dtype == torch.uint8:
                    if return_weight:
                        return out_weight
                    if inplace_update:
                        self.weight.data.copy_(out_weight)
                    else:
                        self.weight = nn.Parameter(out_weight, requires_grad=False)
                    return

                packed, scales = _requantize_int4(out_weight.float(), self._rot_need, self._rot_size)

                if return_weight:
                    return packed

                if inplace_update:
                    self.weight.data.copy_(packed)
                else:
                    self.weight = nn.Parameter(packed, requires_grad=False)

                if hasattr(self, "weight_scale"):
                    self.weight_scale = scales.to(self.weight.device) if self.weight.device.type != "meta" else scales

            def set_bias(self, out_bias, inplace_update=False, seed=0, return_weight=False, **kwargs):
                if out_bias is None:
                    return None
                new_bias = out_bias
                if return_weight:
                    return new_bias
                if inplace_update:
                    if self.bias is not None:
                        self.bias.data.copy_(new_bias)
                else:
                    self.bias = nn.Parameter(new_bias, requires_grad=False)

            @torch.no_grad()
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # ── Fallback: non-quantized layers use the standard manual_cast path ──
                if not self._is_int4:
                    return super().forward(x)

                # ── ComfyUI weight casting (low-VRAM / offload mode) ──
                # When ComfyUI manages weight offloading, cast_bias_weight() handles
                # moving the packed uint8 weight to the compute device.
                need_cast = (
                    _MANUAL_CAST_AVAILABLE
                    and (
                        getattr(self, "comfy_cast_weights", False)
                        or len(getattr(self, "weight_function", [])) > 0
                        or len(getattr(self, "bias_function", [])) > 0
                        or getattr(self, "weight_lowvram_function", None) is not None
                        or getattr(self, "bias_lowvram_function", None) is not None
                    )
                )

                if need_cast and _MANUAL_CAST_AVAILABLE:
                    weight, bias, offload_stream = cast_bias_weight(
                        self, input=None, dtype=torch.uint8, device=x.device,
                        bias_dtype=x.dtype, offloadable=True
                    )
                else:
                    weight = self.weight
                    bias = self.bias
                    offload_stream = None

                # ── Hadamard rotation ──
                # Applied before the dtype check so the early-return fallback
                # (when weight was already dequantized by a LoRA patch) also
                # uses rotated activations.
                if self._rot_need:
                    x_rot = self._rotate(x)
                else:
                    x_rot = x

                # ── Early exit: weight already in float (e.g. after LoRA bake-in) ──
                if weight.dtype != torch.uint8:
                    out = F.linear(x_rot, weight, bias)
                    if need_cast and _MANUAL_CAST_AVAILABLE:
                        uncast_bias_weight(self, weight, bias, offload_stream)
                    return out.to(x.dtype)

                # ── Activation padding ──
                # Packed weight stores 2 INT4 values per byte, so the logical
                # in_features is 2× the packed dimension.
                w_in = self.weight.shape[1] * 2

                # Pad activations if shorter than the logical weight dimension
                if x_rot.shape[-1] < w_in:
                    x_rot = torch.nn.functional.pad(x_rot, (0, w_in - x_rot.shape[-1]))

                # ── PermuQuant channel permutation ──
                if self._perm is not None:
                    x_rot = x_rot[..., self._perm.to(x_rot.device)]

                # ── INT4 GEMM: Triton fused unpack or PyTorch fallback ──
                scale_col = self.weight_scale.to(device=x.device, dtype=torch.float16)

                if not IntCrushInt4Ops._use_pytorch and _TRITON_INT4_UNPACK and weight.dtype == torch.uint8:
                    # Triton unpack to float16 + cuBLAS GEMM — fastest W4A16 path
                    weight_dev = weight.to(device=x.device).contiguous()
                    scale_dev = scale_col.view(-1).contiguous()
                    weight_f16 = unpack_int4_to_float16(weight_dev, scale_dev, w_in)
                    out = F.linear(x_rot, weight_f16.to(x_rot.dtype))
                else:
                    # Pure PyTorch fallback: dequantize weights then F.linear
                    from ._quant_utils import unpack_int4
                    unpacked = unpack_int4(weight, w_in).to(device=x.device)
                    weight_f = (unpacked.float() * scale_col.float()).to(x_rot.dtype)
                    out = F.linear(x_rot, weight_f)

                if bias is not None:
                    out = out + bias.to(device=x.device, dtype=out.dtype)

                if need_cast and _MANUAL_CAST_AVAILABLE:
                    uncast_bias_weight(self, weight, bias, offload_stream)

                return out.to(x.dtype)

            def _rotate(self, x: torch.Tensor) -> torch.Tensor:
                return rotate_activations(x, self._rot_size)

        class GroupNorm(manual_cast.GroupNorm):
            pass

        class LayerNorm(manual_cast.LayerNorm):
            pass

        class Conv2d(manual_cast.Conv2d):
            pass

        class Conv3d(manual_cast.Conv3d):
            pass

        class ConvTranspose2d(manual_cast.ConvTranspose2d):
            pass

        class Embedding(manual_cast.Embedding):
            pass


def _get_diffusion_model_list():
    try:
        import folder_paths
        return folder_paths.get_filename_list("diffusion_models")
    except Exception:
        return []


class SimpleINT4UNetLoader:
    """ComfyUI node: loads a INT-Crush-quantized model with INT4 weights."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (_get_diffusion_model_list(),),
                "model_type": (["flux", "wan", "zimage", "chroma", "default"], {"default": "flux"}),
                "rot_size": ([0, 16, 64, 256], {"default": 256}),
            },
            "optional": {
                "use_pytorch": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "loaders"

    def load(self, unet_name: str, model_type: str, rot_size: int, use_pytorch: bool = True) -> tuple[object]:
        import folder_paths
        import comfy.utils
        from comfy.sd import load_diffusion_model

        if not _COMFY_OPS_AVAILABLE:
            raise RuntimeError("comfy.ops.manual_cast not available. Must run inside ComfyUI.")

        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        if unet_path is None:
            raise FileNotFoundError(f"Model not found: {unet_name}")

        # Auto-detect rot_size from metadata
        try:
            _, metadata = comfy.utils.load_torch_file(unet_path, return_metadata=True)
        except Exception:
            metadata = {}

        if metadata and metadata.get("int_crush.format_version") in ("2", "3"):
            detected = metadata.get("int_crush.rot_size")
            if detected is not None:
                try:
                    detected = int(detected)
                    if detected in (0, 16, 64, 256):
                        rot_size = detected
                        print(f"[INT-Crush] Auto-detected rot_size={rot_size} from metadata")
                except (ValueError, TypeError):
                    pass

            method = metadata.get("int_crush.method", "")
            if "gptq" in method:
                IntCrushInt4Ops._is_gptq = True
                print(f"[INT-Crush] GPTQ model detected")

        IntCrushInt4Ops._rot_size_override = rot_size
        IntCrushInt4Ops._use_pytorch = use_pytorch
        if use_pytorch:
            print("[INT-Crush] Using PyTorch fallback path")
        model_options = {"custom_operations": IntCrushInt4Ops}
        model = load_diffusion_model(unet_path, model_options=model_options)

        if IntCrushInt4Ops._perm_count > 0:
            print(f"[INT-Crush] PermuQuant: {IntCrushInt4Ops._perm_count} layers with channel permutations")
        IntCrushInt4Ops._perm_count = 0

        from .model_patcher import INT4ModelPatcher
        model = INT4ModelPatcher.clone(model)

        return (model,)


NODE_CLASS_MAPPINGS = {
    "SimpleINT4UNetLoader": SimpleINT4UNetLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleINT4UNetLoader": "INT4 UNet Loader (INT-Crush)",
}


def _requantize_int8(w_float):
    """Re-quantize a float weight tensor to INT8 with per-channel scales.

    Used by IntCrushInt8Ops.set_weight() when a LoRA patch produces a
    float32 weight that needs to be stored back in INT8 format.  Computes
    a single per-row scale (max-abs / 127), rounds, and clamps to [-128, 127].

    Args:
        w_float: [out_features, in_features] float weight tensor

    Returns:
        w_int8: [out_features, in_features] int8 quantized weights
        s_w: [out_features] float32 per-row scales
    """
    s_w = w_float.abs().amax(dim=1).to(torch.float32).clamp(min=1e-8) / 127.0
    q = w_float / s_w.unsqueeze(1)
    w_int8 = (torch.where(q >= 0, q + 0.5, q - 0.5)
              .clamp(-128, 127).to(torch.int8))
    return w_int8, s_w


if _COMFY_OPS_AVAILABLE:

    class IntCrushInt8Ops(manual_cast):
        """Custom ComfyUI operations for INT-Crush INT8 quantization.

        Detects INT8 weights in INT-Crush format:
          .weight (int8) + .weight_scale (fp16, [out, 1])

        When rot_size > 0, applies Hadamard rotation to activations to match
        the rotated weight space from conversion.

        Uses fused Triton kernels for W8A8 GEMM with dynamic per-token
        activation quantization. Falls back to PyTorch when Triton is
        unavailable.
        """

        _rot_size_override = 0
        _is_gptq = False
        _debug_shapes = False
        _perm_count = 0

        class Linear(manual_cast.Linear):

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._is_int8 = False
                self._rot_need = True
                self._rot_size = IntCrushInt8Ops._rot_size_override
                self._perm = None
                self.register_buffer("weight_scale", None)
                self.compute_dtype = torch.bfloat16

            def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys, error_msgs):
                """Intercept weight loading to detect INT-Crush INT8 tensors.

                When the state dict contains an int8 weight + float16 weight_scale
                pair, this layer enters INT8 mode.  The scale is reshaped from
                [out, 1] to [out] for compatibility with the Triton GEMM kernel.
                """
                weight_key = prefix + "weight"
                scale_key = prefix + "weight_scale"
                perm_key = prefix + "weight.perm"

                if weight_key in state_dict and scale_key in state_dict:
                    wt = state_dict[weight_key]
                    sc = state_dict[scale_key]

                    if IntCrushInt8Ops._debug_shapes:
                        print(f"[INT-Crush8] LOAD {prefix} wt={wt.dtype}{tuple(wt.shape)} "
                              f"sc={sc.dtype}{tuple(sc.shape)}")

                    if wt.dtype == torch.int8 and sc.dtype == torch.float16:
                        self._is_int8 = True
                        self.weight = nn.Parameter(wt, requires_grad=False)
                        # Reshape [out, 1] -> [out] for the Triton kernel
                        self.register_buffer("weight_scale", sc.float().reshape(-1))

                        if perm_key in state_dict:
                            self._perm = state_dict[perm_key].to(torch.int64)
                            IntCrushInt8Ops._perm_count += 1

                        bias_key = prefix + "bias"
                        if bias_key in state_dict:
                            self.bias = nn.Parameter(state_dict[bias_key], requires_grad=False)

                        # Determine rotation (same logic as INT4)
                        name_lower = prefix.lower()
                        if self._rot_size == 0:
                            self._rot_need = False
                        else:
                            self._rot_need = not any(
                                p in name_lower for p in ["embed", "norm", "modulation", "output", "lm_head", "proj_out"]
                            )

                        self.weight_comfy_model_dtype = torch.int8
                        if self.bias is not None:
                            self.bias_comfy_model_dtype = self.bias.dtype

                        state_dict.pop(weight_key, None)
                        state_dict.pop(scale_key, None)
                        state_dict.pop(perm_key, None)
                        state_dict.pop(bias_key, None)
                        return

                if IntCrushInt8Ops._debug_shapes and weight_key in state_dict:
                    wt = state_dict[weight_key]
                    has_scale = scale_key in state_dict
                    print(f"[INT-Crush8] SKIP {prefix} wt={wt.dtype}{tuple(wt.shape)} "
                          f"has_scale={has_scale} "
                          f"{'(scale wrong dtype)' if has_scale else '(no scale key)'}")

                super()._load_from_state_dict(
                    state_dict, prefix, local_metadata,
                    strict, missing_keys, unexpected_keys, error_msgs
                )

            def _get_weight_scale(self):
                return self.weight_scale

            def convert_weight(self, _weight, inplace=False):
                if not self._is_int8:
                    return _weight
                return self.weight

            def set_weight(self, out_weight, inplace_update=False, seed=0, return_weight=False, **kwargs):
                if not self._is_int8:
                    new_weight = out_weight.to(self.weight.dtype)
                    if return_weight:
                        return new_weight
                    if inplace_update:
                        self.weight.data.copy_(new_weight)
                    else:
                        self.weight = nn.Parameter(new_weight, requires_grad=False)
                    return

                if out_weight.dtype == torch.int8:
                    if return_weight:
                        return out_weight
                    if inplace_update:
                        self.weight.data.copy_(out_weight)
                    else:
                        self.weight = nn.Parameter(out_weight, requires_grad=False)
                    return

                w_int8, s_w = _requantize_int8(out_weight.float())

                if return_weight:
                    return w_int8

                if inplace_update:
                    self.weight.data.copy_(w_int8)
                else:
                    self.weight = nn.Parameter(w_int8, requires_grad=False)

                if hasattr(self, "weight_scale"):
                    self.weight_scale = s_w.to(self.weight.device) if self.weight.device.type != "meta" else s_w

            def set_bias(self, out_bias, inplace_update=False, seed=0, return_weight=False, **kwargs):
                if out_bias is None:
                    return None
                new_bias = out_bias
                if return_weight:
                    return new_bias
                if inplace_update:
                    if self.bias is not None:
                        self.bias.data.copy_(new_bias)
                else:
                    self.bias = nn.Parameter(new_bias, requires_grad=False)

            @torch.no_grad()
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # ── Fallback: non-quantized layers use the standard manual_cast path ──
                if not self._is_int8:
                    if IntCrushInt8Ops._debug_shapes:
                        print(f"[INT-Crush8] NOT_INT8 {self.__class__.__qualname__} "
                              f"x={tuple(x.shape)} w={tuple(self.weight.shape)} "
                              f"w_dtype={self.weight.dtype}")
                    return super().forward(x)

                # ── ComfyUI weight casting (low-VRAM / offload mode) ──
                need_cast = (
                    _MANUAL_CAST_AVAILABLE
                    and (
                        getattr(self, "comfy_cast_weights", False)
                        or len(getattr(self, "weight_function", [])) > 0
                        or len(getattr(self, "bias_function", [])) > 0
                        or getattr(self, "weight_lowvram_function", None) is not None
                        or getattr(self, "bias_lowvram_function", None) is not None
                    )
                )

                if need_cast and _MANUAL_CAST_AVAILABLE:
                    weight, bias, offload_stream = cast_bias_weight(
                        self, input=None, dtype=torch.int8, device=x.device,
                        bias_dtype=x.dtype, offloadable=True
                    )
                else:
                    weight = self.weight
                    bias = self.bias
                    offload_stream = None

                # ── Hadamard rotation ──
                if self._rot_need:
                    x = self._rotate(x)

                # ── Activation padding ──
                # Rotation pads in_features to a multiple of rot_size during
                # conversion, so the stored weight may be wider than the model's
                # original in_features.
                w_in = weight.shape[1] if weight.dtype == torch.int8 else self.weight.shape[1]
                if x.shape[-1] < w_in:
                    pad = w_in - x.shape[-1]
                    x = torch.nn.functional.pad(x, (0, pad))
                    if IntCrushInt8Ops._debug_shapes:
                        print(f"[INT-Crush8] PADDED x {x.shape[-1] - pad} -> {x.shape[-1]} "
                              f"to match weight in_features={w_in}")

                # ── PermuQuant channel permutation ──
                if self._perm is not None:
                    x = x[..., self._perm.to(x.device)]

                # ── Early exit: weight already in float (e.g. after LoRA bake-in) ──
                if weight.dtype != torch.int8:
                    if IntCrushInt8Ops._debug_shapes:
                        print(f"[INT-Crush8] FALLBACK {self.__class__.__qualname__} "
                              f"x={tuple(x.shape)} w={tuple(weight.shape)} "
                              f"w_dtype={weight.dtype} is_int8={self._is_int8}")
                    out = F.linear(x, weight, bias)
                    if need_cast and _MANUAL_CAST_AVAILABLE:
                        uncast_bias_weight(self, weight, bias, offload_stream)
                    return out.to(x.dtype)

                w_scale = self.weight_scale
                if isinstance(w_scale, torch.Tensor) and w_scale.device != x.device:
                    w_scale = w_scale.to(x.device, non_blocking=True)

                compute_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16

                x_shape = x.shape
                x_2d = x.reshape(-1, x_shape[-1])

                if IntCrushInt8Ops._debug_shapes:
                    print(f"[INT-Crush8] {self.__class__.__qualname__} "
                          f"x={tuple(x.shape)} w={tuple(weight.shape)} "
                          f"rot={self._rot_need} scale={tuple(w_scale.shape) if w_scale is not None else None}")

                # Ensure weight & bias are on the compute device before branching
                weight = weight.to(x.device, non_blocking=True)
                if bias is not None:
                    bias = bias.to(device=x.device, dtype=x.dtype)

                # ── W8A8 GEMM: Triton fused kernel or PyTorch fallback ──
                # Small batches (<=16) skip Triton to avoid kernel launch overhead.
                if x_2d.shape[0] <= 16 or not _TRITON_AVAILABLE or not x.is_cuda:
                    w_scale_2d = w_scale.reshape(-1, 1) if w_scale.ndim == 1 else w_scale
                    w_float = (weight.float() * w_scale_2d).to(x.dtype)
                    out = F.linear(x_2d, w_float, bias)
                    del w_float
                else:
                    # Fused INT8 GEMM + dequantization via Triton kernel
                    x_int8, s_a = dynamic_quantize_activation(x_2d)
                    out = fused_int8_gemm_dequant(
                        x_int8, weight, w_scale, s_a,
                        bias=bias, out_dtype=compute_dtype,
                    )

                if need_cast and _MANUAL_CAST_AVAILABLE:
                    uncast_bias_weight(self, weight, bias, offload_stream)

                return out.reshape(*x_shape[:-1], -1)

            def _rotate(self, x: torch.Tensor) -> torch.Tensor:
                return rotate_activations(x, self._rot_size)

        class GroupNorm(manual_cast.GroupNorm):
            pass

        class LayerNorm(manual_cast.LayerNorm):
            pass

        class Conv2d(manual_cast.Conv2d):
            pass

        class Conv3d(manual_cast.Conv3d):
            pass

        class ConvTranspose2d(manual_cast.ConvTranspose2d):
            pass

        class Embedding(manual_cast.Embedding):
            pass


class SimpleINT8UNetLoader:
    """ComfyUI node: loads a INT-Crush-quantized model with INT8 weights."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (_get_diffusion_model_list(),),
                "model_type": (["flux", "wan", "zimage", "chroma", "default"], {"default": "flux"}),
                "rot_size": ([0, 16, 64, 256], {"default": 256}),
            },
            "optional": {
                "debug_shapes": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "loaders"

    def load(self, unet_name: str, model_type: str, rot_size: int, debug_shapes: bool = False) -> tuple[object]:
        import folder_paths
        import comfy.utils
        from comfy.sd import load_diffusion_model

        if not _COMFY_OPS_AVAILABLE:
            raise RuntimeError("comfy.ops.manual_cast not available. Must run inside ComfyUI.")

        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        if unet_path is None:
            raise FileNotFoundError(f"Model not found: {unet_name}")

        # Auto-detect rot_size from metadata
        try:
            _, metadata = comfy.utils.load_torch_file(unet_path, return_metadata=True)
        except Exception:
            metadata = {}

        if metadata and metadata.get("int_crush.format_version") == "1":
            detected = metadata.get("int_crush.rot_size")
            if detected is not None:
                try:
                    detected = int(detected)
                    if detected in (0, 16, 64, 256):
                        rot_size = detected
                        print(f"[INT-Crush] INT8: Auto-detected rot_size={rot_size} from metadata")
                except (ValueError, TypeError):
                    pass

            method = metadata.get("int_crush.method", "")
            if "gptq" in method:
                IntCrushInt8Ops._is_gptq = True
                print(f"[INT-Crush] INT8: GPTQ model detected")
            elif "ldlq" in method:
                print(f"[INT-Crush] INT8: LDLQ model detected")

        IntCrushInt8Ops._rot_size_override = rot_size
        IntCrushInt8Ops._debug_shapes = debug_shapes
        model_options = {"custom_operations": IntCrushInt8Ops}
        model = load_diffusion_model(unet_path, model_options=model_options)

        if IntCrushInt8Ops._perm_count > 0:
            print(f"[INT-Crush] INT8: PermuQuant: {IntCrushInt8Ops._perm_count} layers with channel permutations")
        IntCrushInt8Ops._perm_count = 0

        # Fix model config for padded layers.
        # ComfyUI reads in_channels from img_in.weight.shape[1], but rotation
        # padding inflates that dimension. We stored the originals in metadata.
        padded_str = (metadata or {}).get("int_crush.padded_layers", "")
        if padded_str:
            padded_map = {}
            for entry in padded_str.split(";"):
                if "=" in entry:
                    k, v = entry.split("=", 1)
                    padded_map[k.strip()] = int(v.strip())

            m = model.model
            for layer_key, orig_in in padded_map.items():
                # layer_key is the state dict key, e.g. "img_in.weight"
                # module path is everything before the last ".weight"
                module_path = layer_key.rsplit(".weight", 1)[0]
                try:
                    module = comfy.utils.get_attr(m, module_path)
                except AttributeError:
                    continue

                if hasattr(module, 'weight') and module.weight is not None:
                    padded_in = module.weight.shape[1]
                    if padded_in > orig_in:
                        module.weight = torch.nn.Parameter(
                            module.weight[:, :orig_in].contiguous(),
                            requires_grad=False,
                        )
                        print(f"[INT-Crush] INT8: Fixed {layer_key} "
                              f"in_features {padded_in} -> {orig_in}")

                # Fix parent model's in_channels if this is img_in
                if module_path == "img_in" and hasattr(m, 'in_channels'):
                    ps = getattr(m, 'patch_size', 1)
                    m.in_channels = orig_in // (ps * ps) if ps else orig_in
                    print(f"[INT-Crush] INT8: Fixed model.in_channels = {m.in_channels}")

        from .model_patcher import INT8ModelPatcher
        model = INT8ModelPatcher.clone(model)

        if debug_shapes:
            print("[INT-Crush] INT8: Shape debugging enabled — forward() will print x/weight shapes")
            m = model.model
            if hasattr(m, 'params'):
                p = m.params
                print(f"[INT-Crush] Model params: in_channels={p.in_channels} "
                      f"patch_size={p.patch_size} hidden_size={p.hidden_size}")
                feat_dim = p.in_channels * p.patch_size * p.patch_size
                print(f"[INT-Crush] Expected patch feature dim: {p.in_channels} * {p.patch_size} * {p.patch_size} = {feat_dim}")
                if hasattr(m, 'img_in'):
                    w = m.img_in.weight if hasattr(m.img_in, 'weight') else None
                    if w is not None:
                        print(f"[INT-Crush] img_in weight: {tuple(w.shape)} (in_features={w.shape[1]})")
                        if w.shape[1] != feat_dim:
                            print(f"[INT-Crush] WARNING: img_in.in_features ({w.shape[1]}) != patch feature dim ({feat_dim})")
            # Count int8 layers
            int8_count = 0
            total_count = 0
            for name, module in m.named_modules():
                if hasattr(module, '_is_int8'):
                    total_count += 1
                    if module._is_int8:
                        int8_count += 1
            print(f"[INT-Crush] INT8 layers: {int8_count}/{total_count}")

        return (model,)


NODE_CLASS_MAPPINGS["SimpleINT8UNetLoader"] = SimpleINT8UNetLoader
NODE_DISPLAY_NAME_MAPPINGS["SimpleINT8UNetLoader"] = "INT8 UNet Loader (INT-Crush)"
