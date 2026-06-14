"""ComfyUI nodes for INT4/INT8 model loading with INT-Crush.

Provides:
  - SimpleINT4UNetLoader: loads a quantized model with INT4 weights in memory
  - SimpleINT8UNetLoader: loads a quantized model with INT8 weights in memory
"""

import time
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
_TRITON_INT8_GEMM = False
_TRITON_INT4_INT8_UNPACK = False
_TRITON_DYNQUANT = False
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
    from .kernels.triton_quantize import dynamic_quantize_activation
    _TRITON_DYNQUANT = True
except Exception:
    pass

try:
    from .kernels.triton_int8_gemm import fused_int8_gemm_dequant
    _TRITON_INT8_GEMM = True
except Exception:
    pass

_HAS_FUSED_QUANT_GEMM = False
try:
    from .kernels.triton_int8_gemm import fused_quant_int8_gemm_dequant
    _HAS_FUSED_QUANT_GEMM = True
except Exception:
    pass

try:
    from .kernels.triton_int4_to_int8_unpack import unpack_int4_to_int8
    _TRITON_INT4_INT8_UNPACK = True
except Exception:
    pass

_TRITON_FHT = False

if _TRITON_INT8_GEMM and _TRITON_DYNQUANT:
    _TRITON_AVAILABLE = True
    print("[INT-Crush] Triton INT8 GEMM + dynamic quantizer loaded")
if _TRITON_INT4_INT8_UNPACK:
    print("[INT-Crush] Triton INT4->INT8 unpack kernel loaded")

try:
    import triton.backends.nvidia.driver as _triton_nvidia

    _orig_get_cache = _triton_nvidia.CUDADriver.get_empty_cache_for_benchmark

    def _get_cache_small(self):
        try:
            free_mem, _ = torch.cuda.mem_get_info()
            budget = min(32 * 1024 * 1024, free_mem // 8)
        except Exception:
            budget = 32 * 1024 * 1024
        if budget < 4 * 1024 * 1024:
            return torch.empty(0, dtype=torch.int, device="cuda")
        return torch.empty(budget // 4, dtype=torch.int, device="cuda")

    _triton_nvidia.CUDADriver.get_empty_cache_for_benchmark = _get_cache_small
except Exception:
    pass

try:
    from comfy.ops import manual_cast
    _COMFY_OPS_AVAILABLE = True
except ImportError:
    _COMFY_OPS_AVAILABLE = False

try:
    from comfy.ops import cast_bias_weight, uncast_bias_weight
    _MANUAL_CAST_AVAILABLE = True
except ImportError:
    try:
        from comfy.ops.manual_cast import cast_bias_weight, uncast_bias_weight
        _MANUAL_CAST_AVAILABLE = True
    except ImportError:
        _MANUAL_CAST_AVAILABLE = False


def _requantize_int4(w_float, rot_need, rot_size):
    """Re-quantize a float weight to packed INT4 with per-row scales.

    Used by set_weight() when LoRA or other op produces a float weight.
    Optionally applies rotation, then rounds + packs two INT4 per uint8.

    Args:
        w_float: [out_features, in_features] float weight
        rot_need: whether to apply Hadamard rotation
        rot_size: Hadamard group size (16, 64, 256, 1024, or 4096)

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


# ── Profiling helper ─────────────────────────────────────────────────────────

class _LayerProfiler:
    """Accumulates per-section timings across forward calls and prints a
    summary every ``every`` calls.  Uses CUDA events for accurate GPU
    timing without forcing full-device synchronisation."""

    enabled = False  # Set True only for profiling; kills async overlap

    def __init__(self, name, every=50):
        self.name = name
        self.every = every
        self.sections: dict[str, list[float]] = {}
        self.call_count = 0
        self._starts: dict[str, torch.cuda.Event] = {}
        self._ends: dict[str, torch.cuda.Event] = {}

    def start(self, section: str):
        if not self.enabled:
            return
        evt = torch.cuda.Event(enable_timing=True)
        evt.record()
        self._starts[section] = evt

    def end(self, section: str):
        if not self.enabled:
            return
        evt = torch.cuda.Event(enable_timing=True)
        evt.record()
        self._ends[section] = evt

    def finish_call(self):
        if not self.enabled:
            return
        torch.cuda.synchronize()
        for section, end_evt in self._ends.items():
            start_evt = self._starts[section]
            dt = start_evt.elapsed_time(end_evt) / 1000.0  # ms → s
            self.sections.setdefault(section, []).append(dt)
        self._starts.clear()
        self._ends.clear()
        self.call_count += 1
        if self.call_count % self.every == 0:
            self._print_summary()

    def _print_summary(self):
        lines = [f"\n{'='*60}", f"  {self.name} — after {self.call_count} calls", f"{'='*60}"]
        total_avg = 0.0
        for sec, times in self.sections.items():
            avg = sum(times[-self.every:]) / len(times[-self.every:])
            total_avg += avg
            lines.append(f"  {sec:>30s}: {avg*1000:>8.3f} ms")
        lines.append(f"  {'TOTAL':>30s}: {total_avg*1000:>8.3f} ms")
        lines.append(f"{'='*60}")
        print("\n".join(lines))


_int4_profiler = _LayerProfiler("IntCrushInt4Ops.Linear", every=50)
_int8_profiler = _LayerProfiler("IntCrushInt8Ops.Linear", every=50)


if _COMFY_OPS_AVAILABLE:

    class IntCrushInt4Ops(manual_cast):
        """Custom ComfyUI operations for INT-Crush INT4 quantization."""

        _rot_size_override = 0
        _is_gptq = False
        _use_w4a8 = True  # Default to W4A8 IMMA path
        _perm_count = 0

        class Linear(manual_cast.Linear):

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._is_int4 = False
                self._rot_need = True
                self._rot_size = IntCrushInt4Ops._rot_size_override
                self._perm = None  # PermuQuant permutation indices
                self.register_buffer("weight_scale", None)
                self.register_buffer("weight_zp", None)
                self.weight_comfy_model_dtype = getattr(self.weight, "dtype", None)
                self.bias_comfy_model_dtype = getattr(self.bias, "dtype", None)

            def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys, error_msgs):
                """Intercept loading to detect INT-Crush INT4 tensors (uint8 weight + fp16 scale).

                Enters INT4 mode when detected: stores raw packed weights/scales directly,
                sets rotation flag from layer name, and loads optional PermuQuant permutation.
                """
                weight_key = prefix + "weight"
                scale_key = prefix + "weight_scale"
                zp_key = prefix + "weight_zp"
                perm_key = prefix + "weight.perm"

                if weight_key in state_dict and scale_key in state_dict:
                    wt = state_dict[weight_key]
                    sc = state_dict[scale_key]

                    if wt.dtype == torch.uint8 and sc.dtype == torch.float16:
                        self._is_int4 = True
                        self.weight = nn.Parameter(wt, requires_grad=False)
                        self.register_buffer("weight_scale", sc)

                        # Load zero-point if present (asymmetric quantization)
                        if zp_key in state_dict:
                            self.register_buffer("weight_zp", state_dict[zp_key])

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
                        state_dict.pop(zp_key, None)
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
                p = _int4_profiler
                # ── Fallback: non-quantized layers ──
                if not self._is_int4:
                    return super().forward(x)

                # ── ComfyUI weight casting (low-VRAM / offload mode) ──
                p.start("cast")
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
                p.end("cast")

                # ── Hadamard rotation ──
                # Only skip Python rotation when the FHT Triton kernel will
                # actually be taken (requires batch > 16, all Triton kernels
                # available, weight still packed uint8, etc.).
                _pre_batch = x.reshape(-1, x.shape[-1]).shape[0]
                use_fht = (
                    self._rot_need and _TRITON_FHT
                    and self._rot_size >= 256 and x.is_cuda
                    and IntCrushInt4Ops._use_w4a8
                    and _pre_batch > 16
                    and _TRITON_INT8_GEMM
                    and _TRITON_DYNQUANT
                    and _TRITON_INT4_INT8_UNPACK
                    and weight.dtype == torch.uint8
                )
                p.start("rotation")
                if self._rot_need and not use_fht:
                    x_rot = self._rotate(x)
                else:
                    x_rot = x
                p.end("rotation")

                # ── Early exit: weight already in float (e.g. after LoRA bake-in) ──
                if weight.dtype != torch.uint8:
                    out = F.linear(x_rot, weight, bias)
                    if need_cast and _MANUAL_CAST_AVAILABLE:
                        uncast_bias_weight(self, weight, bias, offload_stream)
                    p.finish_call()
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

                # ── INT4 GEMM: W4A8 / Triton W4A16 / PyTorch fallback ──
                p.start("scale_xfer")
                scale_col = self.weight_scale
                if scale_col.device != x.device:
                    scale_col = scale_col.to(x.device)
                # Convert to fp32 for GEMM kernel precision (INT8 path does this at load time)
                scale_flat = scale_col.reshape(-1).float().contiguous()
                x_2d = x_rot.reshape(-1, x_rot.shape[-1])
                batch = x_2d.shape[0]

                # Zero-point correction (recomputed each call — tiny tensor)
                has_zp = self.weight_zp is not None
                if has_zp:
                    zp_cor = (scale_col.reshape(-1).float() * self.weight_zp.to(x.device).reshape(-1).float()).to(torch.float16)
                p.end("scale_xfer")

                if (IntCrushInt4Ops._use_w4a8
                        and self._rot_need
                        and batch > 16
                        and _TRITON_INT8_GEMM
                        and _TRITON_DYNQUANT
                        and _TRITON_INT4_INT8_UNPACK
                        and weight.dtype == torch.uint8
                        and x.is_cuda):
                    # W4A8: INT8 act quant + INT4→INT8 unpack + Triton INT8 GEMM
                    p.start("quantize")
                    if use_fht:
                        x_int8, s_a = fht_quantize_activation(x_2d, self._rot_size)
                    else:
                        x_int8, s_a = dynamic_quantize_activation(x_2d)
                    p.end("quantize")
                    p.start("unpack")
                    weight_dev = weight
                    if not weight_dev.is_contiguous():
                        weight_dev = weight_dev.contiguous()
                    w_int8 = unpack_int4_to_int8(weight_dev, w_in)
                    p.end("unpack")
                    p.start("gemm")
                    out = fused_int8_gemm_dequant(
                        x_int8, w_int8, scale_flat, s_a,
                        bias=bias, out_dtype=x.dtype,
                    )
                    out = out.reshape(*x_rot.shape[:-1], -1)
                    del x_int8, w_int8, s_a
                    p.end("gemm")
                    if has_zp:
                        p.start("zp_cor")
                        x_sum = x_rot.sum(dim=-1, keepdim=True)
                        out = out - x_sum * zp_cor
                        p.end("zp_cor")
                    p.start("uncast")
                    if need_cast and _MANUAL_CAST_AVAILABLE:
                        uncast_bias_weight(self, weight, bias, offload_stream)
                    p.end("uncast")
                    p.finish_call()
                    return out.to(x.dtype)
                elif not IntCrushInt4Ops._use_pytorch and _TRITON_INT4_UNPACK and weight.dtype == torch.uint8:
                    # Triton unpack to float16 + cuBLAS GEMM — fastest W4A16 path
                    p.start("unpack")
                    weight_dev = weight
                    if not weight_dev.is_contiguous():
                        weight_dev = weight_dev.contiguous()
                    weight_f16 = unpack_int4_to_float16(weight_dev, scale_flat.contiguous(), w_in)
                    p.end("unpack")
                    p.start("gemm")
                    out = F.linear(x_rot, weight_f16.to(x_rot.dtype))
                    del weight_f16
                    p.end("gemm")
                    if has_zp:
                        p.start("zp_cor")
                        x_sum = x_rot.sum(dim=-1, keepdim=True)
                        out = out - x_sum * zp_cor
                        p.end("zp_cor")
                else:
                    # Pure PyTorch fallback: dequantize weights then F.linear
                    p.start("dequant_pytorch")
                    from ._quant_utils import unpack_int4
                    unpacked = unpack_int4(weight, w_in).to(device=x.device)
                    if has_zp:
                        zp_col = self.weight_zp.to(device=x.device, dtype=torch.float16)
                        weight_f = (unpacked.to(x_rot.dtype) - zp_col.reshape(-1, 1)) * scale_col
                    else:
                        weight_f = unpacked.to(x_rot.dtype) * scale_col
                    p.end("dequant_pytorch")
                    p.start("gemm")
                    out = F.linear(x_rot, weight_f)
                    p.end("gemm")

                if bias is not None:
                    out = out + bias.to(device=x.device, dtype=out.dtype)

                p.start("uncast")
                if need_cast and _MANUAL_CAST_AVAILABLE:
                    uncast_bias_weight(self, weight, bias, offload_stream)
                p.end("uncast")

                p.finish_call()
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
                "rot_size": ([0, 16, 64, 256, 1024, 4096], {"default": 256}),
            },
            "optional": {
                "use_pytorch": ("BOOLEAN", {"default": False}),
                "use_w4a16": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "loaders"

    def load(self, unet_name: str, model_type: str, rot_size: int, use_pytorch: bool = False, use_w4a16: bool = False) -> tuple[object]:
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
                    if detected in (0, 16, 64, 256, 1024, 4096):
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
        IntCrushInt4Ops._use_w4a8 = not use_w4a16
        if use_w4a16:
            print("[INT-Crush] Using W4A16 path")
            if use_pytorch:
                print("[INT-Crush] Using PyTorch fallback for W4A16")
        elif rot_size == 0:
            print("[INT-Crush] WARNING: W4A8 requires rotation — falling back to W4A16")
            IntCrushInt4Ops._use_w4a8 = False
        else:
            print("[INT-Crush] Using W4A8 path (default, fastest)")
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
    """Re-quantize a float weight to INT8 with per-row scales.

    Used by IntCrushInt8Ops.set_weight() when a LoRA patch produces a float weight.
    Per-row scale: max-abs / 127, round, clamp to [-128, 127].

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

        Detects INT8 weights (.weight int8 + .weight_scale fp16), applies Hadamard
        rotation when rot_size > 0, and uses fused Triton W8A8 GEMM kernels.
        """

        _rot_size_override = 0
        _is_gptq = False
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
                p = _int8_profiler
                # ── Fallback: non-quantized layers ──
                if not self._is_int8:
                    return super().forward(x)

                # ── ComfyUI weight casting (low-VRAM / offload mode) ──
                p.start("cast")
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
                p.end("cast")

                # ── Hadamard rotation ──
                # Only skip Python rotation when FHT kernel will be taken.
                _pre_batch = x.reshape(-1, x.shape[-1]).shape[0]
                use_fht = (
                    self._rot_need and _TRITON_FHT
                    and self._rot_size >= 256 and x.is_cuda
                    and _pre_batch > 16
                    and _TRITON_AVAILABLE
                )
                p.start("rotation")
                if self._rot_need and not use_fht:
                    x = self._rotate(x)
                p.end("rotation")

                # ── Activation padding ──
                # Weight may be wider than model's in_features due to rotation padding.
                w_in = weight.shape[1] if weight.dtype == torch.int8 else self.weight.shape[1]
                if x.shape[-1] < w_in:
                    pad = w_in - x.shape[-1]
                    x = torch.nn.functional.pad(x, (0, pad))

                # ── PermuQuant channel permutation ──
                if self._perm is not None:
                    x = x[..., self._perm.to(x.device)]

                # ── Early exit: weight already in float (e.g. after LoRA bake-in) ──
                if weight.dtype != torch.int8:
                    out = F.linear(x, weight, bias)
                    if need_cast and _MANUAL_CAST_AVAILABLE:
                        uncast_bias_weight(self, weight, bias, offload_stream)
                    p.finish_call()
                    return out.to(x.dtype)

                p.start("scale_xfer")
                w_scale = self.weight_scale
                if isinstance(w_scale, torch.Tensor) and w_scale.device != x.device:
                    w_scale = w_scale.to(x.device, non_blocking=True)
                p.end("scale_xfer")

                compute_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16

                x_shape = x.shape
                x_2d = x.reshape(-1, x_shape[-1])
                batch = x_2d.shape[0]

                # ── W8A8 GEMM: Triton fused kernel or PyTorch fallback ──
                if batch <= 16 or not _TRITON_AVAILABLE or not x.is_cuda:
                    # Small batches: PyTorch fallback
                    p.start("dequant_pytorch")
                    w_scale_2d = w_scale.reshape(-1, 1) if w_scale.ndim == 1 else w_scale
                    w_float = weight.to(compute_dtype) * w_scale_2d.to(compute_dtype)
                    p.end("dequant_pytorch")
                    p.start("gemm")
                    out = F.linear(x_2d, w_float, bias)
                    del w_float
                    p.end("gemm")
                elif use_fht:
                    # FHT: fused rotation + quantize (O(N log N) for large rot_sizes)
                    p.start("fht_quantize")
                    x_int8, s_a = fht_quantize_activation(x_2d, self._rot_size)
                    p.end("fht_quantize")
                    p.start("gemm")
                    out = fused_int8_gemm_dequant(
                        x_int8, weight, w_scale, s_a,
                        bias=bias, out_dtype=compute_dtype,
                    )
                    del x_int8, s_a
                    p.end("gemm")
                elif (batch <= 32 or (batch <= 128 and x_2d.shape[1] <= 4096)
                      ) and _HAS_FUSED_QUANT_GEMM and _TRITON_DYNQUANT:
                    # Small M, moderate K: fused quantize+GEMM+dequant
                    p.start("fused_quant_gemm")
                    out = fused_quant_int8_gemm_dequant(
                        x_2d, weight, w_scale,
                        bias=bias, out_dtype=compute_dtype,
                    )
                    p.end("fused_quant_gemm")
                else:
                    # Large M: two-kernel path
                    p.start("quantize")
                    x_int8, s_a = dynamic_quantize_activation(x_2d)
                    p.end("quantize")
                    p.start("gemm")
                    out = fused_int8_gemm_dequant(
                        x_int8, weight, w_scale, s_a,
                        bias=bias, out_dtype=compute_dtype,
                    )
                    del x_int8, s_a
                    p.end("gemm")

                p.start("uncast")
                if need_cast and _MANUAL_CAST_AVAILABLE:
                    uncast_bias_weight(self, weight, bias, offload_stream)
                p.end("uncast")

                p.finish_call()
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
                "rot_size": ([0, 16, 64, 256, 1024, 4096], {"default": 256}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "loaders"

    def load(self, unet_name: str, model_type: str, rot_size: int) -> tuple[object]:
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
                    if detected in (0, 16, 64, 256, 1024, 4096):
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

        return (model,)


NODE_CLASS_MAPPINGS["SimpleINT8UNetLoader"] = SimpleINT8UNetLoader
NODE_DISPLAY_NAME_MAPPINGS["SimpleINT8UNetLoader"] = "INT8 UNet Loader (INT-Crush)"
