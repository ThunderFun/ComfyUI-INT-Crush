"""ComfyUI loader nodes for INT-Crush quantized models.

Uses QuantizedTensor weights for VBAR demand-paging and memory management,
but routes forward() directly to Triton kernels (or PyTorch fallback)
without the mixed_precision_ops overhead.

Provides two ComfyUI nodes:
  - SimpleINT4UNetLoader: loads a model quantized with INT4 weights
  - SimpleINT8UNetLoader: loads a model quantized with INT8 weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._quant_utils import rotate_activations
from .quant_layout import IntCrushInt4Layout, IntCrushInt8Layout

# ── Lazy ComfyUI imports (cached on first call) ─────────────────────────────

_QuantizedTensor = None
_cast_bias_weight = None
_uncast_bias_weight = None

def _get_qt():
    global _QuantizedTensor
    if _QuantizedTensor is None:
        from comfy.quant_ops import QuantizedTensor
        _QuantizedTensor = QuantizedTensor
    return _QuantizedTensor

def _get_cast():
    global _cast_bias_weight, _uncast_bias_weight
    if _cast_bias_weight is None:
        from comfy.ops import cast_bias_weight, uncast_bias_weight
        _cast_bias_weight = cast_bias_weight
        _uncast_bias_weight = uncast_bias_weight
    return _cast_bias_weight, _uncast_bias_weight

# ── Triton kernel availability flags ──────────────────────────────────────────

_TRITON_AVAILABLE = False
_TRITON_INT4_UNPACK = False
_TRITON_W4A16_GEMM = False
_TRITON_W4A8_GEMM = False
_TRITON_INT8_GEMM = False
_TRITON_INT4_INT8_UNPACK = False
_TRITON_DYNQUANT = False
_HAS_FUSED_QUANT_GEMM = False

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
    from .kernels.triton_w4a8_gemm import fused_w4a8_gemm_dequant
    _TRITON_W4A8_GEMM = True
except Exception:
    pass

try:
    from .kernels.triton_int8_gemm import fused_int8_gemm_dequant
    _TRITON_INT8_GEMM = True
except Exception:
    pass

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

if _TRITON_W4A8_GEMM and _TRITON_DYNQUANT and _TRITON_INT4_INT8_UNPACK:
    print("[INT-Crush] Triton W4A8 GEMM + dynamic quantizer + INT4 unpack loaded")
if _TRITON_INT8_GEMM and _TRITON_DYNQUANT:
    _TRITON_AVAILABLE = True
    if not _TRITON_W4A8_GEMM:
        print("[INT-Crush] Triton INT8 GEMM + dynamic quantizer loaded (W4A8 will use W8A8 kernel)")
if _TRITON_INT4_INT8_UNPACK:
    print("[INT-Crush] Triton INT4->INT8 unpack kernel loaded")

# Shrink Triton autotune cache to avoid VRAM bloat from benchmark allocations.
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


# ── Module-level config (set by loader nodes before model load) ──────────────

_use_pytorch = False
_use_w4a16 = False


# ── Ops factory ──────────────────────────────────────────────────────────────

def _make_intcrush_ops(quant_format, rot_size):
    """Build a manual_cast ops class with lean INT-Crush forward.

    Returns QuantizedTensor weights for VBAR/memory management, but the
    forward() bypasses mixed_precision_ops overhead — going directly from
    packed weights to Triton kernels (or PyTorch fallback) without the
    per-input QuantizedTensor.from_float / cast_bias_weight round-trip.
    """
    from comfy.ops import manual_cast

    class IntCrushOps(manual_cast):

        class Linear(manual_cast.Linear):
            """INT-Crush Linear: QuantizedTensor storage with lean forward()."""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._intcrush_is_quantized = False
                self._intcrush_rot_need = True
                self._intcrush_rot_size = rot_size
                self._intcrush_perm = None
                self._intcrush_smooth = None
                self.quant_format = None
                self.layout_type = None

            def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys, error_msgs):
                from comfy.quant_ops import QuantizedTensor, get_layout_class, QUANT_ALGOS

                weight_key = prefix + "weight"
                scale_key = prefix + "weight_scale"
                zp_key = prefix + "weight_zp"
                perm_key = prefix + "weight_perm"
                smooth_key = prefix + "weight_smooth"
                quant_key = prefix + "comfy_quant"

                weight = state_dict.get(weight_key)
                scale = state_dict.get(scale_key)

                # Detect INT-Crush format by dtype signature:
                #   INT4: weight=uint8, scale=float16
                #   INT8: weight=int8,  scale=float16
                is_int4 = (weight is not None and scale is not None
                           and weight.dtype == torch.uint8 and scale.dtype == torch.float16)
                is_int8 = (weight is not None and scale is not None
                           and weight.dtype == torch.int8 and scale.dtype == torch.float16)

                if not (is_int4 or is_int8):
                    # Not an INT-Crush layer — delegate to default loader.
                    super()._load_from_state_dict(
                        state_dict, prefix, local_metadata,
                        strict, missing_keys, unexpected_keys, error_msgs)
                    return

                # ── INT-Crush layer: wrap as QuantizedTensor ──
                fmt = "int4_crush" if is_int4 else "int8_crush"
                qconfig = QUANT_ALGOS[fmt]
                layout_type = qconfig["comfy_tensor_layout"]
                layout_cls = get_layout_class(layout_type)

                # Pop all tensors from state_dict.
                weight = state_dict.pop(weight_key)
                scale = state_dict.pop(scale_key)
                zp = state_dict.pop(zp_key, None)
                perm = state_dict.pop(perm_key, None)
                smooth = state_dict.pop(smooth_key, None)
                bias_tensor = state_dict.pop(prefix + "bias", None)
                state_dict.pop(quant_key, None)

                device = getattr(self, 'weight', None)
                device = device.device if device is not None else torch.device("cpu")

                # Determine whether this layer needs Hadamard rotation based on name.
                name_lower = prefix.lower()
                rot_need = (
                    rot_size > 0 and not any(
                        p in name_lower for p in
                        ["embed", "norm", "modulation", "output", "lm_head", "proj_out"]
                    )
                )
                self._intcrush_rot_need = rot_need
                self._intcrush_rot_size = rot_size
                self._intcrush_is_quantized = True

                if is_int4:
                    params = layout_cls.Params(
                        scale=scale.float().to(device=device),
                        orig_dtype=torch.float16,
                        orig_shape=(self.out_features, self.in_features),
                        rot_need=rot_need,
                        rot_size=rot_size,
                        perm=perm.to(device=device) if perm is not None else None,
                        zp=zp.to(device=device) if zp is not None else None,
                    )
                else:
                    params = layout_cls.Params(
                        scale=scale.float().reshape(-1).to(device=device),
                        orig_dtype=torch.float16,
                        orig_shape=(self.out_features, self.in_features),
                        rot_need=rot_need,
                        rot_size=rot_size,
                        perm=perm.to(device=device) if perm is not None else None,
                    )

                self._intcrush_perm = params.perm
                self._intcrush_w_in = self.in_features
                self._intcrush_smooth = (
                    smooth.float().to(device=device) if smooth is not None else None
                )

                self.weight = nn.Parameter(
                    QuantizedTensor(
                        weight.to(device=device, dtype=qconfig["storage_t"]),
                        layout_type, params,
                    ),
                    requires_grad=False,
                )
                self.quant_format = fmt
                self.layout_type = layout_type

                if bias_tensor is not None:
                    self.bias = nn.Parameter(
                        bias_tensor.to(device=device), requires_grad=False,
                    )
                else:
                    self.bias = None

                if perm is not None:
                    self.register_parameter("weight_perm",
                        nn.Parameter(perm.to(device=device), requires_grad=False))
                if zp is not None:
                    self.register_parameter("weight_zp",
                        nn.Parameter(zp.to(device=device), requires_grad=False))

                for k in (weight_key, scale_key, zp_key, perm_key, smooth_key,
                          prefix + "bias", quant_key):
                    if k in missing_keys:
                        missing_keys.remove(k)

            def convert_weight(self, weight, inplace=False):
                """Keep QuantizedTensor as-is for ModelPatcher compatibility."""
                from comfy.quant_ops import QuantizedTensor
                if self._intcrush_is_quantized and isinstance(weight, QuantizedTensor):
                    return weight
                return weight

            def set_weight(self, out_weight, inplace_update=False, seed=0,
                           return_weight=False, **kwargs):
                """Requantize float weights, preserving rotation and permutation params."""
                if not self._intcrush_is_quantized:
                    new_weight = out_weight.to(self.weight.dtype)
                    if return_weight:
                        return new_weight
                    self.weight = nn.Parameter(new_weight, requires_grad=False)
                    return

                from comfy.quant_ops import QuantizedTensor, get_layout_class
                import dataclasses

                if isinstance(out_weight, QuantizedTensor):
                    if return_weight:
                        return out_weight
                    self.weight = nn.Parameter(out_weight, requires_grad=False)
                    return

                layout_cls = get_layout_class(self.layout_type)
                packed, params = layout_cls.quantize(
                    out_weight.float(), scale="recalculate",
                    stochastic_rounding=seed,
                )
                params = dataclasses.replace(
                    params,
                    rot_need=self._intcrush_rot_need,
                    rot_size=self._intcrush_rot_size,
                    perm=self._intcrush_perm,
                )
                qt = QuantizedTensor(packed, self.layout_type, params)
                if return_weight:
                    return qt
                self.weight = nn.Parameter(qt, requires_grad=False)

            def set_bias(self, out_bias, inplace_update=False, seed=0,
                         return_weight=False, **kwargs):
                if out_bias is None:
                    return None
                if return_weight:
                    return out_bias
                self.bias = nn.Parameter(out_bias, requires_grad=False)

            @torch.no_grad()
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # ── Non-quantized fallback ──
                if not self._intcrush_is_quantized:
                    return super().forward(x)

                # ── ComfyUI weight casting (low-VRAM / offload) ──
                # Check once: most layers have all False → short-circuit fast.
                wfn = self.weight_function
                bfn = self.bias_function
                need_cast = (
                    self.comfy_cast_weights
                    or wfn or bfn
                    or self.weight_lowvram_function is not None
                    or self.bias_lowvram_function is not None
                )

                if need_cast:
                    _cbw, _ucbw = _get_cast()
                    weight, bias, offload_stream = _cbw(
                        self, input=None, dtype=torch.float16,
                        device=x.device, bias_dtype=x.dtype, offloadable=True,
                    )
                else:
                    weight = self.weight
                    bias = self.bias

                QT = _get_qt()

                # Weight already dequantized (e.g. after LoRA).
                if not isinstance(weight, QT):
                    x_2d = x.reshape(-1, x.shape[-1])
                    if self._intcrush_rot_need:
                        x_2d = rotate_activations(x_2d, self._intcrush_rot_size)
                    out = F.linear(x_2d, weight, bias)
                    if need_cast:
                        _ucbw = _get_cast()[1]
                        _ucbw(self, weight, bias, offload_stream)
                    return out.to(x.dtype).reshape(*x.shape[:-1], -1)

                qdata = weight._qdata
                w_in = self._intcrush_w_in
                perm = self._intcrush_perm

                # ── INT4 forward paths ──
                if isinstance(weight._params, IntCrushInt4Layout.Params):
                    # Path 1: W4A8 — unpack INT4→INT8, dynamic-quantize activations,
                    # fused INT8 GEMM + dequant (fastest when all Triton kernels available).
                    _have_w4a8_kernel = (
                        (_TRITON_W4A8_GEMM or _TRITON_INT8_GEMM)
                        and _TRITON_DYNQUANT
                        and _TRITON_INT4_INT8_UNPACK
                    )
                    if (not _use_pytorch and not _use_w4a16
                            and self._intcrush_rot_need
                            and _have_w4a8_kernel):

                        x_2d = x.reshape(-1, x.shape[-1])
                        x_2d = rotate_activations(x_2d, self._intcrush_rot_size)

                        if x_2d.shape[-1] < w_in:
                            x_2d = F.pad(x_2d, (0, w_in - x_2d.shape[-1]))
                        if perm is not None:
                            x_2d = x_2d[..., perm]

                        x_int8, s_a = dynamic_quantize_activation(x_2d)
                        w_int8 = unpack_int4_to_int8(qdata, w_in)
                        scale_flat = weight._params.scale.reshape(-1).float().contiguous()
                        _gemm_fn = fused_w4a8_gemm_dequant if _TRITON_W4A8_GEMM else fused_int8_gemm_dequant
                        out = _gemm_fn(
                            x_int8, w_int8, scale_flat, s_a,
                            bias=bias, out_dtype=x.dtype,
                        )
                        if weight._params.zp is not None:
                            zp_cor = (scale_flat * weight._params.zp.reshape(-1).float()).to(torch.float16)
                            out = out - x_2d.sum(dim=-1, keepdim=True) * zp_cor

                        if need_cast:
                            _get_cast()[1](self, weight, bias, offload_stream)
                        return out.to(x.dtype).reshape(*x.shape[:-1], -1)

                    # Path 2: W4A16 — Triton unpack to float16, then cuBLAS GEMM.
                    if not _use_pytorch and _TRITON_INT4_UNPACK:

                        x_2d = x.reshape(-1, x.shape[-1])
                        if self._intcrush_rot_need:
                            x_2d = rotate_activations(x_2d, self._intcrush_rot_size)

                        if x_2d.shape[-1] < w_in:
                            x_2d = F.pad(x_2d, (0, w_in - x_2d.shape[-1]))
                        if perm is not None:
                            x_2d = x_2d[..., perm]

                        scale_flat = weight._params.scale.reshape(-1).float().contiguous()
                        weight_f16 = unpack_int4_to_float16(qdata, scale_flat, w_in)
                        out = F.linear(x_2d, weight_f16)
                        if weight._params.zp is not None:
                            zp_cor = (scale_flat * weight._params.zp.reshape(-1).float()).to(torch.float16)
                            out = out - x_2d.sum(dim=-1, keepdim=True) * zp_cor

                        if need_cast:
                            _get_cast()[1](self, weight, bias, offload_stream)
                        return out.to(x.dtype).reshape(*x.shape[:-1], -1)

                    # Path 3: PyTorch fallback — full dequant to float, then F.linear.
                    w_float = IntCrushInt4Layout.dequantize(qdata, weight._params).to(x.device, x.dtype)
                    x_2d = x.reshape(-1, x.shape[-1])
                    if self._intcrush_rot_need:
                        x_2d = rotate_activations(x_2d, self._intcrush_rot_size)
                    if perm is not None:
                        x_2d = x_2d[..., perm]
                    out = F.linear(x_2d, w_float)

                    if need_cast:
                        _get_cast()[1](self, weight, bias, offload_stream)
                    return out.to(x.dtype).reshape(*x.shape[:-1], -1)

                # ── INT8 forward paths ──
                if isinstance(weight._params, IntCrushInt8Layout.Params):
                    w_scale = weight._params.scale

                    x_2d = x.reshape(-1, x.shape[-1])
                    if self._intcrush_rot_need:
                        x_2d = rotate_activations(x_2d, self._intcrush_rot_size)

                    if x_2d.shape[-1] < w_in:
                        x_2d = F.pad(x_2d, (0, w_in - x_2d.shape[-1]))
                    if perm is not None:
                        x_2d = x_2d[..., perm]

                    # SmoothQuant: apply inverse smoothing factors (1/λ).
                    # The converter stores λ per-channel; at inference we
                    # divide activations by λ to complete the transformation
                    # Y = (X/λ) @ (λ·W).  Only present when --smoothquant
                    # was used during conversion.
                    smooth = self._intcrush_smooth
                    if smooth is not None:
                        x_2d = x_2d / smooth.to(device=x_2d.device, dtype=x_2d.dtype)

                    batch = x_2d.shape[0]
                    compute_dtype = (x_2d.dtype if x_2d.dtype in (torch.float16, torch.bfloat16)
                                     else torch.bfloat16)

                    # Small batch or no Triton: dequant weights to float and use F.linear.
                    if batch <= 16 or not _TRITON_AVAILABLE:
                        w_scale_2d = w_scale.reshape(-1, 1) if w_scale.ndim == 1 else w_scale
                        w_float = qdata.to(compute_dtype) * w_scale_2d.to(compute_dtype)
                        out = F.linear(x_2d, w_float, bias)
                    elif (batch <= 32 or (batch <= 128 and x_2d.shape[1] <= 4096)
                          ) and _HAS_FUSED_QUANT_GEMM and _TRITON_DYNQUANT:
                        # Medium batch: fused quantize+GEMM+dequant in a single kernel.
                        out = fused_quant_int8_gemm_dequant(
                            x_2d, qdata, w_scale,
                            bias=bias, out_dtype=compute_dtype,
                        )
                    else:
                        # Large batch: separate dynamic quantize + fused INT8 GEMM.
                        x_int8, s_a = dynamic_quantize_activation(x_2d)
                        out = fused_int8_gemm_dequant(
                            x_int8, qdata, w_scale, s_a,
                            bias=bias, out_dtype=compute_dtype,
                        )

                    if need_cast:
                        _get_cast()[1](self, weight, bias, offload_stream)
                    return out.to(x.dtype).reshape(*x.shape[:-1], -1)

                # Unknown layout — delegate to standard manual_cast forward.
                return super().forward(x)

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

    return IntCrushOps


# ── Loader helpers ───────────────────────────────────────────────────────────

def _get_diffusion_model_list():
    try:
        import folder_paths
        return folder_paths.get_filename_list("diffusion_models")
    except Exception:
        return []


def _detect_rot_size(metadata, default_rot_size, format_versions):
    """Read rot_size from safetensors metadata if available."""
    if not metadata:
        return default_rot_size
    if metadata.get("int_crush.format_version") not in format_versions:
        return default_rot_size
    detected = metadata.get("int_crush.rot_size")
    if detected is not None:
        try:
            detected = int(detected)
            if detected in (0, 16, 64, 256, 1024, 4096):
                print(f"[INT-Crush] Auto-detected rot_size={detected} from metadata")
                return detected
        except (ValueError, TypeError):
            pass
    return default_rot_size


# ── ComfyUI Node: INT4 Loader ────────────────────────────────────────────────

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
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "loaders"

    def load(self, unet_name: str, model_type: str, rot_size: int) -> tuple[object]:
        import folder_paths
        import comfy.utils
        from comfy.sd import load_diffusion_model

        global _use_pytorch, _use_w4a16

        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        if unet_path is None:
            raise FileNotFoundError(f"Model not found: {unet_name}")

        try:
            _, metadata = comfy.utils.load_torch_file(unet_path, return_metadata=True)
        except Exception:
            metadata = {}

        rot_size = _detect_rot_size(metadata, rot_size, ("2", "3"))

        _use_pytorch = False
        _use_w4a16 = False
        if rot_size == 0:
            print("[INT-Crush] W4A8 requires rotation — falling back to W4A16")
            _use_w4a16 = True

        ops_cls = _make_intcrush_ops("int4_crush", rot_size)
        model_options = {"custom_operations": ops_cls}
        model = load_diffusion_model(unet_path, model_options=model_options)

        return (model,)


# ── ComfyUI Node: INT8 Loader ────────────────────────────────────────────────

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

        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        if unet_path is None:
            raise FileNotFoundError(f"Model not found: {unet_name}")

        try:
            _, metadata = comfy.utils.load_torch_file(unet_path, return_metadata=True)
        except Exception:
            metadata = {}

        rot_size = _detect_rot_size(metadata, rot_size, ("1",))

        ops_cls = _make_intcrush_ops("int8_crush", rot_size)
        model_options = {"custom_operations": ops_cls}
        model = load_diffusion_model(unet_path, model_options=model_options)

        # Fix model config for padded layers (rotation padding inflates in_features).
        padded_str = (metadata or {}).get("int_crush.padded_layers", "")
        if padded_str:
            padded_map = {}
            for entry in padded_str.split(";"):
                if "=" in entry:
                    k, v = entry.split("=", 1)
                    padded_map[k.strip()] = int(v.strip())

            m = model.model
            for layer_key, orig_in in padded_map.items():
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

                if module_path == "img_in" and hasattr(m, 'in_channels'):
                    ps = getattr(m, 'patch_size', 1)
                    m.in_channels = orig_in // (ps * ps) if ps else orig_in
                    print(f"[INT-Crush] INT8: Fixed model.in_channels = {m.in_channels}")

        return (model,)


# ── Node registration ────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "SimpleINT4UNetLoader": SimpleINT4UNetLoader,
    "SimpleINT8UNetLoader": SimpleINT8UNetLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleINT4UNetLoader": "INT4 UNet Loader (INT-Crush)",
    "SimpleINT8UNetLoader": "INT8 UNet Loader (INT-Crush)",
}
