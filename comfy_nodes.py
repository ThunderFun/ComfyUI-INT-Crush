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
                self._intcrush_smoothrot_factors = None
                self._intcrush_L1 = None
                self._intcrush_L2 = None
                self.quant_format = None
                self.layout_type = None
                # ── INT-Crush LoRA residual buffers ──
                self._intcrush_lora_down = None
                self._intcrush_lora_up = None
                self._intcrush_lora_scale = None

            def _intcrush_lora_apply(self, x_2d, out):
                """Apply LoRA residual via fused addmm_.

                Computes ``out += scale * (x @ down.T) @ up.T`` with an
                intermediate of shape [N, rank] only — no full [N, out]
                temporary is allocated.
                """
                if self._intcrush_lora_down is None:
                    return out
                down = self._intcrush_lora_down
                up = self._intcrush_lora_up
                if down.device != x_2d.device:
                    self._intcrush_lora_down = down.to(device=x_2d.device, dtype=x_2d.dtype)
                    self._intcrush_lora_up = up.to(device=x_2d.device, dtype=x_2d.dtype)
                    down = self._intcrush_lora_down
                    up = self._intcrush_lora_up
                scale = self._intcrush_lora_scale
                mid = F.linear(x_2d, down)                      # [N, rank]
                out.addmm_(mid, up.t(), beta=1, alpha=scale)    # fused into out
                return out

            def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys, error_msgs):
                from comfy.quant_ops import QuantizedTensor, get_layout_class, QUANT_ALGOS

                weight_key = prefix + "weight"
                scale_key = prefix + "weight_scale"
                zp_key = prefix + "weight_zp"
                perm_key = prefix + "weight_perm"
                smooth_key = prefix + "weight_smooth"
                smoothrot_key = prefix + "weight_smoothrot_factors"
                quant_key = prefix + "comfy_quant"
                L1_key = prefix + "weight_L1"
                L2_key = prefix + "weight_L2"

                weight = state_dict.get(weight_key)
                scale = state_dict.get(scale_key)

                # Detect INT-Crush format by dtype signature:
                #   INT4: weight=uint8, scale=float16 or float32
                #   INT8: weight=int8,  scale=float16 or float32
                is_int4 = (weight is not None and scale is not None
                           and weight.dtype == torch.uint8 and scale.dtype in (torch.float16, torch.float32))
                is_int8 = (weight is not None and scale is not None
                           and weight.dtype == torch.int8 and scale.dtype in (torch.float16, torch.float32))

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
                smoothrot_factors = state_dict.pop(smoothrot_key, None)
                L1 = state_dict.pop(L1_key, None)
                L2 = state_dict.pop(L2_key, None)
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
                # SmoothRot factors: applied BEFORE Hadamard (1/s → R).
                # Distinct from _intcrush_smooth which is applied AFTER (R → /s).
                self._intcrush_smoothrot_factors = (
                    smoothrot_factors.float().to(device=device) if smoothrot_factors is not None else None
                )
                # SVD low-rank factors: FP16 branch absorbed before quantization.
                self._intcrush_L1 = (
                    L1.to(device=device, dtype=torch.float16) if L1 is not None else None
                )
                self._intcrush_L2 = (
                    L2.to(device=device, dtype=torch.float16) if L2 is not None else None
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
                          smoothrot_key, L1_key, L2_key, prefix + "bias", quant_key):
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

                # ── SVD low-rank branch (computed on raw input, before rotation) ──
                svd_L1 = self._intcrush_L1
                svd_L2 = self._intcrush_L2
                has_svd = svd_L1 is not None and svd_L2 is not None
                if has_svd:
                    svd_L1 = svd_L1.to(device=x.device, dtype=torch.float16)
                    svd_L2 = svd_L2.to(device=x.device, dtype=torch.float16)

                # ── ComfyUI weight casting (VBAR + low-VRAM) ──
                wfn = self.weight_function
                bfn = self.bias_function
                need_cast = (
                    self.comfy_cast_weights
                    or wfn or bfn
                    or getattr(self, "weight_lowvram_function", None) is not None
                    or getattr(self, "bias_lowvram_function", None) is not None
                )

                QT = _get_qt()
                uncast = None

                if need_cast:
                    weight_dtype = (
                        self.weight._params.orig_dtype
                        if isinstance(self.weight, QT) else x.dtype
                    )
                    _cbw, _ucbw = _get_cast()
                    weight, bias, offload_stream = _cbw(
                        self, input=None, dtype=weight_dtype,
                        device=x.device, bias_dtype=x.dtype, offloadable=True,
                    )
                    uncast = lambda: _ucbw(self, weight, bias, offload_stream)
                else:
                    weight = self.weight
                    bias = self.bias

                def finish(out: torch.Tensor) -> torch.Tensor:
                    if uncast is not None:
                        uncast()
                    return out.to(x.dtype).reshape(*x.shape[:-1], -1)

                # ── Weight already dequantized (e.g. wrapper patch ran) ──
                if not isinstance(weight, QT):
                    x_2d = x.reshape(-1, x.shape[-1])
                    # SmoothRot: 1/s BEFORE Hadamard.
                    smoothrot_factors = self._intcrush_smoothrot_factors
                    if smoothrot_factors is not None:
                        x_2d = x_2d / smoothrot_factors.to(device=x_2d.device, dtype=x_2d.dtype)
                    if self._intcrush_rot_need:
                        x_2d = rotate_activations(x_2d, self._intcrush_rot_size)
                    out = F.linear(x_2d, weight, bias)
                    if has_svd:
                        x_raw = x.reshape(-1, x.shape[-1])
                        lr_out = x_raw.to(svd_L1.dtype) @ svd_L2.T @ svd_L1.T
                        out = out + lr_out.to(out.dtype)
                    # ── INT-Crush LoRA residual (unrotated space) ──
                    out = self._intcrush_lora_apply(x.reshape(-1, x.shape[-1]), out)
                    return finish(out)

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

                        # SmoothRot: 1/s BEFORE Hadamard.
                        smoothrot_factors = self._intcrush_smoothrot_factors
                        if smoothrot_factors is not None:
                            x_2d = x_2d / smoothrot_factors.to(device=x_2d.device, dtype=x_2d.dtype)
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

                        if has_svd:
                            x_raw = x.reshape(-1, x.shape[-1])
                            lr_out = x_raw.to(svd_L1.dtype) @ svd_L2.T @ svd_L1.T
                            out = out + lr_out.to(out.dtype)

                        # ── INT-Crush LoRA residual (unrotated space) ──
                        self._intcrush_lora_apply(x.reshape(-1, x.shape[-1]), out)

                        return finish(out)

                    # Path 2: W4A16 — Triton unpack to float16, then cuBLAS GEMM.
                    if not _use_pytorch and _TRITON_INT4_UNPACK:

                        x_2d = x.reshape(-1, x.shape[-1])

                        # SmoothRot: 1/s BEFORE Hadamard.
                        smoothrot_factors = self._intcrush_smoothrot_factors
                        if smoothrot_factors is not None:
                            x_2d = x_2d / smoothrot_factors.to(device=x_2d.device, dtype=x_2d.dtype)
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

                        if has_svd:
                            x_raw = x.reshape(-1, x.shape[-1])
                            lr_out = x_raw.to(svd_L1.dtype) @ svd_L2.T @ svd_L1.T
                            out = out + lr_out.to(out.dtype)

                        # ── INT-Crush LoRA residual (unrotated space) ──
                        self._intcrush_lora_apply(x.reshape(-1, x.shape[-1]), out)

                        return finish(out)

                    # Path 3: PyTorch fallback — full dequant to float, then F.linear.
                    w_float = IntCrushInt4Layout.dequantize(qdata, weight._params).to(x.device, x.dtype)
                    x_2d = x.reshape(-1, x.shape[-1])

                    # SmoothRot: 1/s BEFORE Hadamard.
                    smoothrot_factors = self._intcrush_smoothrot_factors
                    if smoothrot_factors is not None:
                        x_2d = x_2d / smoothrot_factors.to(device=x_2d.device, dtype=x_2d.dtype)
                    if self._intcrush_rot_need:
                        x_2d = rotate_activations(x_2d, self._intcrush_rot_size)
                    if perm is not None:
                        x_2d = x_2d[..., perm]
                    out = F.linear(x_2d, w_float)
                    if has_svd:
                        x_raw = x.reshape(-1, x.shape[-1])
                        lr_out = x_raw.to(svd_L1.dtype) @ svd_L2.T @ svd_L1.T
                        out = out + lr_out.to(out.dtype)
                    # ── INT-Crush LoRA residual (unrotated space) ──
                    out = self._intcrush_lora_apply(x.reshape(-1, x.shape[-1]), out)
                    return finish(out)

                # ── INT8 forward paths ──
                if isinstance(weight._params, IntCrushInt8Layout.Params):
                    w_scale = weight._params.scale

                    x_2d = x.reshape(-1, x.shape[-1])

                    # SmoothRot: apply 1/s BEFORE Hadamard (correct order).
                    # Old SmoothQuant: apply 1/s AFTER Hadamard (legacy order).
                    smoothrot_factors = self._intcrush_smoothrot_factors
                    if smoothrot_factors is not None:
                        x_2d = x_2d / smoothrot_factors.to(device=x_2d.device, dtype=x_2d.dtype)
                    if self._intcrush_rot_need:
                        x_2d = rotate_activations(x_2d, self._intcrush_rot_size)

                    if x_2d.shape[-1] < w_in:
                        x_2d = F.pad(x_2d, (0, w_in - x_2d.shape[-1]))
                    if perm is not None:
                        x_2d = x_2d[..., perm]

                    # Old SmoothQuant: 1/s AFTER Hadamard (only for non-SmoothRot layers).
                    if smoothrot_factors is None:
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
                    else:
                        # Two-kernel path (quantize → GEMM) is always faster than
                        # the fused quant+GEMM kernel for batch > 16: the fused kernel
                        # reads fp16 twice (abs-max pass + quantize pass), while this
                        # path reads fp16 once, then int8 on the GEMM (half bandwidth).
                        x_int8, s_a = dynamic_quantize_activation(x_2d)
                        out = fused_int8_gemm_dequant(
                            x_int8, qdata, w_scale, s_a,
                            bias=bias, out_dtype=compute_dtype,
                        )

                        if has_svd:
                            x_raw = x.reshape(-1, x.shape[-1])
                            lr_out = x_raw.to(svd_L1.dtype) @ svd_L2.T @ svd_L1.T
                            out = out + lr_out.to(out.dtype)

                        # ── INT-Crush LoRA residual (unrotated space) ──
                        self._intcrush_lora_apply(x.reshape(-1, x.shape[-1]), out)

                        return finish(out)

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
    CATEGORY = "loaders/INT-Crush"

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
    CATEGORY = "loaders/INT-Crush"

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


# ── INT-Crush LoRA loader ────────────────────────────────────────────────────

def _clear_intcrush_lora(model):
    """Remove LoRA buffers from all IntCrushOps.Linear modules in a model."""
    count = 0
    for module in model.model.modules():
        if hasattr(module, '_intcrush_lora_down'):
            module._intcrush_lora_down = None
            module._intcrush_lora_up = None
            module._intcrush_lora_scale = None
            count += 1
    return count


def _attach_lora_as_buffers(model, lora_sd, strength):
    """Use ComfyUI's LoRA parsing to extract A/B matrices and attach as
    residual buffers on IntCrushOps.Linear modules.

    Returns (attached_count, total_patches).
    """
    import comfy.lora
    import comfy.lora_convert
    import comfy.weight_adapter

    lora_sd = comfy.lora_convert.convert_lora(lora_sd)

    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    patches = comfy.lora.load_lora(lora_sd, key_map)

    if not patches:
        return 0, 0

    _clear_intcrush_lora(model)

    weight_key_to_module = {}
    for name, module in model.model.named_modules():
        if hasattr(module, '_intcrush_lora_down'):
            weight_key_to_module[f"{name}.weight"] = module

    attached = 0
    for weight_key, patch in patches.items():
        module = weight_key_to_module.get(weight_key)
        if module is None:
            continue

        if not isinstance(patch, comfy.weight_adapter.WeightAdapterBase):
            continue

        mat_up, mat_down = patch.weights[0], patch.weights[1]
        alpha = patch.weights[2]
        mid = patch.weights[3]

        if mid is not None:
            continue  # Conv LoRA (tucker) not supported yet

        if mat_down.dim() > 2:
            mat_down = mat_down.reshape(mat_down.shape[0], -1)
        if mat_up.dim() > 2:
            mat_up = mat_up.reshape(mat_up.shape[0], -1)

        rank = mat_down.shape[0]
        if alpha is not None and rank > 0:
            scale = strength * (float(alpha) / rank)
        else:
            scale = strength

        # Store low-rank matrices on CPU; they are moved to GPU on first forward.
        module._intcrush_lora_down = mat_down.to(dtype=torch.float16)
        module._intcrush_lora_up = mat_up.to(dtype=torch.float16)
        module._intcrush_lora_scale = scale
        attached += 1

    return attached, len(patches)


class IntCrushLoRALoader:
    """Load a LoRA file and attach it as a residual on the raw (unrotated)
    activation, preserving ConvRot/SmoothRot quantization and the fast
    Triton INT4/INT8 GEMM kernel paths.

    Standard ComfyUI LoRA patching adds ΔW to the *rotated* weight matrix,
    which corrupts the LoRA contribution. This node instead stores the LoRA
    as separate low-rank buffers (down-projection A, up-projection B) and
    applies ``out += (x_raw @ Aᵀ) @ Bᵀ * scale`` in the forward pass,
    *before* any rotation or smoothing. This is mathematically equivalent
    to the true unrotated LoRA and keeps the main weight as a
    QuantizedTensor so the Triton kernels stay active.

    Usage: connect MODEL output from an INT4/INT8 loader into this node,
    then connect the output to KSampler (or wherever the model is used).
    """

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the LoRA."}),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "loaders/INT-Crush"

    def load(self, model, lora_name, strength):
        import folder_paths
        import comfy.utils

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora_sd, _metadata = comfy.utils.load_torch_file(
            lora_path, safe_load=True, return_metadata=True)

        if not lora_sd:
            raise ValueError(f"INT-Crush LoRA: empty file '{lora_name}'")

        attached, total = _attach_lora_as_buffers(model, lora_sd, strength)

        if attached == 0:
            print(f"[INT-Crush LoRA] WARNING: no modules matched "
                  f"({total} patches parsed but none mapped to INT-Crush layers)")
        else:
            print(f"[INT-Crush LoRA] Attached to {attached} layer(s)")

        return (model,)


class IntCrushLoRAUnloader:
    """Remove INT-Crush LoRA residual buffers from a model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "unload"
    CATEGORY = "loaders/INT-Crush"

    def unload(self, model):
        n = _clear_intcrush_lora(model)
        print(f"[INT-Crush LoRA] Cleared LoRA from {n} layer(s)")
        return (model,)


# ── Node registration ────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "SimpleINT4UNetLoader": SimpleINT4UNetLoader,
    "SimpleINT8UNetLoader": SimpleINT8UNetLoader,
    "IntCrushLoRALoader": IntCrushLoRALoader,
    "IntCrushLoRAUnloader": IntCrushLoRAUnloader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleINT4UNetLoader": "INT4 UNet Loader (INT-Crush)",
    "SimpleINT8UNetLoader": "INT8 UNet Loader (INT-Crush)",
    "IntCrushLoRALoader": "LoRA Loader (INT-Crush)",
    "IntCrushLoRAUnloader": "LoRA Unloader (INT-Crush)",
}
