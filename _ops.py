"""INT-Crush ops factory — builds custom manual_cast ops for quantized inference.

Returns QuantizedTensor weights for VBAR/memory management, but the
forward() bypasses mixed_precision_ops overhead — going directly from
packed weights to Triton kernels (or PyTorch fallback) without the
per-input QuantizedTensor.from_float / cast_bias_weight round-trip.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._quant_utils import rotate_activations
from .quant_layout import IntCrushInt4Layout, IntCrushInt8Layout
from . import _triton_runtime as _tr

_log = logging.getLogger(__name__)

# ── Module-level constants ───────────────────────────────────────────────────

# Layer-name substrings that disable Hadamard rotation (embedding, norm, head,
# projection layers are not rotated — they have isotropic weight distributions).
_NO_ROTATION_NAME_SUBSTRINGS: tuple[str, ...] = (
    "embed", "norm", "modulation", "output", "lm_head", "proj_out",
)

# Safetensors state-dict key suffixes used by INT-Crush quantized layers.
# All keys are prefixed by the module path (e.g. "model.layer.weight_scale").
_SD_KEY_WEIGHT = "weight"
_SD_KEY_SCALE = "weight_scale"
_SD_KEY_ZP = "weight_zp"
_SD_KEY_PERM = "weight_perm"
_SD_KEY_SMOOTH = "weight_smooth"
_SD_KEY_SMOOTHROT = "weight_smoothrot_factors"
_SD_KEY_L1 = "weight_L1"
_SD_KEY_L2 = "weight_L2"
_SD_KEY_QUANT = "comfy_quant"
_SD_KEY_BIAS = "bias"

# Valid float dtypes for weight scales.
_VALID_SCALE_DTYPES = (torch.float16, torch.float32)

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


# ── Ops factory ──────────────────────────────────────────────────────────────

_ops_cache: dict[tuple, type] = {}


def make_intcrush_ops(
    quant_format: str,
    rot_size: int,
    use_pytorch: bool = False,
    use_w4a16: bool = False,
) -> type:
    """Build a manual_cast ops class with lean INT-Crush forward.

    Args:
        quant_format: "int4_crush" or "int8_crush"
        rot_size: Hadamard rotation group size
        use_pytorch: force PyTorch fallback (no Triton kernels)
        use_w4a16: disable W4A8 path, use W4A16 unpack+GEMM instead
    """
    cache_key = (quant_format, rot_size, use_pytorch, use_w4a16)
    if cache_key in _ops_cache:
        return _ops_cache[cache_key]
    from comfy.ops import manual_cast

    class IntCrushOps(manual_cast):

        class Linear(manual_cast.Linear):
            """INT-Crush Linear: QuantizedTensor storage with lean forward()."""

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self._intcrush_is_quantized: bool = False
                self._intcrush_rot_need: bool = True
                self._intcrush_rot_size: int = rot_size
                self._intcrush_perm: torch.Tensor | None = None
                self._intcrush_smooth: torch.Tensor | None = None
                self._intcrush_smoothrot_factors: torch.Tensor | None = None
                self._intcrush_L1: torch.Tensor | None = None
                self._intcrush_L2: torch.Tensor | None = None
                self._intcrush_use_pytorch: bool = use_pytorch
                self._intcrush_use_w4a16: bool = use_w4a16
                self._intcrush_w_in: int | None = None
                self.quant_format: str | None = None
                self.layout_type: str | None = None
                # ── INT-Crush LoRA residual buffers ──
                self._intcrush_lora_down: torch.Tensor | None = None
                self._intcrush_lora_up: torch.Tensor | None = None
                self._intcrush_lora_scale: float | None = None

            def _intcrush_lora_apply(self, x_2d: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
                """Apply LoRA residual via fused addmm_.

                Computes ``out += scale * (x @ down.T) @ up.T`` with an
                intermediate of shape [N, rank] only — no full [N, out]
                temporary is allocated.
                """
                if self._intcrush_lora_down is None:
                    return out
                down = self._intcrush_lora_down
                up = self._intcrush_lora_up
                target_dtype = out.dtype
                # Move/cast on device OR dtype mismatch.  Store in out.dtype
                # so addmm_ doesn't need a temporary cast of mid/up.
                if down.device != x_2d.device or down.dtype != target_dtype:
                    self._intcrush_lora_down = down.to(device=x_2d.device, dtype=target_dtype)
                    self._intcrush_lora_up = up.to(device=x_2d.device, dtype=target_dtype)
                    down = self._intcrush_lora_down
                    up = self._intcrush_lora_up
                scale = self._intcrush_lora_scale
                mid = F.linear(x_2d, down)                      # [N, rank]
                out.addmm_(mid, up.t(), beta=1, alpha=scale)
                return out

            def _prep_activation(self, x: torch.Tensor, pad_to_w_in: bool = True, apply_perm: bool = True) -> torch.Tensor:
                """Common activation prep: reshape → SmoothRot → Hadamard → pad → permute.

                Args:
                    x: raw input tensor [..., in_features]
                    pad_to_w_in: pad feature dim to match packed weight width
                    apply_perm: apply PermuQuant channel permutation
                """
                x_2d = x.reshape(-1, x.shape[-1])

                # SmoothRot: 1/s BEFORE Hadamard.
                smoothrot_factors = self._intcrush_smoothrot_factors
                if smoothrot_factors is not None:
                    x_2d = x_2d / smoothrot_factors.to(device=x_2d.device, dtype=x_2d.dtype)

                if self._intcrush_rot_need:
                    x_2d = rotate_activations(x_2d, self._intcrush_rot_size)

                if pad_to_w_in:
                    w_in = self._intcrush_w_in
                    if x_2d.shape[-1] < w_in:
                        x_2d = F.pad(x_2d, (0, w_in - x_2d.shape[-1]))

                if apply_perm:
                    perm = self._intcrush_perm
                    if perm is not None:
                        x_2d = x_2d[..., perm]

                return x_2d

            def _apply_residuals(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
                """Apply SVD low-rank residual and LoRA residual in unrotated space."""
                svd_L1 = self._intcrush_L1
                svd_L2 = self._intcrush_L2
                if svd_L1 is not None and svd_L2 is not None:
                    # Rebind to self after device/dtype move so subsequent
                    # forwards reuse the cached tensor (avoids re-allocation
                    # when the model has been relocated to a different device).
                    if svd_L1.device != x.device or svd_L1.dtype != torch.float16:
                        svd_L1 = svd_L1.to(device=x.device, dtype=torch.float16)
                        self._intcrush_L1 = svd_L1
                    if svd_L2.device != x.device or svd_L2.dtype != torch.float16:
                        svd_L2 = svd_L2.to(device=x.device, dtype=torch.float16)
                        self._intcrush_L2 = svd_L2
                    x_raw = x.reshape(-1, x.shape[-1])
                    mid = x_raw.to(svd_L1.dtype) @ svd_L2.T
                    out.addmm_(mid, svd_L1.T.to(out.dtype), beta=1, alpha=1)

                self._intcrush_lora_apply(x.reshape(-1, x.shape[-1]), out)
                return out

            def _finish_forward(self, out: torch.Tensor, x: torch.Tensor, uncast: Callable[..., None] | None) -> torch.Tensor:
                """Uncast weight, convert dtype, reshape to original input shape."""
                if uncast is not None:
                    uncast()
                out = out.to(x.dtype)
                return out.reshape(*x.shape[:-1], -1)

            # ── forward-path helpers ──────────────────────────────────────────

            def _setup_casting(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, Callable[..., None] | None, type]:
                """Resolve weight/bias/uncast for the forward pass.

                Handles VBAR casting, low-VRAM offload, and the fast path
                where no casting is needed. Returns (weight, bias, uncast, QT).
                """
                QT = _get_qt()
                wfn = self.weight_function
                bfn = self.bias_function
                need_cast = (
                    self.comfy_cast_weights
                    or wfn or bfn
                    or getattr(self, "weight_lowvram_function", None) is not None
                    or getattr(self, "bias_lowvram_function", None) is not None
                )

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
                    uncast = None

                return weight, bias, uncast, QT

            def _forward_predequant(self, weight: torch.Tensor, bias: torch.Tensor | None,
                                    x: torch.Tensor, uncast: Callable[..., None] | None) -> torch.Tensor:
                """Forward when weight is already dequantized (e.g. wrapper patch ran)."""
                x_2d = self._prep_activation(x, pad_to_w_in=False, apply_perm=False)
                out = F.linear(x_2d, weight, bias)
                out = self._apply_residuals(x, out)
                return self._finish_forward(out, x, uncast)

            def _forward_int4_w4a8(self, qdata: torch.Tensor, weight: Any,
                                   bias: torch.Tensor | None, x: torch.Tensor,
                                   uncast: Callable[..., None] | None) -> torch.Tensor:
                """INT4 Path 1: W4A8 — unpack INT4→INT8, dynamic-quantize activations,
                fused INT8 GEMM + dequant (fastest when all Triton kernels available)."""
                x_2d = self._prep_activation(x)
                x_int8, s_a = _tr.dynamic_quantize_activation(x_2d)
                needs_zp = weight._params.zp is not None
                x_2d_sum = x_2d.sum(dim=-1, keepdim=True) if needs_zp else None
                del x_2d
                w_int8 = _tr.unpack_int4_to_int8(qdata, self._intcrush_w_in)
                scale_flat = weight._params.scale.reshape(-1).float().contiguous()
                _gemm_fn = _tr.fused_w4a8_gemm_dequant if _tr.TRITON_W4A8_GEMM else _tr.fused_int8_gemm_dequant
                out = _gemm_fn(
                    x_int8, w_int8, scale_flat, s_a,
                    bias=bias, out_dtype=x.dtype,
                )
                del x_int8, w_int8
                if needs_zp:
                    zp_cor = (scale_flat * weight._params.zp.reshape(-1).float()).to(out.dtype)
                    out.addmm_(x_2d_sum, zp_cor.unsqueeze(0), beta=1, alpha=-1)

                out = self._apply_residuals(x, out)
                return self._finish_forward(out, x, uncast)

            def _forward_int4_w4a16(self, qdata: torch.Tensor, weight: Any,
                                    bias: torch.Tensor | None, x: torch.Tensor,
                                    uncast: Callable[..., None] | None) -> torch.Tensor:
                """INT4 Path 2: W4A16 — Triton unpack to float16, then cuBLAS GEMM."""
                x_2d = self._prep_activation(x)
                scale_flat = weight._params.scale.reshape(-1).float().contiguous()
                needs_zp = weight._params.zp is not None
                x_2d_sum = x_2d.sum(dim=-1, keepdim=True) if needs_zp else None
                weight_f16 = _tr.unpack_int4_to_float16(qdata, scale_flat, self._intcrush_w_in)
                out = F.linear(x_2d, weight_f16)
                del x_2d, weight_f16
                if needs_zp:
                    zp_cor = (scale_flat * weight._params.zp.reshape(-1).float()).to(out.dtype)
                    out.addmm_(x_2d_sum, zp_cor.unsqueeze(0), beta=1, alpha=-1)

                out = self._apply_residuals(x, out)
                return self._finish_forward(out, x, uncast)

            def _forward_int4_pytorch(self, qdata: torch.Tensor, weight: Any,
                                      x: torch.Tensor, uncast: Callable[..., None] | None) -> torch.Tensor:
                """INT4 Path 3: PyTorch fallback — full dequant to float, then F.linear."""
                w_float = IntCrushInt4Layout.dequantize(qdata, weight._params).to(x.device, x.dtype)
                x_2d = self._prep_activation(x, pad_to_w_in=False)
                out = F.linear(x_2d, w_float)
                del x_2d, w_float
                out = self._apply_residuals(x, out)
                return self._finish_forward(out, x, uncast)

            def _forward_int8(self, qdata: torch.Tensor, weight: Any,
                              bias: torch.Tensor | None, x: torch.Tensor,
                              uncast: Callable[..., None] | None) -> torch.Tensor:
                """INT8 forward: small-batch PyTorch fallback or Triton two-kernel path."""
                w_scale = weight._params.scale
                x_2d = self._prep_activation(x)

                # Old SmoothQuant: 1/s AFTER Hadamard (only for non-SmoothRot layers).
                if self._intcrush_smoothrot_factors is None:
                    smooth = self._intcrush_smooth
                    if smooth is not None:
                        x_2d = x_2d / smooth.to(device=x_2d.device, dtype=x_2d.dtype)

                batch = x_2d.shape[0]
                compute_dtype = (x_2d.dtype if x_2d.dtype in (torch.float16, torch.bfloat16)
                                 else torch.bfloat16)

                # Small batch or no Triton: dequant weights to float and use F.linear.
                if batch <= 16 or not _tr.TRITON_AVAILABLE:
                    w_scale_2d = w_scale.reshape(-1, 1) if w_scale.ndim == 1 else w_scale
                    w_float = qdata.to(compute_dtype) * w_scale_2d.to(compute_dtype)
                    out = F.linear(x_2d, w_float, bias)
                    del x_2d, w_float
                else:
                    # Two-kernel path (quantize → GEMM) is always faster than
                    # the fused quant+GEMM kernel for batch > 16: the fused kernel
                    # reads fp16 twice (abs-max pass + quantize pass), while this
                    # path reads fp16 once, then int8 on the GEMM (half bandwidth).
                    x_int8, s_a = _tr.dynamic_quantize_activation(x_2d)
                    del x_2d
                    out = _tr.fused_int8_gemm_dequant(
                        x_int8, qdata, w_scale, s_a,
                        bias=bias, out_dtype=compute_dtype,
                    )
                    del x_int8

                out = self._apply_residuals(x, out)
                return self._finish_forward(out, x, uncast)

            # ── state-dict loading helpers ────────────────────────────────────

            @staticmethod
            def _detect_weight_format(weight: torch.Tensor | None, scale: torch.Tensor | None) -> str | None:
                """Detect INT-Crush format by weight/scale dtype signature.

                Returns "int4_crush" / "int8_crush" / None.
                """
                if weight is None or scale is None:
                    return None
                if scale.dtype not in _VALID_SCALE_DTYPES:
                    return None
                if weight.dtype == torch.uint8:
                    return "int4_crush"
                if weight.dtype == torch.int8:
                    return "int8_crush"
                return None

            def _pop_intcrush_tensors(self, state_dict: dict, prefix: str) -> dict[str, Any]:
                """Pop all INT-Crush tensors from *state_dict* and return them as a dict.

                The returned dict has keys: weight, scale, zp, perm, smooth,
                smoothrot_factors, L1, L2, bias.  All popped keys are
                returned so they can be removed from missing_keys.
                """
                result: dict[str, Any] = {}
                result["weight"] = state_dict.pop(prefix + _SD_KEY_WEIGHT)
                result["scale"] = state_dict.pop(prefix + _SD_KEY_SCALE)
                result["zp"] = state_dict.pop(prefix + _SD_KEY_ZP, None)
                result["perm"] = state_dict.pop(prefix + _SD_KEY_PERM, None)
                result["smooth"] = state_dict.pop(prefix + _SD_KEY_SMOOTH, None)
                result["smoothrot_factors"] = state_dict.pop(prefix + _SD_KEY_SMOOTHROT, None)
                result["L1"] = state_dict.pop(prefix + _SD_KEY_L1, None)
                result["L2"] = state_dict.pop(prefix + _SD_KEY_L2, None)
                result["bias"] = state_dict.pop(prefix + _SD_KEY_BIAS, None)
                state_dict.pop(prefix + _SD_KEY_QUANT, None)
                return result

            def _store_intcrush_attrs(
                self,
                tensors: dict[str, Any],
                params: Any,
                device: torch.device,
            ) -> None:
                """Store quantization metadata and auxiliary tensors as instance attributes."""
                self._intcrush_perm = params.perm
                self._intcrush_w_in = self.in_features
                self._intcrush_smooth = (
                    tensors["smooth"].float().to(device=device)
                    if tensors["smooth"] is not None else None
                )
                # SmoothRot factors: applied BEFORE Hadamard (1/s → R).
                # Distinct from _intcrush_smooth which is applied AFTER (R → /s).
                self._intcrush_smoothrot_factors = (
                    tensors["smoothrot_factors"].float().to(device=device)
                    if tensors["smoothrot_factors"] is not None else None
                )
                # SVD low-rank factors: FP16 branch absorbed before quantization.
                self._intcrush_L1 = (
                    tensors["L1"].to(device=device, dtype=torch.float16)
                    if tensors["L1"] is not None else None
                )
                self._intcrush_L2 = (
                    tensors["L2"].to(device=device, dtype=torch.float16)
                    if tensors["L2"] is not None else None
                )

            @staticmethod
            def _clear_known_missing_keys(
                missing_keys: list[str],
                prefix: str,
            ) -> None:
                """Remove INT-Crush state-dict keys from missing_keys so PyTorch
                doesn't complain about legitimately absent standard keys."""
                known_suffixes = (
                    _SD_KEY_WEIGHT, _SD_KEY_SCALE, _SD_KEY_ZP, _SD_KEY_PERM,
                    _SD_KEY_SMOOTH, _SD_KEY_SMOOTHROT, _SD_KEY_L1, _SD_KEY_L2,
                    _SD_KEY_BIAS, _SD_KEY_QUANT,
                )
                for suffix in known_suffixes:
                    k = prefix + suffix
                    if k in missing_keys:
                        missing_keys.remove(k)

            # ── PyTorch Module overrides ──────────────────────────────────────

            def _load_from_state_dict(
                self,
                state_dict: dict,
                prefix: str,
                local_metadata: Any,
                strict: bool,
                missing_keys: list[str],
                unexpected_keys: list[str],
                error_msgs: list[str],
            ) -> None:
                from comfy.quant_ops import QuantizedTensor, get_layout_class, QUANT_ALGOS

                weight = state_dict.get(prefix + _SD_KEY_WEIGHT)
                scale = state_dict.get(prefix + _SD_KEY_SCALE)
                fmt = self._detect_weight_format(weight, scale)

                if fmt is None:
                    # Not an INT-Crush layer — delegate to default loader.
                    super()._load_from_state_dict(
                        state_dict, prefix, local_metadata,
                        strict, missing_keys, unexpected_keys, error_msgs)
                    return

                # ── INT-Crush layer: wrap as QuantizedTensor ──
                qconfig = QUANT_ALGOS[fmt]
                layout_type = qconfig["comfy_tensor_layout"]
                layout_cls = get_layout_class(layout_type)
                tensors = self._pop_intcrush_tensors(state_dict, prefix)

                device = getattr(self, 'weight', None)
                device = device.device if device is not None else torch.device("cpu")

                # Determine whether this layer needs Hadamard rotation based on name.
                name_lower = prefix.lower()
                rot_need = (
                    rot_size > 0 and not any(
                        p in name_lower for p in _NO_ROTATION_NAME_SUBSTRINGS
                    )
                )
                self._intcrush_rot_need = rot_need
                self._intcrush_rot_size = rot_size
                self._intcrush_is_quantized = True

                perm = tensors["perm"]
                zp = tensors["zp"]

                if fmt == "int4_crush":
                    params = layout_cls.Params(
                        scale=tensors["scale"].float().to(device=device),
                        orig_dtype=torch.float16,
                        orig_shape=(self.out_features, self.in_features),
                        rot_need=rot_need,
                        rot_size=rot_size,
                        perm=perm.to(device=device) if perm is not None else None,
                        zp=zp.to(device=device) if zp is not None else None,
                    )
                else:
                    params = layout_cls.Params(
                        scale=tensors["scale"].float().reshape(-1).to(device=device),
                        orig_dtype=torch.float16,
                        orig_shape=(self.out_features, self.in_features),
                        rot_need=rot_need,
                        rot_size=rot_size,
                        perm=perm.to(device=device) if perm is not None else None,
                    )

                self._store_intcrush_attrs(tensors, params, device)

                self.weight = nn.Parameter(
                    QuantizedTensor(
                        tensors["weight"].to(device=device, dtype=qconfig["storage_t"]),
                        layout_type, params,
                    ),
                    requires_grad=False,
                )
                self.quant_format = fmt
                self.layout_type = layout_type

                bias_tensor = tensors["bias"]
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

                self._clear_known_missing_keys(missing_keys, prefix)

            def convert_weight(self, weight: torch.Tensor, inplace: bool = False) -> torch.Tensor:
                """Keep QuantizedTensor as-is for ModelPatcher compatibility."""
                from comfy.quant_ops import QuantizedTensor
                if self._intcrush_is_quantized and isinstance(weight, QuantizedTensor):
                    return weight
                return weight

            def set_weight(self, out_weight: torch.Tensor, inplace_update: bool = False, seed: int = 0,
                           return_weight: bool = False, **kwargs: Any) -> torch.Tensor | None:
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

            def set_bias(self, out_bias: torch.Tensor | None, inplace_update: bool = False, seed: int = 0,
                         return_weight: bool = False, **kwargs: Any) -> torch.Tensor | None:
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

                # ── ComfyUI weight casting (VBAR + low-VRAM) ──
                weight, bias, uncast, QT = self._setup_casting(x)

                # ── Weight already dequantized (e.g. wrapper patch ran) ──
                if not isinstance(weight, QT):
                    return self._forward_predequant(weight, bias, x, uncast)

                qdata = weight._qdata

                # ── INT4 forward paths ──
                if isinstance(weight._params, IntCrushInt4Layout.Params):
                    _have_w4a8_kernel = (
                        (_tr.TRITON_W4A8_GEMM or _tr.TRITON_INT8_GEMM)
                        and _tr.TRITON_DYNQUANT
                        and _tr.TRITON_INT4_INT8_UNPACK
                    )
                    if (not self._intcrush_use_pytorch and not self._intcrush_use_w4a16
                            and self._intcrush_rot_need
                            and _have_w4a8_kernel):
                        return self._forward_int4_w4a8(qdata, weight, bias, x, uncast)

                    if not self._intcrush_use_pytorch and _tr.TRITON_INT4_UNPACK:
                        return self._forward_int4_w4a16(qdata, weight, bias, x, uncast)

                    return self._forward_int4_pytorch(qdata, weight, x, uncast)

                # ── INT8 forward paths ──
                if isinstance(weight._params, IntCrushInt8Layout.Params):
                    return self._forward_int8(qdata, weight, bias, x, uncast)

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

    _ops_cache[cache_key] = IntCrushOps
    return IntCrushOps


__all__ = ["make_intcrush_ops"]
