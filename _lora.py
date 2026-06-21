"""INT-Crush adapter residual management.

Provides helpers to attach/detach weight adapters (LoRA, LoKr, LoHa,
OFT, BOFT, GLoRA) as residual buffers on IntCrushOps.Linear modules.
The adapter's ``h(x, out)`` computes the additive delta and ``g(out)``
applies any output transformation (identity for most; orthogonal
rotation for OFT/BOFT).

All six adapter types supported by ComfyUI's weight_adapter system
work out of the box — the adapter's own math is used directly.
"""

import logging
import torch
from typing import Any

__all__ = ["clear_intcrush_lora", "attach_lora_as_buffers"]

_log = logging.getLogger(__name__)


def clear_intcrush_lora(model: Any) -> int:
    """Remove adapter residuals from all IntCrushOps.Linear modules."""
    count = 0
    for module in model.model.modules():
        if hasattr(module, '_intcrush_adapter'):
            module._intcrush_adapter = None
            module._intcrush_lora_strength = None
            module._intcrush_adapter_ready = False
            module._intcrush_adapter_device = None
            module._intcrush_lokr_cache = None
            module._intcrush_lokr_cache_adapter_id = None
            module._intcrush_lokr_cache_strength = None
            count += 1
    return count


def attach_lora_as_buffers(model: Any, lora_sd: dict, strength: float) -> tuple[int, int]:
    """Parse a LoRA state-dict and attach adapter objects as residual buffers.

    Uses ComfyUI's LoRA parsing to extract weight adapters, then stores
    each adapter on its matching IntCrushOps.Linear module.  At forward
    time the adapter's ``h(x, out)`` / ``g(out)`` methods are called
    directly — no per-type matrix extraction needed.

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

    clear_intcrush_lora(model)

    weight_key_to_module = {}
    for name, module in model.model.named_modules():
        if hasattr(module, '_intcrush_adapter'):
            weight_key_to_module[f"{name}.weight"] = module

    attached = 0
    for weight_key, patch in patches.items():
        module = weight_key_to_module.get(weight_key)
        if module is None:
            continue

        if not isinstance(patch, comfy.weight_adapter.WeightAdapterBase):
            continue

        # Store the adapter object — h()/g() handle the math for all types.
        module._intcrush_adapter = patch
        module._intcrush_lora_strength = strength
        attached += 1

    if attached == 0:
        _log.warning(
            "[INT-Crush LoRA] No modules matched (%d patches parsed "
            "but none mapped to INT-Crush layers)", len(patches),
        )

    return attached, len(patches)
