"""INT-Crush LoRA residual buffer management.

Provides helpers to attach/detach LoRA adapters as low-rank residual
buffers on IntCrushOps.Linear modules.  This avoids corrupting the
quantized weight by applying the LoRA in *unrotated* activation space.
"""

import torch
from typing import Any

__all__ = ["clear_intcrush_lora", "attach_lora_as_buffers"]


def clear_intcrush_lora(model: Any) -> int:
    """Remove LoRA buffers from all IntCrushOps.Linear modules in a model."""
    count = 0
    for module in model.model.modules():
        if hasattr(module, '_intcrush_lora_down'):
            module._intcrush_lora_down = None
            module._intcrush_lora_up = None
            module._intcrush_lora_scale = None
            count += 1
    return count


def attach_lora_as_buffers(model: Any, lora_sd: dict, strength: float) -> tuple[int, int]:
    """Parse a LoRA state-dict and attach A/B matrices as residual buffers.

    Uses ComfyUI's LoRA parsing to extract up/down matrices, then stores
    them as ``_intcrush_lora_down`` / ``_intcrush_lora_up`` on each
    matching IntCrushOps.Linear module.

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
        if hasattr(module, '_intcrush_lora_down'):
            weight_key_to_module[f"{name}.weight"] = module

    attached = 0
    for weight_key, patch in patches.items():
        module = weight_key_to_module.get(weight_key)
        if module is None:
            continue

        if not isinstance(patch, comfy.weight_adapter.WeightAdapterBase):
            continue

        # Weights layout: [0]=up, [1]=down, [2]=alpha, [3]=mid-decomposition
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
