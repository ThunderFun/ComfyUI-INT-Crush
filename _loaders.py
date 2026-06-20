"""ComfyUI loader nodes for INT-Crush quantized models.

Provides four ComfyUI nodes:
  - SimpleINT4UNetLoader: loads a model quantized with INT4 weights
  - SimpleINT8UNetLoader: loads a model quantized with INT8 weights
  - IntCrushLoRALoader: attaches a LoRA as a residual buffer
  - IntCrushLoRAUnloader: removes LoRA residual buffers
"""

import logging
import torch

from ._ops import make_intcrush_ops
from ._lora import clear_intcrush_lora, attach_lora_as_buffers

log = logging.getLogger(__name__)

# ── Shared constants ─────────────────────────────────────────────────────────

# Valid rotation-group sizes (0 = disable rotation, fall back to W4A16 for INT4).
VALID_ROT_SIZES: tuple[int, ...] = (0, 16, 64, 256, 1024, 4096)

# Safetensors format_version values that use INT-Crush quantization.
_INT4_FORMAT_VERSIONS: tuple[str, ...] = ("2", "3")
_INT8_FORMAT_VERSIONS: tuple[str, ...] = ("1",)


# ── Loader helpers ───────────────────────────────────────────────────────────

def _get_diffusion_model_list() -> list[str]:
    try:
        import folder_paths
        return folder_paths.get_filename_list("diffusion_models")
    except Exception:
        return []


def _get_lora_list() -> list[str]:
    try:
        import folder_paths
        return folder_paths.get_filename_list("loras")
    except Exception:
        return []


def _detect_rot_size(metadata: dict | None, default_rot_size: int, format_versions: tuple[str, ...]) -> int:
    """Read rot_size from safetensors metadata if available."""
    if not metadata:
        return default_rot_size
    if metadata.get("int_crush.format_version") not in format_versions:
        return default_rot_size
    detected = metadata.get("int_crush.rot_size")
    if detected is not None:
        try:
            detected = int(detected)
            if detected in VALID_ROT_SIZES:
                log.info("[INT-Crush] Auto-detected rot_size=%d from metadata", detected)
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
                "rot_size": ([s for s in VALID_ROT_SIZES], {"default": 256}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "loaders/INT-Crush"

    def load(self, unet_name: str, rot_size: int) -> tuple[object]:
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

        rot_size = _detect_rot_size(metadata, rot_size, _INT4_FORMAT_VERSIONS)

        use_w4a16 = False
        if rot_size == 0:
            log.info("[INT-Crush] W4A8 requires rotation — falling back to W4A16")
            use_w4a16 = True

        ops_cls = make_intcrush_ops("int4_crush", rot_size, use_w4a16=use_w4a16)
        model_options = {"custom_operations": ops_cls}
        torch.cuda.empty_cache()
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
                "rot_size": ([s for s in VALID_ROT_SIZES], {"default": 256}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "loaders/INT-Crush"

    def load(self, unet_name: str, rot_size: int) -> tuple[object]:
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

        rot_size = _detect_rot_size(metadata, rot_size, _INT8_FORMAT_VERSIONS)

        ops_cls = make_intcrush_ops("int8_crush", rot_size)
        model_options = {"custom_operations": ops_cls}
        torch.cuda.empty_cache()
        model = load_diffusion_model(unet_path, model_options=model_options)

        # Fix model config for padded layers (rotation padding inflates in_features).
        padded_str = (metadata or {}).get("int_crush.padded_layers", "")
        if padded_str:
            _apply_padded_layer_fixes(model, padded_str)

        return (model,)


def _apply_padded_layer_fixes(model, padded_str: str) -> None:
    """Trim weight rows and fix in_channels for layers whose in_features
    were padded during quantization.

    *padded_str* is a semicolon-separated ``"key=orig_in_features"`` list
    stored in safetensors metadata under ``int_crush.padded_layers``.
    """
    import comfy.utils

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
                log.info("[INT-Crush] INT8: Fixed %s in_features %d -> %d",
                         layer_key, padded_in, orig_in)

        if module_path == "img_in" and hasattr(m, 'in_channels'):
            ps = getattr(m, 'patch_size', 1)
            m.in_channels = orig_in // (ps * ps) if ps else orig_in
            log.info("[INT-Crush] INT8: Fixed model.in_channels = %d", m.in_channels)


# ── INT-Crush LoRA nodes ─────────────────────────────────────────────────────

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
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (_get_lora_list(), {"tooltip": "The name of the LoRA."}),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "loaders/INT-Crush"

    def load(self, model: object, lora_name: str, strength: float) -> tuple[object]:
        import folder_paths
        import comfy.utils

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora_sd, _metadata = comfy.utils.load_torch_file(
            lora_path, safe_load=True, return_metadata=True)

        if not lora_sd:
            raise ValueError(f"INT-Crush LoRA: empty file '{lora_name}'")

        attached, total = attach_lora_as_buffers(model, lora_sd, strength)

        if attached == 0:
            log.warning("[INT-Crush LoRA] No modules matched (%d patches parsed but none mapped to INT-Crush layers)", total)
        else:
            log.info("[INT-Crush LoRA] Attached to %d layer(s)", attached)

        return (model,)


class IntCrushLoRAUnloader:
    """Remove INT-Crush LoRA residual buffers from a model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "unload"
    CATEGORY = "loaders/INT-Crush"

    def unload(self, model: object) -> tuple[object]:
        n = clear_intcrush_lora(model)
        log.info("[INT-Crush LoRA] Cleared LoRA from %d layer(s)", n)
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

__all__ = [
    "SimpleINT4UNetLoader",
    "SimpleINT8UNetLoader",
    "IntCrushLoRALoader",
    "IntCrushLoRAUnloader",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
