"""ModelPatcher extensions for INT-Crush INT4/INT8 models.

Provides:
  - INT4ModelPatcher / INT8ModelPatcher: intercept weight patching so LoRA
    deltas are applied in float space then re-quantized back to INT4/INT8.
  - INT4LowVramPatch / INT8LowVramPatch: on-demand bake-in of LoRA patches
    for --fast dynamic_vram.
"""

import torch
import torch.nn as nn
import comfy.model_patcher
import comfy.utils
import comfy.lora
import inspect

from ._quant_utils import unpack_int4, calculate_scales, quantize_weights, pack_int4

try:
    _prefetch_sig = inspect.signature(comfy.lora.prefetch_prepared_value)
    _use_new_prefetch = len(_prefetch_sig.parameters) == 5
except Exception:
    _use_new_prefetch = False


class INT4LowVramPatch:
    """Low-VRAM patch that dequantizes, applies LoRA, and re-quantizes to INT4."""
    is_lowvram_patch = True

    def __init__(self, key, patches, module):
        self.key = key
        self.patches = patches
        self.module = module
        self.prepared_patches = None

    def memory_required(self):
        if not _use_new_prefetch:
            return 0
        counter = [0]
        for patch in self.patches[self.key]:
            comfy.lora.prefetch_prepared_value(patch[1], counter, None, None, False)
        return counter[0]

    def prepare(self, *args, **kwargs):
        if _use_new_prefetch:
            destination = args[0] if len(args) > 0 else kwargs.get("destination")
            stream = args[1] if len(args) > 1 else kwargs.get("stream")
            copy = args[2] if len(args) > 2 else kwargs.get("copy", True)
            commit = args[3] if len(args) > 3 else kwargs.get("commit", True)

            counter = [0]
            prepared_patches = [
                (patch[0], comfy.lora.prefetch_prepared_value(patch[1], counter, destination, stream, copy), patch[2], patch[3], patch[4])
                for patch in self.patches[self.key]
            ]
            if commit:
                self.prepared_patches = prepared_patches
            return prepared_patches
        else:
            allocate_buffer = args[0] if len(args) > 0 else kwargs.get("allocate_buffer")
            stream = args[1] if len(args) > 1 else kwargs.get("stream")

            self.prepared_patches = [
                (patch[0], comfy.lora.prefetch_prepared_value(patch[1], allocate_buffer, stream), patch[2], patch[3], patch[4])
                for patch in self.patches[self.key]
            ]
            return self.prepared_patches

    def clear_prepared(self):
        self.prepared_patches = None

    def __call__(self, weight):
        patches = self.prepared_patches if self.prepared_patches is not None else self.patches[self.key]

        scale = self.module._get_weight_scale()
        if isinstance(scale, torch.Tensor):
            scale = scale.to(weight.device)

        # Dequantize INT4 -> float (per-row)
        # Weight is in rotated+permuted space; dequant gives the same.
        unpacked = unpack_int4(weight, self.module.in_features)
        weight_float = (unpacked.float() * scale.float()).reshape(self.module.out_features, self.module.in_features)

        # Apply LoRA patches in float space
        patches_list = self.patches.get(self.key, [])
        patched_weight_float = comfy.lora.calculate_weight(patches_list, weight_float, self.key)

        # Re-quantize to INT4 (per-row)
        # No re-rotation: weight is already in rotated space from the converter.
        in_features = patched_weight_float.shape[1]
        scales = calculate_scales(patched_weight_float, in_features)
        int_rounded = quantize_weights(patched_weight_float, scales, in_features)
        packed = pack_int4(int_rounded).to(torch.uint8)

        # Update module scale buffer
        if hasattr(self.module, "weight_scale"):
            if isinstance(scales, torch.Tensor):
                self.module.weight_scale = scales.to(weight.device)
            else:
                self.module.weight_scale = torch.tensor([scales], dtype=torch.float16, device=weight.device)

        return packed


class INT4ModelPatcher(comfy.model_patcher.ModelPatcher):
    """Custom ModelPatcher that intercepts patching for INT4 layers.

    Routes patching through a bake-in path (dequant -> patch -> requant).
    """

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False, return_weight=False, force_cast=False):
        if key not in self.patches and not force_cast:
            return super().patch_weight_to_device(key, device_to, inplace_update, return_weight, force_cast)

        module_path = key.rsplit('.', 1)[0]
        try:
            module = comfy.utils.get_attr(self.model, module_path)
        except AttributeError:
            module = None

        is_int4_module = hasattr(module, "_is_int4") and module._is_int4
        patches = self.patches.get(key, [])

        if is_int4_module and key.endswith(".bias"):
            return comfy.utils.get_attr(self.model, key) if return_weight else None

        if is_int4_module:
            # Dequant -> Patch -> Requant
            current_weight = comfy.utils.get_attr(self.model, key)
            scale = module._get_weight_scale()

            if device_to is None:
                device_to = current_weight.device

            # Use backup as source to prevent additive stacking
            if key not in self.backup:
                import collections
                BackupEntry = collections.namedtuple('Dimension', ['weight', 'inplace_update'])
                self.backup[key] = BackupEntry(
                    weight=current_weight.to(device=self.offload_device, copy=inplace_update),
                    inplace_update=inplace_update,
                )
                source_weight = current_weight
            else:
                source_weight = self.backup[key].weight

            # Dequantize to float (per-row)
            if isinstance(scale, torch.Tensor):
                scale = scale.to(device_to)

            unpacked = unpack_int4(source_weight.to(device_to), module.in_features)
            weight_float = (unpacked.float() * scale.float()).reshape(module.out_features, module.in_features)

            # Apply LoRA patches in float space
            patches_list = self.patches.get(key, [])
            patched_weight_float = comfy.lora.calculate_weight(patches_list, weight_float, key)

            # Re-quantize to INT4 (per-row)
            in_features = patched_weight_float.shape[1]
            scales = calculate_scales(patched_weight_float, in_features)
            int_rounded = quantize_weights(patched_weight_float, scales, in_features)
            packed = pack_int4(int_rounded).to(torch.uint8)

            # Move back to original device and store
            patched_weight = packed.to(current_weight.device)

            if return_weight:
                return patched_weight

            if inplace_update:
                current_weight.data.copy_(patched_weight)
            else:
                comfy.utils.set_attr(self.model, key, nn.Parameter(patched_weight, requires_grad=False))

            # Update scale buffer
            if hasattr(module, "weight_scale"):
                if isinstance(scales, torch.Tensor):
                    module.weight_scale = scales.to(current_weight.device)
                else:
                    module.weight_scale = torch.tensor([scales], dtype=torch.float16, device=current_weight.device)

            return

        # Non-INT4 module path
        return super().patch_weight_to_device(key, device_to, inplace_update, return_weight, force_cast)

    def load(self, *args, **kwargs):
        # Cleanup stale dynamic LoRA patches
        for name, module in self.model.named_modules():
            if hasattr(module, "lora_patches"):
                module.lora_patches = []

        res = super().load(*args, **kwargs) if hasattr(super(), "load") else None

        # For non-dynamic mode with active patches, install lowvram functions so
        # --fast dynamic_vram can stage patched weights through the VBAR.
        if res is not None and hasattr(self, "patches"):
            for name, module in self.model.named_modules():
                if hasattr(module, "_is_int4") and module._is_int4:
                    weight_key = name + ".weight" if name else "weight"
                    if weight_key in self.patches:
                        lowvram_patch = INT4LowVramPatch(
                            weight_key,
                            self.patches,
                            module,
                        )
                        if hasattr(module, "weight_lowvram_function"):
                            module.weight_lowvram_function = lowvram_patch

        return res

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            for name, module in self.model.named_modules():
                if hasattr(module, "lora_patches"):
                    module.lora_patches = []
        return super().unpatch_model(device_to, unpatch_weights)

    def clone(self, *args, **kwargs):
        """Clone this patcher, preserving INT4-aware patching in the copy.

        ComfyUI's ModelPatcher subclasses (e.g. ModelPatcherDynamic for
        --fast mode) each have their own clone() that returns an instance
        of *their own class*.  That clone would lose the INT4 bake-in
        logic unless we inject INT4ModelPatcher into the MRO.

        Strategy: temporarily swap self.__class__ to a dynamic subclass
        that inherits from both INT4ModelPatcher and the original class,
        call super().clone() (which copies the class pointer), then
        restore self.__class__ so we don't pollute the original object.
        """
        src_cls = self.__class__

        # Pure INT4ModelPatcher — no class surgery needed
        if src_cls is INT4ModelPatcher:
            return super().clone(*args, **kwargs)

        # Create a dynamic class: INT4ModelPatcher + original class
        if not issubclass(src_cls, INT4ModelPatcher):
            name = f"INT4_{src_cls.__name__}"
            dynamic_cls = type(name, (INT4ModelPatcher, src_cls), {})
        else:
            dynamic_cls = src_cls

        # Temporarily become the dynamic class so super().clone() copies it
        self.__class__ = dynamic_cls

        # Provide a fallback for non-dynamic delegates
        if getattr(self, "cached_patcher_init", None) is None:
            self.cached_patcher_init = (lambda *a, **kw: self, ())

        n = super().clone(*args, **kwargs)

        # In --fast dynamic_vram mode, src_cls is ModelPatcherDynamic.
        # Fix up missing delegate attribute.
        try:
            import comfy.model_patcher as _cmp
            if isinstance(n, _cmp.ModelPatcherDynamic) and not hasattr(n, "non_dynamic_delegate_model"):
                n.non_dynamic_delegate_model = None
        except Exception:
            pass

        # Handle disable_dynamic: when ComfyUI disables dynamic mode, the
        # clone's class may not include INT4ModelPatcher.  Re-inject it.
        disable_dyn = kwargs.get("disable_dynamic", False)
        if len(args) > 0:
            disable_dyn = args[0]

        if disable_dyn and not issubclass(n.__class__, INT4ModelPatcher):
            new_cls = type(f"INT4_{n.__class__.__name__}", (INT4ModelPatcher, n.__class__), {})
            n.__class__ = new_cls

        # Restore original class on self
        self.__class__ = src_cls
        return n


def _requantize_int8(w_float: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-quantize a float weight to INT8 with per-row scales."""
    s_w = w_float.abs().amax(dim=1).to(torch.float32).clamp(min=1e-8) / 127.0
    q = w_float / s_w.unsqueeze(1)
    w_int8 = (torch.where(q >= 0, q + 0.5, q - 0.5)
              .clamp(-128, 127).to(torch.int8))
    return w_int8, s_w


class INT8LowVramPatch:
    """Low-VRAM patch that dequantizes, applies LoRA, and re-quantizes to INT8."""
    is_lowvram_patch = True

    def __init__(self, key, patches, module):
        self.key = key
        self.patches = patches
        self.module = module
        self.prepared_patches = None

    def memory_required(self):
        if not _use_new_prefetch:
            return 0
        counter = [0]
        for patch in self.patches[self.key]:
            comfy.lora.prefetch_prepared_value(patch[1], counter, None, None, False)
        return counter[0]

    def prepare(self, *args, **kwargs):
        if _use_new_prefetch:
            destination = args[0] if len(args) > 0 else kwargs.get("destination")
            stream = args[1] if len(args) > 1 else kwargs.get("stream")
            copy = args[2] if len(args) > 2 else kwargs.get("copy", True)
            commit = args[3] if len(args) > 3 else kwargs.get("commit", True)

            counter = [0]
            prepared_patches = [
                (patch[0], comfy.lora.prefetch_prepared_value(patch[1], counter, destination, stream, copy), patch[2], patch[3], patch[4])
                for patch in self.patches[self.key]
            ]
            if commit:
                self.prepared_patches = prepared_patches
            return prepared_patches
        else:
            allocate_buffer = args[0] if len(args) > 0 else kwargs.get("allocate_buffer")
            stream = args[1] if len(args) > 1 else kwargs.get("stream")

            self.prepared_patches = [
                (patch[0], comfy.lora.prefetch_prepared_value(patch[1], allocate_buffer, stream), patch[2], patch[3], patch[4])
                for patch in self.patches[self.key]
            ]
            return self.prepared_patches

    def clear_prepared(self):
        self.prepared_patches = None

    def __call__(self, weight):
        patches = self.prepared_patches if self.prepared_patches is not None else self.patches[self.key]

        scale = self.module._get_weight_scale()
        if isinstance(scale, torch.Tensor):
            scale = scale.to(weight.device)

        weight_float = weight.float() * scale.reshape(-1, 1)
        patched_weight_float = comfy.lora.calculate_weight(patches, weight_float, self.key)

        w_int8, s_w = _requantize_int8(patched_weight_float)
        if hasattr(self.module, "weight_scale"):
            self.module.weight_scale = s_w.to(weight.device)
        return w_int8


class INT8ModelPatcher(comfy.model_patcher.ModelPatcher):
    """Custom ModelPatcher that intercepts patching for INT8 layers.

    Routes patching through a bake-in path (dequant -> patch -> requant).
    """

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False, return_weight=False, force_cast=False):
        if key not in self.patches and not force_cast:
            return super().patch_weight_to_device(key, device_to, inplace_update, return_weight, force_cast)

        module_path = key.rsplit('.', 1)[0]
        try:
            module = comfy.utils.get_attr(self.model, module_path)
        except AttributeError:
            module = None

        is_int8_module = hasattr(module, "_is_int8") and module._is_int8
        patches = self.patches.get(key, [])

        if is_int8_module and key.endswith(".bias"):
            return comfy.utils.get_attr(self.model, key) if return_weight else None

        if is_int8_module:
            current_weight = comfy.utils.get_attr(self.model, key)
            scale = module._get_weight_scale()

            if device_to is None:
                device_to = current_weight.device

            if key not in self.backup:
                import collections
                BackupEntry = collections.namedtuple('Dimension', ['weight', 'inplace_update'])
                self.backup[key] = BackupEntry(
                    weight=current_weight.to(device=self.offload_device, copy=inplace_update),
                    inplace_update=inplace_update,
                )
                source_weight = current_weight
            else:
                source_weight = self.backup[key].weight

            if isinstance(scale, torch.Tensor):
                scale = scale.to(device_to)

            weight_float = source_weight.to(device_to).float() * scale.reshape(-1, 1)
            patches_list = self.patches.get(key, [])
            patched_weight_float = comfy.lora.calculate_weight(patches_list, weight_float, key)

            w_int8, s_w = _requantize_int8(patched_weight_float)
            patched_weight = w_int8.to(current_weight.device)

            if return_weight:
                return patched_weight

            if inplace_update:
                current_weight.data.copy_(patched_weight)
            else:
                comfy.utils.set_attr(self.model, key, nn.Parameter(patched_weight, requires_grad=False))

            if hasattr(module, "weight_scale"):
                module.weight_scale = s_w.to(current_weight.device)

            return

        return super().patch_weight_to_device(key, device_to, inplace_update, return_weight, force_cast)

    def load(self, *args, **kwargs):
        for name, module in self.model.named_modules():
            if hasattr(module, "lora_patches"):
                module.lora_patches = []

        res = super().load(*args, **kwargs) if hasattr(super(), "load") else None

        if res is not None and hasattr(self, "patches"):
            for name, module in self.model.named_modules():
                if hasattr(module, "_is_int8") and module._is_int8:
                    weight_key = name + ".weight" if name else "weight"
                    if weight_key in self.patches:
                        lowvram_patch = INT8LowVramPatch(
                            weight_key,
                            self.patches,
                            module,
                        )
                        if hasattr(module, "weight_lowvram_function"):
                            module.weight_lowvram_function = lowvram_patch

        return res

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            for name, module in self.model.named_modules():
                if hasattr(module, "lora_patches"):
                    module.lora_patches = []
        return super().unpatch_model(device_to, unpatch_weights)

    def clone(self, *args, **kwargs):
        """Clone this patcher, preserving INT8-aware patching in the copy.

        Same class-surgery strategy as INT4ModelPatcher.clone() — see its
        docstring for the rationale.
        """
        src_cls = self.__class__

        if src_cls is INT8ModelPatcher:
            return super().clone(*args, **kwargs)

        if not issubclass(src_cls, INT8ModelPatcher):
            name = f"INT8_{src_cls.__name__}"
            dynamic_cls = type(name, (INT8ModelPatcher, src_cls), {})
        else:
            dynamic_cls = src_cls

        self.__class__ = dynamic_cls

        if getattr(self, "cached_patcher_init", None) is None:
            self.cached_patcher_init = (lambda *a, **kw: self, ())

        n = super().clone(*args, **kwargs)

        try:
            import comfy.model_patcher as _cmp
            if isinstance(n, _cmp.ModelPatcherDynamic) and not hasattr(n, "non_dynamic_delegate_model"):
                n.non_dynamic_delegate_model = None
        except Exception:
            pass

        disable_dyn = kwargs.get("disable_dynamic", False)
        if len(args) > 0:
            disable_dyn = args[0]

        if disable_dyn and not issubclass(n.__class__, INT8ModelPatcher):
            new_cls = type(f"INT8_{n.__class__.__name__}", (INT8ModelPatcher, n.__class__), {})
            n.__class__ = new_cls

        self.__class__ = src_cls
        return n
