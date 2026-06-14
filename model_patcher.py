"""ModelPatcher compatibility shim for INT-Crush.

With QuantizedTensor integration, ComfyUI's built-in ModelPatcher handles
all weight patching, LoRA, offloading, and DynamicVRAM natively.  This
module provides backward-compatible aliases only.
"""

import comfy.model_patcher

# These are plain ModelPatcher now — QuantizedTensor + mixed_precision_ops
# handle LoRA dequant→patch→requant, VBAR demand-paging, and async offload
# through ComfyUI's standard cast_bias_weight pipeline.
INT4ModelPatcher = comfy.model_patcher.ModelPatcher
INT8ModelPatcher = comfy.model_patcher.ModelPatcher
