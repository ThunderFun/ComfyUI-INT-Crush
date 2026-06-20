"""ComfyUI loader nodes for INT-Crush quantized models.

This module re-exports node mappings for ComfyUI discovery.
Actual implementation lives in:
  - _triton_runtime.py: Triton kernel detection
  - _ops.py: ops factory and IntCrushOps Linear class
  - _lora.py: LoRA buffer attach/detach
  - _loaders.py: ComfyUI node classes
"""

from ._loaders import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
