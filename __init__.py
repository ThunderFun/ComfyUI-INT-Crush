"""INT-Crush: INT4/INT8 quantized inference for ComfyUI diffusion models.

Exports ConvLinear4bit for standalone use, registers INT4/INT8 layout
classes with ComfyUI's QuantizedTensor dispatch, and exposes loader
node mappings for the ComfyUI graph.
"""

import logging

log = logging.getLogger(__name__)

__all__: list[str] = []

try:
    from .convlinear import ConvLinear4bit
    __all__.append("ConvLinear4bit")
except ImportError:
    ConvLinear4bit = None

# Register INT4/INT8 layout classes so QuantizedTensor can dispatch to them.
try:
    from .quant_layout import register_intcrush_layouts
    register_intcrush_layouts()
except Exception:
    log.warning("INT-Crush: layout registration failed", exc_info=True)

# Expose ComfyUI node mappings so the host can discover loader nodes.
try:
    from .comfy_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    __all__.extend(["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"])
except Exception:
    log.warning("INT-Crush: node import failed", exc_info=True)
