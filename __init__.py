"""INT-Crush inference loader."""

import logging
import torch

try:
    from .convlinear import ConvLinear4bit
    __all__ = ["ConvLinear4bit"]
except ImportError:
    ConvLinear4bit = None
    __all__ = []


def _register_layouts():
    """Register INT4/INT8 layout classes and QUANT_ALGOS entries with ComfyUI.

    Called at import time so ComfyUI can recognise and dequantize INT-Crush
    weight tensors from safetensors checkpoints.  Warns silently if the
    ComfyUI quantization system is unavailable.
    """
    try:
        from comfy.quant_ops import QUANT_ALGOS, register_layout_class, QuantizedLayout

        class Int4TensorwiseLayout(QuantizedLayout):
            """Minimal layout class to satisfy ComfyUI's registry requirements."""

            class Params:
                def __init__(self, scale=None, orig_dtype=None, orig_shape=None, **kwargs):
                    self.scale = scale
                    self.orig_dtype = orig_dtype
                    self.orig_shape = orig_shape

                def clone(self):
                    return Int4TensorwiseLayout.Params(
                        scale=self.scale.clone() if isinstance(self.scale, torch.Tensor) else self.scale,
                        orig_dtype=self.orig_dtype,
                        orig_shape=self.orig_shape,
                    )

            @classmethod
            def state_dict_tensors(cls, qdata, params):
                return {"": qdata, "weight_scale": params.scale}

            @classmethod
            def dequantize(cls, qdata, params):
                return qdata.float() * params.scale

        register_layout_class("Int4TensorwiseLayout", Int4TensorwiseLayout)

        QUANT_ALGOS.setdefault(
            "int4_tensorwise",
            {
                "storage_t": torch.uint8,
                "parameters": {"weight_scale"},
                "comfy_tensor_layout": "Int4TensorwiseLayout",
            }
        )

        class Int8TensorwiseLayout(QuantizedLayout):
            """Minimal layout class for INT8 quantized weights."""

            class Params:
                def __init__(self, scale=None, orig_dtype=None, orig_shape=None, **kwargs):
                    self.scale = scale
                    self.orig_dtype = orig_dtype
                    self.orig_shape = orig_shape

                def clone(self):
                    return Int8TensorwiseLayout.Params(
                        scale=self.scale.clone() if isinstance(self.scale, torch.Tensor) else self.scale,
                        orig_dtype=self.orig_dtype,
                        orig_shape=self.orig_shape,
                    )

            @classmethod
            def state_dict_tensors(cls, qdata, params):
                return {"": qdata, "weight_scale": params.scale}

            @classmethod
            def dequantize(cls, qdata, params):
                return qdata.float() * params.scale

        register_layout_class("Int8TensorwiseLayout", Int8TensorwiseLayout)

        QUANT_ALGOS.setdefault(
            "int8_tensorwise",
            {
                "storage_t": torch.int8,
                "parameters": {"weight_scale"},
                "comfy_tensor_layout": "Int8TensorwiseLayout",
            }
        )

    except ImportError:
        logging.warning("INT-Crush: ComfyUI Quantization system not found (Update ComfyUI?)")
    except Exception as e:
        logging.error(f"INT-Crush: Failed to register layouts: {e}")


_register_layouts()

try:
    from .comfy_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    __all__ += ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
except (ImportError, RuntimeError):  # noqa: BLE001
    pass

try:
    from .comfy_nodes import IntCrushInt4Ops
    __all__ += ["IntCrushInt4Ops"]
except (ImportError, RuntimeError):  # noqa: BLE001
    IntCrushInt4Ops = None

try:
    from .comfy_nodes import IntCrushInt8Ops
    __all__ += ["IntCrushInt8Ops"]
except (ImportError, RuntimeError):  # noqa: BLE001
    IntCrushInt8Ops = None
