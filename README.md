# ComfyUI-INT-Crush

ComfyUI loader for INT-Crush quantized models (INT4 + INT8). Supports models quantized with [INT-Crush Converter](https://github.com/ThunderFun/int_crush_converter)

> **⚠️ WARNING:** This code has not been thoroughly tested.

*Developed with AI assistance.*

## Installation

Clone this repository into your ComfyUI `custom_nodes/` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ThunderFun/ComfyUI-INT-Crush.git
```

## Nodes

**INT4 UNet Loader (INT-Crush)** — `SimpleINT4UNetLoader`

| Input | Description |
|-------|-------------|
| `unet_name` | Quantized `.safetensors` from `models/diffusion_models/` |
| `rot_size` | `0`/`16`/`64`/`256`/`1024`/`4096` (auto-detected from metadata) |

**INT8 UNet Loader (INT-Crush)** — `SimpleINT8UNetLoader`

| Input | Description |
|-------|-------------|
| `unet_name` | Quantized `.safetensors` from `models/diffusion_models/` |
| `rot_size` | `0`/`16`/`64`/`256`/`1024`/`4096` (auto-detected from metadata) |

**LoRA Loader (INT-Crush)** — `IntCrushLoRALoader`

| Input | Description |
|-------|-------------|
| `model` | MODEL output from an INT4/INT8 loader |
| `lora_name` | LoRA file from `models/loras/` |
| `strength` | LoRA strength (default 1.0) |

**LoRA Unloader (INT-Crush)** — `IntCrushLoRAUnloader`

| Input | Description |
|-------|-------------|
| `model` | MODEL with INT-Crush LoRA buffers to remove |

## Workflow

```bash
# 1. Quantize (from int_crush_converter/)
python -m converter.cli -i model.safetensors -o ./out --rot-size 256 --int-bits 4

# 2. Copy to ComfyUI
cp ./out/model.safetensors /path/to/ComfyUI/models/diffusion_models/

# 3. Load with INT4/INT8 UNet Loader node (auto-detects rot_size from metadata)
```

## Inference

The inference path is automatically selected based on Triton kernel availability and batch size:

**INT4** — three paths, auto-selected:
- W4A8 (fastest): Triton INT4→INT8 unpack → dynamic-quantize activations → fused INT8 GEMM + dequant. Requires all Triton kernels and `rot_size > 0`.
- W4A16: Triton INT4→float16 unpack → cuBLAS GEMM. Used when `rot_size == 0` or W4A8 kernels unavailable.
- PyTorch fallback: full dequant to float → `F.linear`. Used when `use_pytorch=True` or no Triton.

**INT8** — auto-selects by batch size:
- Batch ≤ 16: PyTorch dequant → `F.linear`
- Batch > 16, Triton available: dynamic per-token INT8 quantization → fused INT8 GEMM + dequant
- Batch > 16, no Triton: PyTorch fallback

## LoRA

`IntCrushLoRALoader` attaches LoRA as a residual buffer in **unrotated** activation space.
This avoids corrupting the quantized weight's Hadamard rotation by applying the LoRA
before any rotation or smoothing. The low-rank matrices (A, B) are stored on CPU and
moved to GPU on first forward. Standard ComfyUI LoRA patching is **not** compatible
with INT-Crush models — use the INT-Crush LoRA nodes instead.

## Weight Format

**INT4:** `<name>.weight` (`uint8`, 2 per byte) + `<name>.weight_scale` (`fp16`, `[out, num_groups]`) + optional `<name>.perm`

**INT8:** `<name>.weight` (`int8`) + `<name>.weight_scale` (`fp16`, `[out, 1]`) + optional `<name>.perm`

## Forward Pass

1. SmoothRot (if SmoothRot factors present) — divide activations by scale factors before rotation
2. Rotate activations (Hadamard, if `rot_size > 0`)
3. Pad to weight `in_features`
4. Apply PermuQuant permutation (if present)
5. Quantize weights and activations, then GEMM (or dequant → `F.linear` for fallback)

## Notes

- ~4× compression (INT4), ~2× (INT8) vs fp16.
- Padded layers auto-detected and fixed.
- `rot_size` is auto-detected from safetensors metadata (`int_crush.format_version` / `int_crush.rot_size`).
