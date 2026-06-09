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
| `model_type` | `flux` / `wan` / `zimage` / `chroma` / `default` |
| `rot_size` | `0`/`16`/`64`/`256` (auto-detected from metadata) |
| `use_pytorch` | `True` (default): PyTorch path. `False`: Triton + cuBLAS |

**INT8 UNet Loader (INT-Crush)** — `SimpleINT8UNetLoader`

| Input | Description |
|-------|-------------|
| `unet_name` | Quantized `.safetensors` from `models/diffusion_models/` |
| `model_type` | `flux` / `wan` / `zimage` / `chroma` / `default` |
| `rot_size` | `0`/`16`/`64`/`256` (auto-detected from metadata) |
| `debug_shapes` | Print shape diagnostics |

## Workflow

```bash
# 1. Quantize (from int_crush_converter/)
python -m converter.cli -i model.safetensors -o ./out --rot-size 256 --int-bits 4

# 2. Copy to ComfyUI
cp ./out/model.safetensors /path/to/ComfyUI/models/diffusion_models/

# 3. Load with INT4/INT8 UNet Loader node (auto-detects rot_size from metadata)
```

## Inference

**INT4** — controlled by `use_pytorch`:
- `False`: Triton INT4 unpack → float16 → cuBLAS GEMM
- `True` (default): PyTorch unpack → float → `F.linear`

**INT8** — auto-selects:
- Batch ≤ 16: PyTorch dequant → `F.linear`
- Batch > 16, Triton available: W8A8 GEMM with dynamic per-token activation quantization
- Batch > 16, no Triton: PyTorch fallback

## LoRA

`INT4ModelPatcher` / `INT8ModelPatcher` route LoRA through dequant → patch → requant. Works automatically, no manual handling needed. Low-VRAM supported via `INT4LowVramPatch` / `INT8LowVramPatch`.

## Weight Format

**INT4:** `<name>.weight` (`uint8`, 2 per byte) + `<name>.weight_scale` (`fp16`, `[out, num_groups]`) + optional `<name>.perm`

**INT8:** `<name>.weight` (`int8`) + `<name>.weight_scale` (`fp16`, `[out, 1]`) + optional `<name>.perm`

## Forward Pass

1. Rotate activations (Hadamard, if `rot_size > 0`)
2. Pad to weight `in_features`
3. Apply PermuQuant permutation (if present)
4. Dequantize weights → `F.linear`

## Notes

- ~4× compression (INT4), ~2× (INT8) vs fp16.
- Padded layers auto-detected and fixed.
- `model_type` is informational, doesn't affect loading.
