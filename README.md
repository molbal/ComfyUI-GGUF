# ComfyUI-GGUF
GGUF Quantization support for native ComfyUI models including the custom Q8_CR

> [!NOTE]  
> This is a fork of the original nodes, updated to support loading Ideogram 4 GGUFs and Krea 2 GGUFs. 
> To use this maintained fork, clone `https://github.com/molbal/ComfyUI-GGUF`.

While quantization wasn't feasible for regular UNET models (conv2d), transformer/DiT models such as flux seem less affected by quantization. This allows running it in much lower bits per weight variable bitrate quants on low-end GPUs. For further VRAM savings, a node to load a quantized version of the T5 text encoder is also included.

## Installation

> [!IMPORTANT]  
> Make sure your ComfyUI is on v0.27.0 or later.

To install the custom node normally, git clone this repository into your custom nodes folder (`ComfyUI/custom_nodes`) and install the only dependency for inference (`pip install --upgrade gguf`)

```
git clone https://github.com/molbal/ComfyUI-GGUF
```

To install the custom node on a standalone ComfyUI release, open a CMD inside the "ComfyUI_windows_portable" folder (where your `run_nvidia_gpu.bat` file is) and use the following commands:

```
git clone https://github.com/molbal/ComfyUI-GGUF ComfyUI/custom_nodes/ComfyUI-GGUF
.\python_embeded\python.exe -s -m pip install -r .\ComfyUI\custom_nodes\ComfyUI-GGUF\requirements.txt
```

On MacOS sequoia, torch 2.4.1 seems to be required, as 2.6.X nightly versions cause a "M1 buffer is not large enough" error. See [this issue](https://github.com/city96/ComfyUI-GGUF/issues/107) for more information/workarounds.

## Usage

Simply use the GGUF Unet loader found under the `bootleg` category. Place the .gguf model files in your `ComfyUI/models/unet` folder.

LoRA loading is experimental but it should work with just the built-in LoRA loader node(s).

Pre-quantized models (🍴 icon on ones added by this fork):

- [flux1-dev GGUF](https://huggingface.co/city96/FLUX.1-dev-gguf)
- [flux1-schnell GGUF](https://huggingface.co/city96/FLUX.1-schnell-gguf)
- [stable-diffusion-3.5-large GGUF](https://huggingface.co/city96/stable-diffusion-3.5-large-gguf)
- [stable-diffusion-3.5-large-turbo GGUF](https://huggingface.co/city96/stable-diffusion-3.5-large-turbo-gguf)
- [Krea 2 (Both Turbo and Raw)](https://huggingface.co/molbal/krea2-gguf) 🍴
- [Ideogram 4](https://huggingface.co/molbal/ideogram-4-gguf) 🍴
- [MiniMax H3](https://huggingface.co/molbal/MiniMax-H3-GGUF) 🍴


> [!IMPORTANT]  
> Please note, that this fork does not support _K quants on diffusion models, only on text encoders. They may or may not load, but inference speed may be very slow. There may be other forks, or other custom nodes with better support for these quantization types.

Initial support for quantizing T5 has also been added recently, these can be used using the various `*CLIPLoader (gguf)` nodes which can be used inplace of the regular ones. For the CLIP model, use whatever model you were using before for CLIP. The loader can handle both types of files - `gguf` and regular `safetensors`/`bin`.

- [t5_v1.1-xxl GGUF](https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf)
- [Qwen3-VL-4B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF) 🍴
- [Qwen3-VL-32B-Instruct-GGUF](https://huggingface.co/unsloth/Qwen3-VL-32B-Instruct-GGUF) 🍴

See the instructions in the [tools](https://github.com/city96/ComfyUI-GGUF/tree/main/tools) folder for how to create your own quants.

## Converting Krea 2, Ideogram 4, and Minimax M3 models

The converter detects supported Krea 2, Ideogram 4, and native Minimax M3
(`minimax_h3`) checkpoints directly.
Provide an existing `.safetensors`, `.ckpt`, `.pt`, `.pth`, or `.bin` diffusion
model file; no model-specific conversion script is required.

Run these commands from the `ComfyUI-GGUF` directory, replacing the source and
destination paths with your model names:

```bash
# Compact standard GGUF
python tools/convert.py --src /path/to/krea2_or_ideogram.safetensors \
  --dst /path/to/model-Q4_0.gguf --quant-type Q4_0

# Higher-quality standard GGUF
python tools/convert.py --src /path/to/krea2_or_ideogram.safetensors \
  --dst /path/to/model-Q8_0.gguf --quant-type Q8_0

# Recommended for RTX 30-series NVIDIA GPUs
python tools/convert.py --src /path/to/krea2_or_ideogram.safetensors \
  --dst /path/to/model-Q8_CR.gguf --quant-type Q8_CR
```

For a native Minimax M3 checkpoint, use the same command:

```bash
python tools/convert.py --src /path/to/minimax_m3.safetensors \
  --dst /path/to/minimax_m3-Q8_CR.gguf --quant-type Q8_CR
```

For the portable Windows distribution, use its embedded Python executable:

```bat
.\python_embeded\python.exe .\ComfyUI\custom_nodes\ComfyUI-GGUF\tools\convert.py ^
  --src C:\path\to\krea2_or_ideogram.safetensors ^
  --dst C:\path\to\model-Q8_CR.gguf --quant-type Q8_CR
```

Place the resulting GGUF in `ComfyUI/models/unet` or
`ComfyUI/models/diffusion_models`, then load it with **Unet Loader (GGUF)**.

## Supported conversion formats

The standard GGUF formats use the package's normal GGML loader path. `Q8_CR`
is a custom ComfyUI-native INT8 layout for eligible Linear weights.

| Format | Storage / execution | Recommended use |
| --- | --- | --- |
| `F16` | FP16 GGUF | Maximum compatibility with half-precision storage. |
| `BF16` | BF16 GGUF | Preserve BF16 source models where the target supports BF16. |
| `Q8_0` | Standard GGML 8-bit quantization | Excellent general-quality 8-bit GGUF; portable and straightforward to offload. |
| `Q5_1` | Standard GGML 5-bit quantization | Lower storage with a quality-oriented 5-bit format. |
| `Q5_0` | Standard GGML 5-bit quantization | Lower storage alternative to `Q5_1`. |
| `Q4_1` | Standard GGML 4-bit quantization | Smaller files when VRAM or RAM is constrained. |
| `Q4_0` | Standard GGML 4-bit quantization | Smallest supported standard diffusion-model format; expect the largest quality trade-off. |
| `Q8_CR` | Per-row INT8 ConvRot through ComfyUI native ops | **Maintainer recommendation for NVIDIA RTX 30-series systems.** |


> [!WARNING]
> `_K` quant formats are not supported for diffusion models; they are supported only for text encoders.

## Native weight-only quantization

The converter supports one custom global quantization mode for DiT/transformer
UNets: `Q8_CR`.

### What Q8_CR is, why it exists, and how it works

`Q8_CR` is an INT8 weight-only format designed for transformer-style diffusion
models. Its purpose is to reduce GGUF model storage and VRAM pressure while
preserving the fast native INT8 Linear operations available on supported NVIDIA
GPUs. It is especially useful when the full diffusion model does not fit in
VRAM and ComfyUI needs to offload weights to CPU memory.

During conversion, the converter:

1. Selects eligible 2-D Linear weights. One-dimensional tensors, small
   tensors, and architecture-designated sensitive tensors stay FP32; Conv2d
   weights stay FP16.
2. Applies the compatible ConvRot/Hadamard rotation to each eligible weight
   matrix.
3. Quantizes the rotated weights to INT8 using an FP32 scale for every output
   row.
4. Stores the INT8 payload, scales, and ConvRot metadata in the GGUF file.

During loading, the GGUF loader recognizes this metadata and passes the raw
INT8 weights and row scales to ComfyUI's `TensorWiseINT8Layout`. On supported
CUDA systems, ComfyUI executes the native INT8/ConvRot Linear path directly;
it does not first expand the weight matrix to FP16 or BF16. The ordinary GGUF
container still provides memory-mapped loading and CPU offload behavior.

- `Q8_CR` stores eligible 2-D Linear weights as per-row INT8 ConvRot. It uses
  ComfyUI's native `TensorWiseINT8Layout` path, so weights remain INT8 during
  inference.

Q8_CR keeps 1-D, small, and architecture-designated high-precision tensors
in FP32. Conv2d weights remain FP16 because these modes accelerate Linear
matrix multiplication only.

### Maintainer recommendation: Q8_CR on RTX 30-series

For NVIDIA RTX 30-series systems, the maintainer recommends `Q8_CR` for Krea 2
and Ideogram 4. It combines:

- Fast native INT8 operations on these GPUs through ComfyUI's ConvRot backend.
- GGUF's convenient CPU offload and memory-mapped model storage behavior.
- The generally excellent image quality expected from 8-bit GGUF
  quantization, while retaining selected sensitive tensors in higher precision.

Use `Q8_0` instead when you need the conventional portable GGML 8-bit format.
Use `Q4_0` primarily when the smaller model footprint matters more than
quality or sampling speed.

### Q8_CR platform support

Q8_CR does not require CUDA. It uses ComfyUI's `comfy_kitchen` layout backend:

- NVIDIA CUDA uses ComfyUI's optimized native INT8 backend when available.
- Linux and non-CUDA environments use the `comfy_kitchen` eager backend.
- CPU Q8_CR loading and inference are supported, but naturally slower than
  optimized CUDA inference.

All GGUF UNET and CLIP loader nodes, including Dynamic VRAM and multi-CLIP
variants, report their tensor-loading progress through ComfyUI's global
progress bar. A multi-CLIP loader uses one bar for every selected file.

### Target-size quantization

Use `tools/convert.py --max-size-mb <MiB>` to create the best supported mixed
quantization below a maximum output size. The converter starts with core 2-D
Linear weights in Q8_CR while preserving 1-D and architecture-sensitive tensors
in FP32. It then changes core matrices closest to the model's center to Q4_0
until the target is met, retaining the beginning and end in Q8_CR for as long as
possible. If every Q4_0-compatible core matrix is already Q4_0, ordinary 1-D
tensors are reduced to BF16; protected tensors remain FP32.

`Q4_0` is the smallest supported core quantization. A target below the minimum
attainable size raises an error that reports that minimum; Q3 and lower are not
used. The **Targeted Quantization (GGUF)** ComfyUI node exposes the same source,
destination, quantization, target-size, and overwrite options, reports loading
and conversion progress, and outputs both the GGUF path and output details.

Reconvert any `Q8_CR` GGUF created before ConvRot weights were marked as
pre-rotated. Older files load safely with native non-rotated INT8 instead.
