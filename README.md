
# ComfyUI-GGUF

GGUF Quantization support for native ComfyUI models including the custom Q8_CR. 

> [!NOTE]  
> This is a fork of the original nodes, updated to support loading Ideogram 4 GGUFs and Krea 2 GGUFs. 
> To use this maintained fork, clone `https://github.com/molbal/ComfyUI-GGUF`.

While quantization was previously unfeasible for regular UNET models (conv2d), transformer/DiT models such as flux are less affected by quantization. This allows running them in lower bits per weight variable bitrate quants on GPUs with less VRAM.

More details on how to use it, pre-converted models, and sample workflows are here: [Documentation](https://molbal.github.io/gguf/ecosystem/using-the-custom-nodes.html)

For technical details on the custom `Q8_CR`  and `Q4_CR` formats , memory-mapped loading, please see [ARCHITECTURE.md](ARCHITECTURE.md).

## Installation

> [!IMPORTANT]  
> Make sure your ComfyUI is on v0.27.0 or later.

To install the custom node normally, git clone this repository into your custom nodes folder (`ComfyUI/custom_nodes`) and restart ComfyUI.

```bash
git clone https://github.com/molbal/ComfyUI-GGUF
```
  


## Usage  
  
Simply use the GGUF Unet loader found under the `bootleg` category. Place the .gguf model files in your `ComfyUI/models/unet` folder.  

Pre-quantized models (🍴 icon on ones added by this fork):  
  
- [flux1-dev GGUF](https://huggingface.co/city96/FLUX.1-dev-gguf)  
- [flux1-schnell GGUF](https://huggingface.co/city96/FLUX.1-schnell-gguf)  
- [stable-diffusion-3.5-large GGUF](https://huggingface.co/city96/stable-diffusion-3.5-large-gguf)  
- [stable-diffusion-3.5-large-turbo GGUF](https://huggingface.co/city96/stable-diffusion-3.5-large-turbo-gguf)  
- [Krea 2 (Both Turbo and Raw)](https://huggingface.co/molbal/krea2-gguf) 🍴  
- [Ideogram 4](https://huggingface.co/molbal/ideogram-4-gguf) 🍴  
- [MiniMax H3](https://huggingface.co/molbal/MiniMax-H3-GGUF) 🍴  
- [MiniMax Music3](https://huggingface.co/molbal/Minimax-Music3-GGUF) 🍴  
- [LTX 2.5](https://huggingface.co/molbal/LTX-2.5-GGUF) 🍴  
- [Qwen Image 2.1](https://huggingface.co/molbal/Qwen-Image-2.1-GGUF) 🍴  

### Qwen Image 2.1 FP16 fallback

The GGUF UNet loaders automatically use FP16 model and manual-cast dtypes for
Qwen Image 2.1 when the active device lacks native BF16 but ComfyUI reports
that FP16 is supported. BF16-capable devices keep the existing BF16 behavior.
This fallback applies only to Qwen Image 2.1 GGUF models; regular checkpoint
loading is unchanged, and explicit ComfyUI FP32, BF16, or FP8 UNet overrides
are respected. Sensitive one-dimensional BF16 tensors remain FP32.

ComfyUI currently lists only BF16 and FP32 as supported inference dtypes for
Qwen Image 2.1, so the GGUF fallback intentionally overrides that restriction.
FP16 may change output numerics or stability; validate it with your model and
workflow. If ComfyUI considers FP16 unsupported on the device, it retains the
existing fallback instead. Standard GGML quantized weights are still
materialized for PyTorch compute; this change does not add native low-bit
inference.

Initial support for quantizing T5 has also been added, these can be used using the various `*CLIPLoader (gguf)` nodes which can be used inplace of the regular ones. For the CLIP model, use whatever model you were using before for CLIP. The loader can handle both types of files - `gguf` and regular `safetensors`/`bin`.  
  
- [t5_v1.1-xxl GGUF](https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf)  
- [Qwen3-VL-4B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF) 🍴  
- [Qwen3-VL-32B-Instruct-GGUF](https://huggingface.co/unsloth/Qwen3-VL-32B-Instruct-GGUF) 🍴  
- [Qwen3-VL-32B-Instruct-MiniMax-H3 pruned GGUFs](https://huggingface.co/nif0/Qwen3-VL-32B-Instruct-MiniMax-H3-GGUF) 🍴  
- [Qwen3.5 GGUF](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF) text encoders (0.8B, 2B, 4B, 9B, and 27B) with a ComfyUI build containing Qwen3.5 TE support. Place the matching `mmproj-*.gguf` beside the text encoder for image conditioning; text-only workflows do not need it. 🍴  
- [Gemma 4 GGUF](https://huggingface.co/unsloth/gemma-4-E4B-it-qat-GGUF) text encoders (E2B, E4B, 12B, and 31B) with ComfyUI v0.30.0 or later. 🍴

### Qwen3-VL 4B text-only Q8_CR

Currently, conversion supports only the official Comfy-Org BF16 checkpoint
[`qwen3vl_4b_bf16.safetensors`](https://huggingface.co/Comfy-Org/Qwen3-VL/blob/main/text_encoders/qwen3vl_4b_bf16.safetensors)
and validates the language tensor names/shapes, visual-tensor count and
signature shapes, and BF16 dtypes. It produces a text-only Q8_CR GGUF by
deliberately dropping the vision tower; the result cannot process
images or use an MMProj sidecar, and non-Q8_CR conversion settings are rejected.

From the ComfyUI-GGUF repository root using the ComfyUI Python environment, set
`$comfyRoot` to your ComfyUI installation and run:

```powershell
$comfyRoot = "D:\path\to\ComfyUI"
python tools\convert.py --src "$comfyRoot\models\text_encoders\qwen3vl_4b_bf16.safetensors" --dst "$comfyRoot\models\text_encoders\qwen3vl_4b_bf16-Q8_CR.gguf" --quant-type Q8_CR --streamed
```

Load the resulting file by itself with **CLIPLoader (GGUF, Dynamic VRAM)**.
Static and legacy GGUF CLIP loaders reject Q8_CR text encoders. The Dynamic
VRAM runtime is required; within it, native INT8 TensorWise execution is
selected only on CUDA Ampere or newer GPUs (compute capability 8.0+) when
ComfyUI exposes its TensorWiseINT8 layout and a working `comfy_kitchen` kernel.
If Dynamic VRAM is available but the GPU/kernel cannot use that route—or a
native kernel fails—the loader restores the weights for dequantized
higher-precision execution. This fallback is not native INT8. The ComfyUI
checkout must include the Qwen3-VL 4B text-encoder runtime (commit
[`fc964047`](https://github.com/comfyanonymous/ComfyUI/commit/fc964047e7f6e837eca776e7c34706c04690ecfd)
or later). The native INT8 route additionally depends on the compatible
Dynamic VRAM / TensorWiseINT8 runtime and CUDA kernel being installed.

An opt-in BF16-vs-Q8_CR benchmark is available in this package:

```powershell
python tools\benchmark_qwen3vl_text_encoder.py --source "$comfyRoot\models\text_encoders\qwen3vl_4b_bf16.safetensors" --quantized "$comfyRoot\models\text_encoders\qwen3vl_4b_bf16-Q8_CR.gguf" --runs 10 --warmup 2 --output qwen3vl-4b-q8-cr-benchmark.json
```

Run it from the ComfyUI-GGUF repository using the ComfyUI Python environment.
It reports load, first-use and warm text-encode latency separately, CUDA
peak-memory measurements, native/fallback execution counts and output
similarity. The BF16 baseline is the full multimodal checkpoint while the GGUF
is text-only, so this is not a like-for-like storage or multimodal benchmark;
results are specific to the reported hardware, runtime, prompts and offload
settings.

For transparency, one 40-encode-per-model run (10 per prompt, 23–416 tokens)
on an RTX 3080 Laptop GPU (compute capability 8.6, PyTorch 2.14.0+cu130, CUDA
13.0, ComfyUI `88ab4a0`, CPU offload) measured a 489.935 ms BF16 warm median
and a 461.125 ms Q8_CR median; Q8_CR was about 5.9% faster in this workload.
The benchmark observed 12,348 native TensorWiseINT8 Linear calls and no
fallback. Peak CUDA allocation while encoding was about 150 MiB for BF16 and
1,497 MiB for Q8_CR in these CPU-offload runs. Tokenization matched, but output
relative L2 error was 10.2–11.9%, and thelargest absolute difference was about 156 (5.09% of the BF16 output's maximum
absolute value). Treat this as a single, workload-specific measurement—not a
speed or quality guarantee. The BF16 baseline retains the full vision tower,
whereas the converted GGUF is text-only.

## Converting Models (Krea 2, Ideogram 4, MiniMax H3, MiniMax Music 3, Qwen Image 2.1, Qwen3-VL 4B)

This node pack includes a GGUF converter. It has 3 possible interfaces that you can use: 
- a python file you can call directly
- a web interface
- a custom node

Each option is documented here: [Quantizing models](https://molbal.github.io/gguf/ecosystem/quantizing-models.html)

## Supported Conversion Formats

| Format | Storage / execution       | Recommended use                                             |
|--------|---------------------------|-------------------------------------------------------------|
| F16    | FP16 GGUF                 | Maximum compatibility with half-precision storage.          |
| BF16   | BF16 GGUF                 | Preserve BF16 source models where the target supports BF16. |
| Q8_0   | Standard GGML 8-bit       | Excellent general-quality 8-bit GGUF.                       |
| Q5_1   | Standard GGML 5-bit       | Lower storage with a quality-oriented 5-bit format.         |
| Q5_0   | Standard GGML 5-bit       | Lower storage alternative to `Q5_1`.                        |
| Q4_1   | Standard GGML 4-bit       | Smaller files when VRAM or RAM is constrained.              |
| Q4_0   | Standard GGML 4-bit       | Smallest supported format for constrained setups.           |
| Q6_K   | Uniform GGML K 6-bit      | High-quality K-quant storage for eligible Linear weights.    |
| Q5_K   | Uniform GGML K 5-bit      | Uniform 5-bit K-quant storage.                               |
| Q5_K_S / Q5_K_M | Mixed GGML K     | Deterministic 5/6-bit distributions; `_M` protects more sensitive tensors. |
| Q4_K   | Uniform GGML K 4-bit      | Uniform 4-bit K-quant storage.                               |
| Q4_K_S / Q4_K_M | Mixed GGML K     | Deterministic 4/5/6-bit distributions; `_M` protects more sensitive tensors. |
| Q8_CR  | Per-row INT8 ConvRot      | Maintainer recommendation for NVIDIA RTX 30-series systems. |
| Q4_CR  | Experimental INT4 ConvRot | Experimental quantization targeting INT4                    |

K-quant creation uses the GGML-compatible deterministic encoder bundled with
this converter and requires every selected row to be divisible by 256.
Architecture-protected, convolution, small, and one-dimensional tensors retain
higher precision. The local encoder is used even when the installed `gguf`
Python package only supports reading K quants; output is still validated against
the repository’s reference dequantizer. K-quant inference uses the portable
dequantization path by default. The experimental bundled Triton/CUDA Linear
backend is opt-in with `COMFYUI_GGUF_KQUANT_BACKEND=bundled`; validate both
output and performance on your hardware before enabling it. Its current
implementation accelerates decoding, then uses the same PyTorch Linear as the
portable path, including the loader's decode precision and bias behavior.
It still materializes a floating-point weight. Its configs are limited to SM
8.6, a few benchmarked LTX Q5_K/Q6_K matrix shapes, and at most 1024 activation
rows; every other route stays on portable PyTorch. This
is a local RTX 3080 Laptop tuning result, not an RTX 5090 or end-to-end speed
claim. See [K-quant inference notes](docs/k-quant-inference.md) for details.
