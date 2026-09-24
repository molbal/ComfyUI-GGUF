# `_K` Quantization During Inference

## Conclusion

`_K` quants reduce GGUF storage and can reduce the resident compressed-weight
footprint. Standard GGML weights use the portable dequantization path when no
native backend is available. Q4_K, Q5_K, and Q6_K Linear weights can instead
use the bundled optional Triton backend on CUDA; it receives the original GGUF
blocks directly and fuses unpacking with the matrix multiply. Unsupported
routes automatically fall back to the reference path, and no universal speed
up should be assumed without a benchmark on the target GPU.

`_K` remains reasonable for text encoders when the compressed file or
CPU/offload footprint is the primary constraint and the one-time text encoding
latency is acceptable. It should not be presented as an inference-speed
optimization in this node.

## What the Formats Save

`_K` types encode 256 weights per super-block with additional per-sub-block
scales/minima. That improves quality at a given storage budget, but it makes
unpacking more complex. The GGML block definitions give these payload sizes:

| Tensor type | Bytes per 256 weights | Payload bits/weight | Storage comparison |
| --- | ---: | ---: | --- |
| `Q2_K` | 84 | 2.625 | Much smaller than FP16 |
| `Q3_K` | 110 | 3.438 | Much smaller than FP16 |
| `Q4_K` | 144 | 4.500 | Same payload density as `Q4_0` |
| `Q5_K` | 176 | 5.500 | Same payload density as `Q5_0` |
| `Q6_K` | 210 | 6.563 | Smaller than `Q8_0` |
| `Q8_0` | 272 | 8.500 | Higher-quality conventional GGML baseline |
| FP16 | 512 | 16.000 | Uncompressed compute-weight baseline |

The `_S`, `_M`, and `_L` suffixes in distribution filenames commonly describe
how a model mixes quantization choices across tensors; they are not a separate
single tensor encoding that this loader can execute differently.

## Creating K-Quant GGUFs

The CLI, local conversion dashboard, and **Targeted Quantization (GGUF)** node
offer uniform `Q4_K`, `Q5_K`, and `Q6_K`, plus mixed `Q4_K_S`, `Q4_K_M`,
`Q5_K_S`, and `Q5_K_M` selections:

```powershell
python tools\convert.py --src model.safetensors --quant-type Q4_K_M
```

Uniform choices apply one K qtype to every eligible matrix. Mixed choices use
a stable tensor-name plan and promote sensitive attention, FFN-down,
embedding, and output categories while honoring each architecture's protected
tensor rules. K rows must be divisible by 256; incompatible eligible tensors
are rejected with their key and shape. Creation uses the converter's bundled
deterministic GGML-compatible encoder, so an installed `gguf` package with
read-only K support is sufficient.
These files do not imply native low-bit execution. Eligible Linear layers may
use the optional backend described below; every other route dequantizes first.

## Runtime Paths

The portable fallback path is explicit:

1. `GGMLLayer.cast_bias_weight()` in [`ops.py`](../ops.py) calls
   `get_weight()` for every weighted operation.
2. `get_weight()` calls `dequantize_tensor()`.
3. `dequant.py` decodes `_K` blocks through PyTorch tensor operations: unpacking
   bit fields, expanding scales/minima, and materializing floating-point
   weights.
4. `torch.nn.functional.linear()` or the matching PyTorch operation then runs
   on the expanded weight.

Dynamic VRAM uses `GGMLLayout.dequantize()` in
[`quant_ops.py`](../quant_ops.py), which follows the same materialization model
for unsupported types and operations. The compressed data may remain
mmap-backed until needed, but the operation still needs a full floating-point
temporary.

For eligible 2-D Q4_K, Q5_K, and Q6_K Linear weights, both the standard
`GGMLOps.Linear` path and Dynamic VRAM's `GGMLLayout` first offer the compressed
GGUF bytes to the optional backend. Weight patches/LoRAs continue to use the
portable path so ComfyUI's patch semantics remain unchanged. Unsupported
devices, dtypes, shapes, operations, missing backends, and rejected kernel
calls also fall back to the existing dequantized operation.

### Optional backend contract

Set `COMFYUI_GGUF_KQUANT_BACKEND` to an importable Python module name to
override the bundled backend. If it is unset, the integration first probes
`comfyui_gguf_kquant` and then the bundled Triton backend. Triton and CUDA are
optional; when they are unavailable the reference path remains active. A
backend module may expose itself (or `backend`) with:

```python
def supports(*, qtype, device, input_dtype, weight_shape, weight_dtype) -> bool:
    ...

def linear(*, input, weight, qtype, weight_shape, bias):
    ...
```

`qtype` is `"Q4_K"`, `"Q5_K"`, or `"Q6_K"`. `weight` is contiguous,
device-resident GGUF block storage; `weight_shape` is the logical `(N, K)`
matrix. `linear` must return the same shape and device as
`torch.nn.functional.linear(input, dequantized_weight, bias)`, using the input
dtype. Alternatively, the module can call `register_kquant_backend()` from
[`kquant_backend.py`](../kquant_backend.py) while it imports.

The integration verifies the block-storage size and output contract. A backend
that rejects or fails a route is disabled for that quant/device/dtype
combination, and subsequent calls use the portable fallback. Backend packages
remain responsible for compiled-kernel correctness, device capability checks,
stream safety, and numerical validation against GGML dequantization.

In contrast, `Q8_CR` is converted to ComfyUI's
`TensorWiseINT8Layout`; [`get_gguf_q8_ops()`](../ops.py) retains INT8 weights
and uses ComfyUI's native INT8 Linear route. That avoids the generic GGML
dequantize-then-FP16/BF16-matmul sequence for eligible Linear layers.

## Practical Implications

| Scenario | `_K` result in this node |
| --- | --- |
| GGUF disk size / mmap-backed source weights | Lower, according to its payload bits per weight. |
| System RAM or VRAM while a layer is not resident | Often lower, especially with offload or Dynamic VRAM. |
| Peak working memory for an executing layer | Still requires a floating temporary; the largest layer is accounted for in `ops.py`. |
| Diffusion denoising latency | Usually unfavorable: every sampling step revisits many layers and repeats unpacking. |
| Text-encoder latency | May be acceptable because encoding occurs once per prompt, but measure it. |
| Output quality at a size budget | Often better than legacy quants of comparable payload size, but architecture- and model-dependent. |
| Native low-bit throughput | Available for eligible Linear layers through the bundled Triton/CUDA backend; other routes use the portable fallback. |

The actual outcome also depends on GPU, CPU, PCIe bandwidth, batch/sequence
size, ComfyUI offload policy, and whether the model is compute- or
transfer-bound. There is no defensible universal tokens/s or seconds/step
multiplier without a benchmark on the target workflow.

## Recommended Choices

- **Diffusion models on NVIDIA:** Prefer `Q8_CR` for eligible transformer/DiT
  Linear weights unless a compatible K-quant backend is installed and measured
  on the target workflow.
- **Portable diffusion GGUF:** Prefer the documented standard formats
  (`Q8_0`, `Q5_0`, `Q4_0`) and choose file size versus output quality. Do not
  select `_K` expecting faster samples.
- **Text encoders under memory pressure:** `_K` can be useful if its measured
  prompt-encoding latency is acceptable. `Q4_K_M`/`Q5_K_M` are quality/storage
  candidates, not speed recommendations.

## Benchmark Protocol

Compare only one variable at a time: use the same source model, ComfyUI
revision, workflow, seed, prompt, resolution, sampling steps, scheduler,
offload mode, and device placement.

1. Warm up the workflow once to exclude compilation and initial allocation.
2. Run at least five measured generations for each quantization.
3. Record median wall-clock seconds per denoise step, total generation time,
   text-encoder time, peak allocated/reserved VRAM, and process RAM.
4. Repeat once with full model residency and once with the intended offload or
   Dynamic VRAM policy; transfer-bound behavior can reverse a result.
5. Compare outputs at the same seed for visible degradation before accepting a
   smaller format.
6. Include loader logs with tensor type counts and document the hardware,
   PyTorch, ComfyUI, and node revision with the result.

## Sources

- [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md):
  GGUF is mmap-compatible and stores model metadata and tensors.
- [GGML quantization reference implementation](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c):
  defines the legacy and `_K` super-block encodings and their dequantization.
- Local implementation: [`dequant.py`](../dequant.py),
  [`ops.py`](../ops.py), and [`quant_ops.py`](../quant_ops.py).
