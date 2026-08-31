# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import gguf
import json
import torch
import logging

import comfy.ops
import comfy.lora
import comfy.model_management
from .dequant import dequantize_tensor, is_quantized

def _build_regular_hadamard(size, dtype=torch.float32, device="cpu"):
    """Build a normalized regular (Sylvester) Hadamard of a power-of-4 size."""
    if size < 4 or (size & (size - 1)) != 0:
        import math
        if not math.log(size, 4).is_integer():
            raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=dtype,
        device=device,
    )
    h = h4
    current_size = 4
    while current_size < size:
        h = torch.kron(h, h4)
        current_size *= 4
    return h / (size ** 0.5)

def _valid_compute_dtype(dtype):
    return dtype in {torch.float16, torch.bfloat16, torch.float32, torch.float64}

def _infer_compute_dtype(tensor_type, fallback=None):
    if _valid_compute_dtype(fallback):
        return fallback
    if tensor_type == gguf.GGMLQuantizationType.BF16:
        return torch.bfloat16
    if tensor_type == gguf.GGMLQuantizationType.F32:
        return torch.float32
    return torch.float16

def chained_hasattr(obj, chained_attr):
    probe = obj
    for attr in chained_attr.split('.'):
        if hasattr(probe, attr):
            probe = getattr(probe, attr)
        else:
            return False
    return True

# A bakcward and forward compatible way to get `torch.compiler.disable`.
def get_torch_compiler_disable_decorator():
    def dummy_decorator(*args, **kwargs):
        def noop(x):
            return x
        return noop

    from packaging import version

    if not chained_hasattr(torch, "compiler.disable"):
        logging.info("ComfyUI-GGUF: Torch too old for torch.compile - bypassing")
        return dummy_decorator # torch too old
    elif version.parse(torch.__version__) >= version.parse("2.8"):
        logging.info("ComfyUI-GGUF: Allowing full torch compile")
        return dummy_decorator # torch compile works
    if chained_hasattr(torch, "_dynamo.config.nontraceable_tensor_subclasses"):
        logging.info("ComfyUI-GGUF: Allowing full torch compile (nightly)")
        return dummy_decorator # torch compile works, nightly before 2.8 release
    else:
        logging.info("ComfyUI-GGUF: Partial torch compile only, consider updating pytorch")
        return torch.compiler.disable

torch_compiler_disable = get_torch_compiler_disable_decorator()

class GGMLTensor(torch.Tensor):
    """
    Main tensor-like class for storing quantized weights
    """
    def __init__(self, *args, tensor_type, tensor_shape, patches=[], compute_dtype=None, **kwargs):
        super().__init__()
        self.tensor_type = tensor_type
        self.tensor_shape = tensor_shape
        self.patches = patches
        self.compute_dtype = compute_dtype

    def __new__(cls, *args, tensor_type, tensor_shape, patches=[], compute_dtype=None, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", new.data.shape)
        new.patches = getattr(self, "patches", []).copy()
        new.compute_dtype = getattr(self, "compute_dtype", None)
        return new

    def clone(self, *args, **kwargs):
        return self

    def detach(self, *args, **kwargs):
        return self

    def copy_(self, *args, **kwargs):
        # fixes .weight.copy_ in comfy/clip_model/CLIPTextModel
        try:
            return super().copy_(*args, **kwargs)
        except Exception as e:
            logging.warning(f"ignoring 'copy_' on tensor: {e}")

    def new_empty(self, size, *args, **kwargs):
        # Intel Arc fix, ref#50
        new_tensor = super().new_empty(size, *args, **kwargs)
        return GGMLTensor(
                new_tensor,
                tensor_type = getattr(self, "tensor_type", None),
                tensor_shape = size,
                patches = getattr(self, "patches", []).copy(),
                compute_dtype = getattr(self, "compute_dtype", None),
        )

    @property
    def dtype(self):
        qtype = getattr(self, "tensor_type", None)
        if qtype in GGMLLayer.torch_compatible_tensor_types:
            # NOTE: use the base-class descriptor instead of torch.Tensor(self):
            # constructing a Tensor from an inference-mode tensor raises
            # "Inference tensors do not track version counter" (hit via
            # low_vram_patch_estimate_vram on bias keys when LoRAs patch biases)
            return torch.Tensor.dtype.__get__(self)
        return _infer_compute_dtype(qtype, getattr(self, "compute_dtype", None))

    @property
    def shape(self):
        if not hasattr(self, "tensor_shape"):
            self.tensor_shape = self.size()
        return self.tensor_shape

class GGMLLayer(torch.nn.Module):
    """
    This (should) be responsible for de-quantizing on the fly
    """
    comfy_cast_weights = True
    dequant_dtype = None
    patch_dtype = None
    largest_layer = False
    torch_compatible_tensor_types = {None, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}

    def is_ggml_quantized(self, *, weight=None, bias=None):
        if weight is None:
            weight = self.weight
        if bias is None:
            bias = self.bias
        return is_quantized(weight) or is_quantized(bias)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        weight, bias = state_dict.get(f"{prefix}weight"), state_dict.get(f"{prefix}bias")
        # NOTE: using modified load for linear due to not initializing on creation, see GGMLOps todo
        if self.is_ggml_quantized(weight=weight, bias=bias) or isinstance(self, torch.nn.Linear):
            return self.ggml_load_from_state_dict(state_dict, prefix, *args, **kwargs)
        # Not strictly required, but fixes embedding shape mismatch. Threshold set in loader.py
        if isinstance(self, torch.nn.Embedding) and self.weight.shape[0] >= (64 * 1024):
            return self.ggml_load_from_state_dict(state_dict, prefix, *args, **kwargs)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def ggml_load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        prefix_len = len(prefix)
        for k,v in state_dict.items():
            if k[prefix_len:] == "weight":
                if isinstance(v, GGMLTensor):
                    v.compute_dtype = self._ggml_compute_dtype(v, "weight")
                self.weight = torch.nn.Parameter(v, requires_grad=False)
            elif k[prefix_len:] == "bias" and v is not None:
                if isinstance(v, GGMLTensor):
                    v.compute_dtype = self._ggml_compute_dtype(v, "bias")
                self.bias = torch.nn.Parameter(v, requires_grad=False)
            else:
                unexpected_keys.append(k)

        # For Linear layer with missing weight
        if self.weight is None and isinstance(self, torch.nn.Linear):
            v = torch.zeros(self.in_features, self.out_features)
            self.weight = torch.nn.Parameter(v, requires_grad=False)
            missing_keys.append(prefix+"weight")

        # for vram estimation (TODO: less fragile logic?)
        if getattr(self.weight, "is_largest_weight", False):
            self.largest_layer = True

    def _ggml_compute_dtype(self, tensor, param_name):
        if self.dequant_dtype is not None and self.dequant_dtype != "target":
            return self.dequant_dtype
        model_dtype = getattr(self, f"{param_name}_comfy_model_dtype", None)
        return _infer_compute_dtype(getattr(tensor, "tensor_type", None), model_dtype)

    def _save_to_state_dict(self, *args, **kwargs):
        if self.is_ggml_quantized():
            return self.ggml_save_to_state_dict(*args, **kwargs)
        return super()._save_to_state_dict(*args, **kwargs)

    def ggml_save_to_state_dict(self, destination, prefix, keep_vars):
        # This is a fake state dict for vram estimation
        weight = torch.zeros_like(self.weight, device=torch.device("meta"))
        destination[prefix + "weight"] = weight
        if self.bias is not None:
            bias = torch.zeros_like(self.bias, device=torch.device("meta"))
            destination[prefix + "bias"] = bias

        # Take into account space required for dequantizing the largest tensor
        if self.largest_layer:
            shape = getattr(self.weight, "tensor_shape", self.weight.shape)
            dtype = self.dequant_dtype if self.dequant_dtype and self.dequant_dtype != "target" else torch.float16
            temp = torch.empty(*shape, device=torch.device("meta"), dtype=dtype)
            destination[prefix + "temp.weight"] = temp

        return
        # This would return the dequantized state dict
        destination[prefix + "weight"] = self.get_weight(self.weight)
        if bias is not None:
            destination[prefix + "bias"] = self.get_weight(self.bias)

    def get_weight(self, tensor, dtype):
        if tensor is None:
            return

        # consolidate and load patches to GPU in async
        patch_list = []
        device = tensor.device
        for patches, key in getattr(tensor, "patches", []):
            patch_list += move_patch_to_device(patches, device)

        # dequantize tensor while patches load
        weight = dequantize_tensor(tensor, dtype, self.dequant_dtype)

        # prevent propagating custom tensor class
        if isinstance(weight, GGMLTensor):
            weight = torch.Tensor(weight)

        # apply patches
        if len(patch_list) > 0:
            if self.patch_dtype is None:
                weight = comfy.lora.calculate_weight(patch_list, weight, key)
            else:
                # for testing, may degrade image quality
                patch_dtype = dtype if self.patch_dtype == "target" else self.patch_dtype
                weight = comfy.lora.calculate_weight(patch_list, weight, key, patch_dtype)
        return weight

    @torch_compiler_disable()
    def cast_bias_weight(s, input=None, dtype=None, device=None, bias_dtype=None):
        if input is not None:
            if dtype is None:
                dtype = getattr(input, "dtype", torch.float32)
            if bias_dtype is None:
                bias_dtype = dtype
            if device is None:
                device = input.device

        bias = None
        non_blocking = comfy.model_management.device_supports_non_blocking(device)
        if s.bias is not None:
            bias = s.get_weight(s.bias.to(device), dtype)
            bias = comfy.ops.cast_to(bias, bias_dtype, device, non_blocking=non_blocking, copy=False)

        weight = s.get_weight(s.weight.to(device), dtype)
        weight = comfy.ops.cast_to(weight, dtype, device, non_blocking=non_blocking, copy=False)
        return weight, bias

    def forward_comfy_cast_weights(self, input, *args, **kwargs):
        if self.is_ggml_quantized():
            out = self.forward_ggml_cast_weights(input, *args, **kwargs)
        else:
            out = super().forward_comfy_cast_weights(input, *args, **kwargs)

        # non-ggml forward might still propagate custom tensor class
        if isinstance(out, GGMLTensor):
            out = torch.Tensor(out)
        return out

    def forward_ggml_cast_weights(self, input):
        raise NotImplementedError

class GGMLOps(comfy.ops.manual_cast):
    """
    Dequantize weights on the fly before doing the compute
    """
    class Linear(GGMLLayer, comfy.ops.manual_cast.Linear):
        def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
            torch.nn.Module.__init__(self)
            # TODO: better workaround for reserved memory spike on windows
            # Issue is with `torch.empty` still reserving the full memory for the layer
            # Windows doesn't over-commit memory so without this 24GB+ of pagefile is used
            self.in_features = in_features
            self.out_features = out_features
            self.weight = None
            self.bias = None
            self.weight_comfy_model_dtype = dtype
            self.bias_comfy_model_dtype = dtype

        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.linear(input, weight, bias)

    class Conv2d(GGMLLayer, comfy.ops.manual_cast.Conv2d):
        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return self._conv_forward(input, weight, bias)

    class Conv3d(GGMLLayer, comfy.ops.manual_cast.Conv3d):
        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return self._conv_forward(input, weight, bias)

    class Embedding(GGMLLayer, comfy.ops.manual_cast.Embedding):
        def forward_ggml_cast_weights(self, input, out_dtype=None):
            output_dtype = out_dtype
            if self.weight.dtype == torch.float16 or self.weight.dtype == torch.bfloat16:
                out_dtype = None
            weight, _bias = self.cast_bias_weight(self, device=input.device, dtype=out_dtype)
            return torch.nn.functional.embedding(
                input, weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse
            ).to(dtype=output_dtype)

    class LayerNorm(GGMLLayer, comfy.ops.manual_cast.LayerNorm):
        def forward_ggml_cast_weights(self, input):
            if self.weight is None:
                return super().forward_comfy_cast_weights(input)
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.layer_norm(input, self.normalized_shape, weight, bias, self.eps)

    class GroupNorm(GGMLLayer, comfy.ops.manual_cast.GroupNorm):
        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.group_norm(input, self.num_groups, weight, bias, self.eps)

def move_patch_to_device(item, device):
    if isinstance(item, torch.Tensor):
        return item.to(device, non_blocking=True)
    elif isinstance(item, tuple):
        return tuple(move_patch_to_device(x, device) for x in item)
    elif isinstance(item, list):
        return [move_patch_to_device(x, device) for x in item]
    else:
        return item

def get_gguf_q8_ops(compute_dtype=torch.bfloat16, full_precision_mm=False):
    """
    Factory for an ops class that uses ComfyUI's native mixed_precision_ops INT8 path.
    Weights are kept as INT8 and matmul uses comfy_kitchen's TensorWiseINT8Layout.
    """
    BaseOps = comfy.ops.mixed_precision_ops(
        quant_config={},
        compute_dtype=compute_dtype,
        full_precision_mm=full_precision_mm,
    )

    class GGUFQ8Ops(BaseOps):
        class Linear(BaseOps.Linear):
            def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
                # Lazy init: don't allocate weight here; it will be loaded from state dict
                torch.nn.Module.__init__(self)
                self.factory_kwargs = {"device": device, "dtype": BaseOps._compute_dtype}
                self.in_features = in_features
                self.out_features = out_features
                self.weight = None
                if bias:
                    self.bias = torch.nn.Parameter(torch.empty(out_features, **self.factory_kwargs))
                else:
                    self.register_parameter("bias", None)
                self._orig_shape = (out_features, in_features)
                self.tensor_class = None
                self._full_precision_mm = BaseOps._full_precision_mm
                self._full_precision_mm_config = False

            def _load_from_state_dict(self, *args):
                state_dict, prefix = args[:2]
                weight_key = f"{prefix}weight"
                # Target-size GGUFs combine native Q8_CR weights with standard
                # Q4_0 weights. The mixed-precision loader only understands
                # the former's comfy_quant metadata, so materialize standard
                # GGML weights before handing them to that loader.
                if weight_key in state_dict and f"{prefix}comfy_quant" not in state_dict:
                    weight = state_dict[weight_key]
                    if is_quantized(weight):
                        state_dict[weight_key] = dequantize_tensor(
                            weight,
                            dtype=self.factory_kwargs["dtype"],
                        )
                    elif hasattr(weight, "dequantize"):
                        state_dict[weight_key] = weight.dequantize().to(
                            dtype=self.factory_kwargs["dtype"],
                        )
                return comfy.ops._load_quantized_module(
                    self,
                    torch.nn.Module._load_from_state_dict.__get__(self, type(self)),
                    *args,
                    load_extra_params=True,
                )

    return GGUFQ8Ops


# Q4_CR_W4A4: custom W4A4 INT4 format backed by comfy_kitchen's fast ConvRot
# int4 tensor-core MMA. On-disk is kitchen-native packed int4 (N, K//2) int8
# + a per-output-row fp16 scale. The weight is pre-rotated by a block-diagonal
# Hadamard along K; at runtime the kernel rotates the activation into the same
# basis, so output is directly in the original space.
def get_gguf_q4_w4a4_ops(compute_dtype=torch.bfloat16, full_precision_mm=False):
    try:
        from comfy_kitchen.tensor.convrot_w4a4 import (
            TensorCoreConvRotW4A4Layout,
            quantize_convrot_w4a4_weight,
        )
        from comfy_kitchen.tensor.base import QuantizedTensor
        _HAVE_KITCHEN = True
    except Exception:
        TensorCoreConvRotW4A4Layout = None
        quantize_convrot_w4a4_weight = None
        QuantizedTensor = None
        _HAVE_KITCHEN = False

    class GGUFQ4W4A4Ops(comfy.ops.disable_weight_init):
        class Linear(torch.nn.Module, comfy.ops.CastWeightBiasOp):
            comfy_cast_weights = True

            def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
                torch.nn.Module.__init__(self)
                self.in_features = in_features
                self.out_features = out_features
                self.weight = None
                self.register_parameter("bias", None)
                self._orig_shape = (out_features, in_features)
                self._convrot_groupsize = 256
                self._quant_group_size = 64
                self._quantized = False
                self.weight_scale = None
                self._compute_dtype = torch.bfloat16
                self._quantized_weight = None
                self._quantized_weight_device = None
                # Cache for a re-quantized fused (LoRA/LoKR) weight. Re-quantization
                # (Hadamard rotate + int4 pack) is expensive, so it runs once per
                # patch identity, not once per forward.
                self._fused_quantized_weight = None
                self._fused_patch_id = None
                self._fused_quantized_weight_device = None

            def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs):
                weight_key = f"{prefix}weight"
                scale_key = f"{prefix}weight_scale"
                quant_key = f"{prefix}comfy_quant"

                weight = state_dict.pop(weight_key, None)
                scale = state_dict.pop(scale_key, None)
                quant_raw = state_dict.pop(quant_key, None)

                bias_key = f"{prefix}bias"
                bias = state_dict.pop(bias_key, None)
                if bias is not None:
                    self.bias = torch.nn.Parameter(bias, requires_grad=False)
                else:
                    self.bias = None
                if bias_key in missing_keys:
                    missing_keys.remove(bias_key)

                if quant_raw is None or weight is None:
                    # Plain (non-quantized) Linear: keep the raw weight.
                    if weight is None:
                        missing_keys.append(weight_key)
                        return
                    self.weight = torch.nn.Parameter(
                        weight if isinstance(weight, torch.Tensor) else torch.Tensor(weight),
                        requires_grad=False,
                    )
                    self._quantized = False
                    return

                if scale is None:
                    raise RuntimeError(f"Missing Q4_CR_W4A4 scale tensor for {prefix}")

                quant_conf = json.loads(bytes(quant_raw.tolist()).decode("utf-8"))
                if quant_conf.get("format") != "int4_cr" or quant_conf.get("backing") != "w4a4":
                    raise ValueError(
                        f"Unsupported Q4_CR_W4A4 format for {prefix}: {quant_conf.get('format')!r} "
                        f"/ {quant_conf.get('backing')!r}"
                    )
                self._convrot_groupsize = quant_conf.get("convrot_groupsize", 256)
                self._quant_group_size = quant_conf.get("quant_group_size", 64)
                self._orig_shape = tuple(quant_conf.get("orig_shape", (self.out_features, self.in_features)))
                self.out_features = self._orig_shape[0]
                self.in_features = self._orig_shape[1]

                self.weight = torch.nn.Parameter(weight, requires_grad=False)
                self.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
                self._quantized = True
                self._compute_dtype = quant_conf.get("orig_dtype", torch.bfloat16)
                self._quantized_weight = None
                self._quantized_weight_device = None
                self._fused_quantized_weight = None
                self._fused_patch_id = None
                self._fused_quantized_weight_device = None

            def _build_quantized_weight(self, device, dtype, packed=None):
                if not _HAVE_KITCHEN:
                    return None
                if packed is None:
                    packed = self.weight.to(device=device).to(torch.int8)
                else:
                    packed = packed.to(device=device).to(torch.int8)
                scale = self.weight_scale.to(device=device, dtype=torch.float32).contiguous()
                params = TensorCoreConvRotW4A4Layout.Params(
                    scale=scale,
                    orig_dtype=dtype,
                    orig_shape=self._orig_shape,
                    convrot_groupsize=self._convrot_groupsize,
                    quant_group_size=self._quant_group_size,
                    linear_dtype="int4",
                )
                return QuantizedTensor(packed, "TensorCoreConvRotW4A4Layout", params)

            def _get_cached_quantized_weight(self, device, packed=None):
                if packed is not None:
                    # A patched/offloaded packed weight may differ from self.weight
                    # (e.g. moved to device each forward). Build fresh; don't cache,
                    # because the packed content can change per forward.
                    return self._build_quantized_weight(device, self._compute_dtype, packed=packed)
                if self._quantized_weight is not None and self._quantized_weight_device == str(device):
                    return self._quantized_weight
                if self._quantized_weight is not None:
                    self._quantized_weight = None
                    if hasattr(self, "_quantized_weight_scale") and self._quantized_weight_scale is not None:
                        # Free the previous device's cached scale/params to avoid a leak.
                        self._quantized_weight_scale = None
                qt = self._build_quantized_weight(device, self._compute_dtype)
                self._quantized_weight = qt
                self._quantized_weight_device = str(device)
                return qt

            def _fused_patch_signature(self):
                """A cheap, stable identity for the active weight/bias patch set.

                ComfyUI installs the adapter (LoRA/LoKR) functions once per model
                load/patch, so the callable objects are stable across forwards. We key
                the fused-weight cache on the callable objects themselves so a re-bound
                patch (even if a function object is reused) is detected and re-fused.

                Under Dynamic VRAM, however, ``GGUFModelPatcherDynamic.load`` promotes a
                freshly created ``LowVramPatch`` into ``weight_function`` every time the
                model is moved to device. That object is recreated on each reload, so
                keying the cache on ``id(f)`` would miss on every forward and re-run the
                expensive Hadamard rotate + int4 pack (a 12x slowdown on large layers).
                For ``LowVramPatch`` we therefore key on its stable *content*: the tensor
                key and the identity of the shared patches dict/list (both live on the
                model patcher and survive recreation). A genuine patch change (e.g. a new
                ``add_patches`` call) rebuilds that list, which is detected as a change.
                All other adapter callables are stable objects, so we key on them directly.
                """
                sig = []
                keepalive = []
                for f in list(getattr(self, "weight_function", ())) + list(getattr(self, "bias_function", ())):
                    if getattr(f, "is_lowvram_patch", False) and getattr(f, "key", None):
                        patches = getattr(f, "patches", None)
                        plist = patches.get(f.key) if patches is not None else None
                        if patches is not None:
                            keepalive.append(patches)
                        if plist is not None:
                            keepalive.append(plist)
                        sig.append(("lowvram", f.key, id(patches), id(plist)))
                    else:
                        sig.append(("callable", f))
                # Keep references alive so address reuse never aliases a stale signature.
                self._fused_patch_keepalive = tuple(keepalive)
                return tuple(sig)

            def _build_fused_quantized_weight(self, device, packed, wscales):
                """Build a QuantizedTensor from a re-quantized (fused) weight.

                Unlike ``_build_quantized_weight`` this uses the freshly recomputed
                per-row scales produced by re-quantizing a fused (LoRA) weight rather
                than the original on-disk ``weight_scale``.
                """
                if not _HAVE_KITCHEN:
                    return None
                packed = packed.to(device=device).to(torch.int8)
                scale = wscales.to(device=device, dtype=torch.float32).contiguous()
                params = TensorCoreConvRotW4A4Layout.Params(
                    scale=scale,
                    orig_dtype=self._compute_dtype,
                    orig_shape=self._orig_shape,
                    convrot_groupsize=self._convrot_groupsize,
                    quant_group_size=self._quant_group_size,
                    linear_dtype="int4",
                )
                return QuantizedTensor(packed, "TensorCoreConvRotW4A4Layout", params)

            def _get_cached_fused_quantized_weight(self, device):
                """Fuse active LoRA/LoKR adapters and re-quantize onto the fast kernel.

                ``weight_function`` holds either a real LoRA/LoKR adapter (float deltas)
                or, under DynamicVRAM, a pure weight-relocate patch. Both are applied to
                the full-precision (un-rotated) weight, then the fused weight is packed
                back into the ConvRot int4 layout. The result is cached per patch
                signature so the expensive Hadamard rotate + int4 pack runs only once per
                patch, not once per forward.

                Returns ``None`` when fusion is unavailable (no comfy_kitchen, or the
                patch cannot be re-quantized), signalling the caller to fall back to the
                dequantized matmul.
                """
                if not _HAVE_KITCHEN or quantize_convrot_w4a4_weight is None:
                    return None
                sig = self._fused_patch_signature()
                if (
                    self._fused_quantized_weight is not None
                    and self._fused_patch_id == sig
                    and self._fused_quantized_weight_device == str(device)
                ):
                    return self._fused_quantized_weight
                # Fuse adapters in the compute (BF16) domain on the un-rotated weight.
                fused = self._dequantized_weight(device, self._compute_dtype)
                for f in self.weight_function:
                    fused = f(fused)
                try:
                    packed, wscales = quantize_convrot_w4a4_weight(
                        fused,
                        convrot_groupsize=self._convrot_groupsize,
                        quant_group_size=self._quant_group_size,
                        stochastic_rounding=0,
                    )
                except Exception:
                    # Unfusable patch (e.g. altered shape). Fall back to dequant matmul.
                    self._fused_quantized_weight = None
                    return None
                qt = self._build_fused_quantized_weight(device, packed, wscales)
                self._fused_quantized_weight = qt
                self._fused_patch_id = sig
                self._fused_quantized_weight_device = str(device)
                return qt

            def forward_comfy_cast_weights(self, input, *args, **kwargs):
                bias = self.bias.to(device=input.device, dtype=input.dtype) if self.bias is not None else None
                if bias is not None:
                    for f in self.bias_function:
                        bias = f(bias)
                if not self._quantized or self.weight is None:
                    # Plain Linear path
                    weight = self.weight.to(device=input.device, dtype=input.dtype)
                    return torch.nn.functional.linear(input, weight, bias)
                if self.weight_function or self.bias_function:
                    # A real LoRA/LoKR/adapter patch is present. Dynamic VRAM promotes a
                    # LowVramPatch into weight_function, and the adapter's factors are
                    # floats, so they must be applied to the full-precision weight. We
                    # fuse them into the (un-rotated) BF16 weight, then re-quantize back
                    # into the ConvRot int4 layout so the fast tensor-core kernel stays
                    # active while an adapter is loaded. The fused weight is cached per
                    # patch signature (see _get_cached_fused_quantized_weight). If fusion
                    # is unavailable or fails (no comfy_kitchen / altered shape), fall
                    # back to a dequantized FP matmul for correctness.
                    fused_qt = self._get_cached_fused_quantized_weight(input.device)
                    if fused_qt is not None:
                        out = torch.nn.functional.linear(input, fused_qt)
                        if bias is not None:
                            out = out + bias
                        return out
                    # No kitchen or unfusable patch: dequantize and run an FP matmul.
                    _orig_w = self._dequantized_weight(input.device, input.dtype)
                    for f in self.weight_function:
                        _orig_w = f(_orig_w)
                    return torch.nn.functional.linear(input, _orig_w, bias)

                weight_qt = self._get_cached_quantized_weight(input.device)
                if weight_qt is None:
                    # No comfy_kitchen / non-CUDA: dequant fallback.
                    _orig_w = self._dequantized_weight(input.device, input.dtype)
                    return torch.nn.functional.linear(input, _orig_w, bias)

                out = torch.nn.functional.linear(input, weight_qt)
                if bias is not None:
                    out = out + bias
                return out

            def forward(self, *args, **kwargs):
                comfy.ops.run_every_op()
                if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                    return self.forward_comfy_cast_weights(*args, **kwargs)
                return torch.nn.functional.linear(
                    input=args[0],
                    weight=self.weight,
                    bias=self.bias,
                )

            def _dequantized_weight(self, device, dtype):
                packed = self.weight.to(device=device).to(torch.int8)
                scale = self.weight_scale.to(device=device, dtype=dtype)
                n, k_half = packed.shape
                k = k_half * 2
                x32 = packed.to(torch.int32)
                lo = (x32 & 0xF).to(torch.int8)
                hi = ((x32 >> 4) & 0xF).to(torch.int8)
                nibbles = torch.stack([lo, hi], dim=-1).reshape(n, k).to(torch.float32)
                # Signed two's-complement int4 -> [-8, 7], scaled per output row.
                nibbles = torch.where(nibbles >= 8, nibbles - 16, nibbles)
                w_rot = nibbles * scale.to(torch.float32).reshape(-1, 1)
                # The weight was pre-rotated by a block-diagonal Hadamard along K;
                # un-rotate so this dequant matches the kernel's effective weight.
                cg = self._convrot_groupsize
                if k % cg == 0:
                    h = _build_regular_hadamard(cg, dtype=torch.float32, device=device)
                    n_groups = k // cg
                    w_rot = (w_rot.reshape(n, n_groups, cg) @ h).reshape(n, k)
                return w_rot.to(dtype)

        def reset_parameters(self):
            return None

    return GGUFQ4W4A4Ops


# Retired experimental Q4_PT implementation. No loader or node runtime path
# references this class until a performant W4A16 backend is available.
class RetiredGGUFQ4Ops(comfy.ops.manual_cast):
    """
    Ops class for PyTorch's native compact INT4 GEMM.

    Packed weights are created transiently at invocation time. This keeps
    low-VRAM offloading viable without retaining a second copy of all weights.
    """
    class Linear(torch.nn.Module, comfy.ops.CastWeightBiasOp):
        comfy_cast_weights = True

        def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
            torch.nn.Module.__init__(self)
            self.in_features = in_features
            self.out_features = out_features
            self.weight = None
            self.register_parameter("bias", None)
            self._orig_shape = (out_features, in_features)
            self._group_size = None
            self._pad = 0
            self._orig_in_features = in_features
            self._is_int4 = False

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            weight_key = f"{prefix}weight"
            scale_key = f"{prefix}weight_scale"
            quant_key = f"{prefix}comfy_quant"

            weight = state_dict.pop(weight_key, None)
            scale = state_dict.pop(scale_key, None)
            quant_raw = state_dict.pop(quant_key, None)

            bias_key = f"{prefix}bias"
            bias = state_dict.pop(bias_key, None)
            if quant_raw is None:
                if weight is None:
                    missing_keys.append(weight_key)
                    return
                self.weight = torch.nn.Parameter(torch.Tensor(weight), requires_grad=False)
                if bias is not None:
                    self.bias = torch.nn.Parameter(torch.Tensor(bias), requires_grad=False)
                self._is_int4 = False
                return

            if weight is None or scale is None:
                raise RuntimeError(f"Missing INT4 tensors for {prefix}")

            quant_conf = json.loads(bytes(quant_raw.tolist()).decode("utf-8"))
            if quant_conf.get("format") not in {"int4_compact_gemm", "int4_pytorch"}:
                raise ValueError(f"Unsupported INT4 format for {prefix}")
            self._group_size = quant_conf["group_size"]
            self._pad = quant_conf.get("pad", 0)
            orig_shape = tuple(quant_conf["orig_shape"])
            self._orig_in_features = orig_shape[1]
            self._orig_shape = orig_shape
            self.out_features = orig_shape[0]

            self.weight = torch.nn.Parameter(weight, requires_grad=False)
            self.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
            self._is_int4 = True

            if bias is not None:
                self.bias = torch.nn.Parameter(bias, requires_grad=False)
            for key in (weight_key, scale_key, quant_key, bias_key):
                if key in missing_keys:
                    missing_keys.remove(key)

        def forward(self, input):
            if self.weight is None:
                raise RuntimeError("Q4_PT weight was not loaded.")
            if not self._is_int4:
                weight = self.weight.to(device=input.device, dtype=input.dtype)
                bias = self.bias.to(device=input.device, dtype=input.dtype) if self.bias is not None else None
                return torch.nn.functional.linear(input, weight, bias)
            if self.weight_function or self.bias_function:
                raise RuntimeError("Q4_PT does not support weight patches or LoRAs without dequantization.")
            input_shape = input.shape
            input_2d = input.reshape(-1, input_shape[-1])
            if self._pad:
                input_2d = torch.nn.functional.pad(input_2d, (0, self._pad))

            if self._group_size != 64:
                raise RuntimeError(
                    f"Q4_PT requires PyTorch's group-size-64 INT4 operator, got {self._group_size}."
                )
            if input_2d.device.type != "cuda":
                raise RuntimeError("Q4_PT requires a CUDA device.")
            if input_2d.dtype != torch.bfloat16:
                raise RuntimeError(
                    f"Q4_PT requires BF16 activations for PyTorch's native INT4 operator, got {input_2d.dtype}."
                )

            weight = torch.Tensor(
                self.weight.to(device=input.device, dtype=torch.uint8, non_blocking=True)
            )
            scale_and_offset = self.weight_scale.to(
                device=input.device,
                dtype=torch.bfloat16,
                non_blocking=True,
            )
            packed_features = weight.size(-1) * 2
            native_padding = (-packed_features) % 128
            if native_padding:
                # PyTorch's INT4 packer needs K to be a multiple of 128.
                # A zero-valued group and zero input features preserve the
                # original result for K=64 projections such as Krea2's input.
                if native_padding % self._group_size:
                    raise RuntimeError(
                        f"Cannot pad Q4_PT input width {packed_features} to PyTorch's INT4 tile size."
                    )
                input_2d = torch.nn.functional.pad(input_2d, (0, native_padding))
                weight = torch.nn.functional.pad(weight, (0, native_padding // 2))
                scale_and_offset = torch.nn.functional.pad(
                    scale_and_offset,
                    (0, 0, 0, 0, 0, native_padding // self._group_size),
                )

            packed_weight = torch._convert_weight_to_int4pack(weight, 8)
            output = torch._weight_int4pack_mm(
                input_2d,
                packed_weight,
                self._group_size,
                scale_and_offset,
            )
            if self.bias is not None:
                output = output + self.bias.to(device=input.device, dtype=input.dtype)
            return output.reshape(*input_shape[:-1], self.out_features)
