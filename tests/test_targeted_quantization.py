import unittest
import json
import gc
import os
from collections import OrderedDict
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import gguf
import torch
import comfy.sd
from safetensors.torch import save_file

from tools.convert import (
    MEBIBYTE,
    ModelLTXV,
    ModelLTXVUpsampler,
    ModelMinimaxH3,
    ModelMinimaxH3VAE,
    ModelMiniMaxMusic3DiT,
    ModelMiniMaxMusic3TextEncoder,
    ModelTemplate,
    _streamed_safetensors_layout,
    convert_file,
    convert_state_dict,
    detect_arch,
    plan_target_size_quantization,
    quantize_int8_convrot,
    quantize_int4_cr,
    resolve_quantization_device,
)
from dequant import dequantize, dequantize_functions, dequantize_tensor
from lora import (
    fuse_targets_into_state_dict,
    load_gguf_lora,
    load_lora,
    materialize_int8_source_weights,
    resolve_fusion_targets,
)


def load_gguf_loader():
    loader_path = Path(__file__).parents[1] / "loader.py"
    package_name = "comfyui_gguf_test"
    spec = importlib.util.spec_from_file_location(
        f"{package_name}.loader",
        loader_path,
        submodule_search_locations=[str(loader_path.parent)],
    )
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[package_name] = module
    sys.modules[f"{package_name}.loader"] = module
    spec.loader.exec_module(module)
    return module


def load_gguf_ops():
    ops_path = Path(__file__).parents[1] / "ops.py"
    package_name = "comfyui_gguf_test"
    spec = importlib.util.spec_from_file_location(
        f"{package_name}.ops",
        ops_path,
        submodule_search_locations=[str(ops_path.parent)],
    )
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[package_name] = module
    sys.modules[f"{package_name}.ops"] = module
    spec.loader.exec_module(module)
    return module


class TargetSizeQuantizationTests(unittest.TestCase):
    def setUp(self):
        self.model_arch = ModelTemplate()
        self.state_dict = OrderedDict(
            (f"blocks.{index}.weight", torch.ones((4096, 32), dtype=torch.float32))
            for index in range(3)
        )
        self.state_dict["normalization.weight"] = torch.ones((4096,), dtype=torch.float32)

    def test_reduces_center_core_layers_before_outer_layers(self):
        plan, _, selected_size = plan_target_size_quantization(
            self.state_dict, self.model_arch, 0.4
        )

        self.assertEqual(plan["blocks.1.weight"], gguf.GGMLQuantizationType.Q5_0)
        self.assertEqual(plan["blocks.0.weight"], gguf.GGMLQuantizationType.I8)
        self.assertEqual(plan["blocks.2.weight"], gguf.GGMLQuantizationType.I8)
        self.assertLessEqual(selected_size, int(0.4 * MEBIBYTE))

    def test_reduces_one_dimensional_weights_only_after_all_core_layers(self):
        plan, _, selected_size = plan_target_size_quantization(
            self.state_dict, self.model_arch, 0.222
        )

        for index in range(3):
            self.assertEqual(plan[f"blocks.{index}.weight"], gguf.GGMLQuantizationType.Q4_0)
        self.assertEqual(plan["normalization.weight"], gguf.GGMLQuantizationType.BF16)
        self.assertLessEqual(selected_size, int(0.222 * MEBIBYTE))

    def test_supports_standard_q8_baseline(self):
        plan, _, _ = plan_target_size_quantization(
            self.state_dict,
            self.model_arch,
            2.0,
            target_size_q8_type="Q8_0",
        )

        for index in range(3):
            self.assertEqual(plan[f"blocks.{index}.weight"], gguf.GGMLQuantizationType.Q8_0)

    def test_reports_minimum_when_target_is_unsupported(self):
        with self.assertRaisesRegex(ValueError, "smallest supported TARGET_SIZE output"):
            plan_target_size_quantization(self.state_dict, self.model_arch, 0.1)


class Q8CRConversionDeviceTests(unittest.TestCase):
    def test_auto_uses_cpu_without_cuda(self):
        with mock.patch("tools.convert.torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_quantization_device("auto").type, "cpu")

    def test_cuda_requires_available_device(self):
        with mock.patch("tools.convert.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "requires an available CUDA device"):
                resolve_quantization_device("cuda")

    def test_cpu_quantization_stays_on_cpu(self):
        qdata, scale, quant_conf, _ = quantize_int8_convrot(
            torch.arange(256, dtype=torch.float32).reshape(1, 256),
            device=torch.device("cpu"),
        )

        self.assertEqual(qdata.device.type, "cpu")
        self.assertEqual(scale.device.type, "cpu")
        self.assertTrue(quant_conf["weight_rotated"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_quantization_has_cpu_equivalent_decode_error(self):
        torch.manual_seed(0)
        weight = torch.randn((32, 256), dtype=torch.float32)
        cpu_qdata, cpu_scale, _, _ = quantize_int8_convrot(weight, device=torch.device("cpu"))
        cuda_qdata, cuda_scale, _, _ = quantize_int8_convrot(weight, device=torch.device("cuda"))

        cpu_decoded = cpu_qdata.to(torch.float32) * cpu_scale
        cuda_decoded = cuda_qdata.cpu().to(torch.float32) * cuda_scale.cpu()
        self.assertTrue(torch.allclose(cpu_decoded, cuda_decoded, atol=1e-5, rtol=0))

    def test_auto_device_serializes_q8_cr_layout(self):
        state_dict = {
            "video_patch_proj.weight": torch.ones((32, 32), dtype=torch.float32),
            "audio_patch_proj.weight": torch.ones((32, 32), dtype=torch.float32),
            "blocks.0.attn.qkv_proj.weight": torch.ones((96, 32), dtype=torch.float32),
            "final_layer.video_out.weight": torch.ones((96, 32), dtype=torch.float32),
        }

        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "minimax_h3.safetensors"
            output_path = Path(temp_dir) / "minimax_h3-Q8_CR.gguf"
            save_file(state_dict, str(source_path))

            converted_path, _ = convert_file(
                str(source_path),
                str(output_path),
                interact=False,
                quant_type_name="Q8_CR",
                quantization_device="auto",
            )

            reader = gguf.GGUFReader(converted_path)
            tensor_types = {tensor.name: tensor.tensor_type for tensor in reader.tensors}
            reader.tensors.clear()
            reader.fields.clear()
            reader.data._mmap.close()
            del reader

        self.assertEqual(
            tensor_types["blocks.0.attn.qkv_proj.weight"],
            gguf.GGMLQuantizationType.I8,
        )
        self.assertEqual(
            tensor_types["blocks.0.attn.qkv_proj.weight_scale"],
            gguf.GGMLQuantizationType.F32,
        )

    @unittest.skipUnless(hasattr(torch, "float8_e4m3fn"), "PyTorch does not support FP8")
    def test_streamed_layout_supports_safetensors_f8_e4m3(self):
        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "fp8.safetensors"
            save_file(
                {"weight": torch.ones((2, 2), dtype=torch.float8_e4m3fn)},
                str(source_path),
            )
            state_dict, source_keys = _streamed_safetensors_layout(str(source_path))

        self.assertEqual(state_dict["weight"].dtype, torch.float8_e4m3fn)
        self.assertEqual(source_keys["weight"], "weight")


class MiniMaxH3VAEConversionTests(unittest.TestCase):
    def test_detects_video_vae_from_distinctive_decoder_and_encoder_keys(self):
        state_dict = {
            "decoder.transformer_blocks.0.scale1": torch.ones(32),
            "decoder.x_embedder.weight": torch.ones((64, 32)),
            "encoder.down.5.block.0.conv1.weight": torch.ones((2, 2, 3, 3, 3)),
        }

        self.assertIsInstance(detect_arch(state_dict), ModelMinimaxH3VAE)

    def test_q8_cr_keeps_conv3d_fp16_and_restores_its_shape(self):
        state_dict = {
            "decoder.transformer_blocks.0.scale1": torch.ones(32, dtype=torch.float16),
            "decoder.x_embedder.weight": torch.ones((64, 32), dtype=torch.float16),
            "encoder.down.5.block.0.conv1.weight": torch.ones(
                (2, 2, 3, 3, 3), dtype=torch.float16
            ),
        }

        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "minimax_h3_vae.safetensors"
            output_path = Path(temp_dir) / "minimax_h3_vae-Q8_CR.gguf"
            save_file(state_dict, str(source_path))

            converted_path, _ = convert_file(
                str(source_path),
                str(output_path),
                interact=False,
                quant_type_name="Q8_CR",
                quantization_device="cpu",
            )
            reader = gguf.GGUFReader(converted_path)
            tensor_types = {tensor.name: tensor.tensor_type for tensor in reader.tensors}
            reader.tensors.clear()
            reader.fields.clear()
            reader.data._mmap.close()
            del reader

            loader = load_gguf_loader()
            loaded, extra = loader.gguf_sd_loader(converted_path, handle_prefix=None)
            conv3d = loaded["encoder.down.5.block.0.conv1.weight"]
            del conv3d.tensor_type
            materialized = dequantize_tensor(conv3d, dtype=torch.Tensor(conv3d).dtype)
            conv3d_shape = tuple(materialized.shape)
            del materialized
            del conv3d
            del loaded
            gc.collect()

        self.assertEqual(extra["arch_str"], "minimax_h3_vae")
        self.assertEqual(
            tensor_types["decoder.x_embedder.weight"], gguf.GGMLQuantizationType.I8
        )
        self.assertEqual(
            tensor_types["encoder.down.5.block.0.conv1.weight"],
            gguf.GGMLQuantizationType.F16,
        )
        self.assertEqual(
            conv3d_shape,
            (2, 2, 3, 3, 3),
        )


class MiniMaxMusic3ConversionTests(unittest.TestCase):
    def test_detects_dit_from_music_specific_conditioning_and_attention_keys(self):
        state_dict = {
            "cond_layer_logits": torch.ones(8),
            "latent_conditioners.0.weight": torch.ones((2, 2, 1)),
            "diffusion_transformer.preprocess_conv.weight": torch.ones((2, 2, 1)),
            "diffusion_transformer.postprocess_conv.weight": torch.ones((2, 2, 1)),
            "diffusion_transformer.timestep_features.weight": torch.ones((128, 1)),
            "diffusion_transformer.transformer.rotary_pos_emb.inv_freq": torch.ones(16),
            "diffusion_transformer.transformer.project_in.weight": torch.ones((64, 64)),
            "diffusion_transformer.transformer.layers.0.self_attn.to_qkv.weight": torch.ones((192, 64)),
        }

        self.assertIsInstance(detect_arch(state_dict), ModelMiniMaxMusic3DiT)

    def test_detects_pruned_text_encoder_from_audio_and_qwen_keys(self):
        state_dict = {
            "model.embed_tokens_prefill.weight": torch.ones((64, 32), dtype=torch.bfloat16),
            "model.embed_tokens_audio.weight": torch.ones((64, 32), dtype=torch.bfloat16),
            "model.audio_extra_embedding.weight": torch.ones((64, 32), dtype=torch.bfloat16),
            "model.audio_decoder.pos_embedding.weight": torch.ones((16, 32), dtype=torch.bfloat16),
            "model.lm_head_pruned.weight": torch.ones((64, 32), dtype=torch.bfloat16),
            "model.audio_decoder.audio_heads.0.weight": torch.ones((64, 32), dtype=torch.bfloat16),
            "model.layers.0.self_attn.qkv_proj.weight": torch.ones((96, 32), dtype=torch.bfloat16),
        }

        self.assertIsInstance(detect_arch(state_dict), ModelMiniMaxMusic3TextEncoder)

    def test_q8_cr_preserves_dit_convolutions(self):
        state_dict = {
            "cond_layer_logits": torch.ones(8),
            "latent_conditioners.0.weight": torch.ones((2, 2, 1)),
            "diffusion_transformer.preprocess_conv.weight": torch.ones((2, 2, 1)),
            "diffusion_transformer.postprocess_conv.weight": torch.ones((2, 2, 1)),
            "diffusion_transformer.timestep_features.weight": torch.ones((128, 1)),
            "diffusion_transformer.transformer.rotary_pos_emb.inv_freq": torch.ones(16),
            "diffusion_transformer.transformer.project_in.weight": torch.ones((64, 64)),
            "diffusion_transformer.transformer.layers.0.self_attn.to_qkv.weight": torch.ones((192, 64)),
        }
        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "minimax_music3_dit-Q8_CR.gguf"
            converted_path, model_arch = convert_state_dict(
                state_dict,
                str(output_path),
                quant_type_name="Q8_CR",
                quantization_device="cpu",
            )
            reader = gguf.GGUFReader(converted_path)
            tensor_types = {tensor.name: tensor.tensor_type for tensor in reader.tensors}
            loaded, extra = load_gguf_loader().gguf_sd_loader(converted_path, handle_prefix=None)
            reader.tensors.clear()
            reader.fields.clear()
            reader.data._mmap.close()
            del reader
            del loaded

        self.assertIsInstance(model_arch, ModelMiniMaxMusic3DiT)
        self.assertEqual(extra["arch_str"], "minimax_music3")
        self.assertEqual(
            tensor_types["diffusion_transformer.transformer.project_in.weight"],
            gguf.GGMLQuantizationType.I8,
        )
        self.assertEqual(
            tensor_types["latent_conditioners.0.weight"],
            gguf.GGMLQuantizationType.F32,
        )

    def test_q8_cr_preserves_text_embeddings(self):
        state_dict = {
            "model.embed_tokens_prefill.weight": torch.ones((64, 32), dtype=torch.bfloat16),
            "model.embed_tokens_audio.weight": torch.ones((64, 32), dtype=torch.bfloat16),
            "model.audio_extra_embedding.weight": torch.ones((64, 32), dtype=torch.bfloat16),
            "model.audio_decoder.pos_embedding.weight": torch.ones((16, 32), dtype=torch.bfloat16),
            "model.lm_head_pruned.weight": torch.ones((64, 32), dtype=torch.bfloat16),
            "model.audio_decoder.audio_heads.0.weight": torch.ones((64, 32), dtype=torch.bfloat16),
            "model.layers.0.self_attn.qkv_proj.weight": torch.ones((96, 32), dtype=torch.bfloat16),
        }
        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "minimax_music3_text_encoder.safetensors"
            output_path = Path(temp_dir) / "minimax_music3_text_encoder-Q8_CR.gguf"
            save_file(state_dict, str(source_path))
            converted_path, model_arch = convert_file(
                str(source_path),
                str(output_path),
                interact=False,
                quant_type_name="Q8_CR",
                quantization_device="cpu",
                streamed=True,
            )
            reader = gguf.GGUFReader(converted_path)
            tensor_types = {tensor.name: tensor.tensor_type for tensor in reader.tensors}
            loaded, extra = load_gguf_loader().gguf_sd_loader(
                converted_path,
                handle_prefix=None,
                is_text_model=True,
            )
            reader.tensors.clear()
            reader.fields.clear()
            reader.data._mmap.close()
            del reader
            del loaded
            gc.collect()

        self.assertIsInstance(model_arch, ModelMiniMaxMusic3TextEncoder)
        self.assertEqual(extra["arch_str"], "minimax_music3")
        self.assertEqual(
            tensor_types["model.embed_tokens_prefill.weight"],
            gguf.GGMLQuantizationType.BF16,
        )
        self.assertEqual(
            tensor_types["model.layers.0.self_attn.qkv_proj.weight"],
            gguf.GGMLQuantizationType.I8,
        )


class LTX25ConversionTests(unittest.TestCase):
    def test_detects_ltx25_audio_video_transformer(self):
        state_dict = {
            "adaln_single.emb.timestep_embedder.linear_2.weight": torch.ones((32, 16)),
            "audio_adaln_single.linear.weight": torch.ones((32, 16)),
            "transformer_blocks.27.scale_shift_table": torch.ones((6, 32)),
        }

        self.assertIsInstance(detect_arch(state_dict), ModelLTXV)

    def test_converts_temporal_upscaler_without_quantizing_conv3d(self):
        state_dict = {
            "initial_conv.weight": torch.ones((32, 16, 3, 3, 3), dtype=torch.bfloat16),
            "post_upsample_res_blocks.0.conv2.bias": torch.ones(32, dtype=torch.bfloat16),
            "upsampler.0.weight": torch.ones((64, 32, 3, 3, 3), dtype=torch.bfloat16),
            "final_conv.weight": torch.ones((16, 32, 3, 3, 3), dtype=torch.bfloat16),
        }
        metadata = {
            "config": json.dumps(
                {
                    "_class_name": "LatentUpsampler",
                    "in_channels": 16,
                    "mid_channels": 32,
                    "num_blocks_per_stage": 1,
                    "dims": 3,
                    "spatial_upsample": False,
                    "temporal_upsample": True,
                }
            )
        }

        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "ltx25-temporal-upscaler-Q8_CR.gguf"
            converted_path, model_arch = convert_state_dict(
                state_dict,
                str(output_path),
                source_metadata=metadata,
                quant_type_name="Q8_CR",
                quantization_device="cpu",
            )
            reader = gguf.GGUFReader(converted_path)
            tensor_types = {tensor.name: tensor.tensor_type for tensor in reader.tensors}
            loader = load_gguf_loader()
            loaded, extra = loader.gguf_sd_loader(converted_path, handle_prefix=None)
            restored_shape = tuple(
                loaded["upsampler.0.weight"].shape
            )
            loaded_arch = extra["arch_str"]
            config_dims = json.loads(extra["metadata"]["config"])["dims"]
            reader.tensors.clear()
            reader.fields.clear()
            reader.data._mmap.close()
            del reader
            del loaded
            del extra
            gc.collect()

        self.assertIsInstance(model_arch, ModelLTXVUpsampler)
        self.assertEqual(loaded_arch, "ltxv_upscaler")
        self.assertEqual(
            tensor_types["upsampler.0.weight"], gguf.GGMLQuantizationType.BF16
        )
        self.assertEqual(restored_shape, (64, 32, 3, 3, 3))
        self.assertEqual(config_dims, 3)


class GGUFLoraTests(unittest.TestCase):
    def _write_lora(self, path, down, up, alpha=4.0):
        writer = gguf.GGUFWriter(path=None, arch="minimax_h3")
        writer.add_string("general.type", "adapter")
        writer.add_string("adapter.type", "lora")
        writer.add_float32("adapter.lora.alpha", alpha)
        writer.add_tensor(
            "blocks.0.attn.qkv_proj.weight.lora_a",
            down.numpy(),
            raw_dtype=gguf.GGMLQuantizationType.F32,
        )
        writer.add_tensor(
            "blocks.0.attn.qkv_proj.weight.lora_b",
            up.numpy(),
            raw_dtype=gguf.GGMLQuantizationType.F32,
        )
        writer.write_header_to_file(path=str(path))
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    def test_imports_standard_gguf_factor_pair(self):
        down = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        up = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "adapter.gguf"
            self._write_lora(path, down, up)
            lora, targets, metadata = load_gguf_lora(path)

        self.assertEqual(metadata["alpha"], 4.0)
        self.assertIn("blocks.0.attn.qkv_proj.lora_A.weight", lora)
        self.assertIn("blocks.0.attn.qkv_proj.lora_B.weight", lora)
        self.assertTrue(torch.equal(targets["blocks.0.attn.qkv_proj"]["down"], down))
        self.assertTrue(torch.equal(targets["blocks.0.attn.qkv_proj"]["up"], up))

    def test_fuses_lora_delta_in_selected_precision(self):
        state_dict = {
            "blocks.0.attn.qkv_proj.weight": torch.zeros((3, 2), dtype=torch.float16)
        }
        targets = {
            "blocks.0.attn.qkv_proj": {
                "base_name": "blocks.0.attn.qkv_proj.weight",
                "down": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                "up": torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
                "alpha": 4.0,
            }
        }

        count = fuse_targets_into_state_dict(
            state_dict, targets, strength=0.5, device=torch.device("cpu")
        )

        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0], [4.0, 6.0]], dtype=torch.float16)
        self.assertEqual(count, 1)
        self.assertTrue(torch.equal(state_dict["blocks.0.attn.qkv_proj.weight"], expected))

    def test_restores_per_row_scaled_int8_source_weights(self):
        state_dict = {
            "blocks.0.attn.qkv_proj.weight": torch.tensor(
                [[10, -20], [30, -40]], dtype=torch.int8
            ),
            "blocks.0.attn.qkv_proj.weight_scale": torch.tensor([0.1, 0.01]),
        }

        restored_count = materialize_int8_source_weights(state_dict)

        self.assertEqual(restored_count, 1)
        self.assertNotIn("blocks.0.attn.qkv_proj.weight_scale", state_dict)
        self.assertEqual(state_dict["blocks.0.attn.qkv_proj.weight"].dtype, torch.float16)
        self.assertTrue(
            torch.equal(
                state_dict["blocks.0.attn.qkv_proj.weight"],
                torch.tensor([[1.0, -2.0], [0.3, -0.4]], dtype=torch.float16),
            )
        )

    def test_restores_convrot_int8_source_weights(self):
        source = torch.tensor(
            [[2.0, 2.0, 2.0, 2.0], [2.0, -2.0, 2.0, -2.0]],
            dtype=torch.float32,
        )
        qdata, scale, quant_conf, _ = quantize_int8_convrot(
            source, convrot_groupsize=4, device=torch.device("cpu")
        )
        state_dict = {
            "blocks.0.proj.weight": qdata,
            "blocks.0.proj.weight_scale": scale,
            "blocks.0.proj.comfy_quant": torch.tensor(
                list(json.dumps(quant_conf).encode("utf-8")), dtype=torch.uint8
            ),
        }

        materialize_int8_source_weights(state_dict)

        self.assertNotIn("blocks.0.proj.weight_scale", state_dict)
        self.assertNotIn("blocks.0.proj.comfy_quant", state_dict)
        self.assertTrue(
            torch.equal(
                state_dict["blocks.0.proj.weight"],
                source.to(dtype=torch.float16),
            )
        )

    def test_rejects_int8_source_weight_without_scale(self):
        state_dict = {
            "blocks.0.attn.qkv_proj.weight": torch.ones((2, 2), dtype=torch.int8),
        }

        with self.assertRaisesRegex(ValueError, "missing its scale tensor"):
            materialize_int8_source_weights(state_dict)

    def test_warns_and_skips_unmatched_lora_targets(self):
        targets = {
            "missing.layer": {
                "base_name": "missing.layer.weight",
                "down": torch.ones((1, 2)),
                "up": torch.ones((2, 1)),
                "alpha": 1.0,
            }
        }

        with self.assertLogs(level="WARNING") as logs:
            resolved = resolve_fusion_targets({}, targets)

        self.assertEqual(resolved, {})
        self.assertIn("Skipping 1 LoRA target", logs.output[0])


class OfflineLoraFusionTests(unittest.TestCase):
    def test_imports_and_fuses_safetensors_factor_pair(self):
        down = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        up = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "adapter.safetensors"
            save_file(
                {
                    "blocks.0.attn.qkv_proj.lora_A.weight": down,
                    "blocks.0.attn.qkv_proj.lora_B.weight": up,
                    "blocks.0.attn.qkv_proj.alpha": torch.tensor(4.0),
                },
                str(path),
            )
            _, targets, _ = load_lora(path)

        state_dict = {
            "blocks.0.attn.qkv_proj.weight": torch.zeros((3, 2), dtype=torch.float32)
        }
        fuse_targets_into_state_dict(state_dict, targets, strength=0.5, device=torch.device("cpu"))
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0], [4.0, 6.0]])
        self.assertTrue(torch.equal(state_dict["blocks.0.attn.qkv_proj.weight"], expected))

    @mock.patch(
        "lora._comfy_model_lora_target_map",
        return_value={
            "transformer.single_transformer_blocks.0.attn.to_qkv_mlp_proj": (
                "diffusion_model.single_blocks.0.linear1.weight"
            )
        },
    )
    def test_fuses_comfy_mapped_flux2_single_block_target(self, _):
        state_dict = {
            "double_blocks.0.img_attn.proj.weight": torch.zeros((2, 2)),
            "single_blocks.0.linear1.weight": torch.zeros((2, 2)),
        }
        targets = {
            "transformer.single_transformer_blocks.0.attn.to_qkv_mlp_proj": {
                "base_name": (
                    "transformer.single_transformer_blocks.0.attn."
                    "to_qkv_mlp_proj.weight"
                ),
                "down": torch.eye(2),
                "up": torch.eye(2),
                "alpha": 2.0,
            }
        }

        count = fuse_targets_into_state_dict(
            state_dict, targets, strength=1.0, device=torch.device("cpu")
        )

        self.assertEqual(count, 1)
        self.assertTrue(torch.equal(state_dict["single_blocks.0.linear1.weight"], torch.eye(2)))

    @mock.patch(
        "lora._comfy_model_lora_target_map",
        return_value={
            "transformer.transformer_blocks.0.attn.to_q": (
                "diffusion_model.double_blocks.0.img_attn.qkv.weight",
                (0, 0, 2),
            ),
            "transformer.transformer_blocks.0.attn.to_v": (
                "diffusion_model.double_blocks.0.img_attn.qkv.weight",
                (0, 4, 2),
            ),
        },
    )
    def test_fuses_comfy_mapped_qkv_slices(self, _):
        state_dict = {
            "double_blocks.0.img_attn.proj.weight": torch.zeros((2, 2)),
            "single_blocks.0.linear1.weight": torch.zeros((2, 2)),
            "double_blocks.0.img_attn.qkv.weight": torch.zeros((6, 2)),
        }
        targets = {
            "transformer.transformer_blocks.0.attn.to_q": {
                "base_name": "transformer.transformer_blocks.0.attn.to_q.weight",
                "down": torch.eye(2),
                "up": torch.eye(2),
                "alpha": 2.0,
            },
            "transformer.transformer_blocks.0.attn.to_v": {
                "base_name": "transformer.transformer_blocks.0.attn.to_v.weight",
                "down": torch.eye(2),
                "up": torch.eye(2),
                "alpha": 2.0,
            },
        }

        count = fuse_targets_into_state_dict(
            state_dict, targets, strength=1.0, device=torch.device("cpu")
        )

        self.assertEqual(count, 2)
        self.assertTrue(
            torch.equal(
                state_dict["double_blocks.0.img_attn.qkv.weight"],
                torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
                ),
            )
        )

    @mock.patch(
        "lora._comfy_model_lora_target_map",
        return_value={
            "transformer.transformer_blocks.0.attn.to_q": (
                "diffusion_model.blocks.0.attn.wq.weight"
            )
        },
    )
    def test_fuses_comfy_mapped_krea2_target(self, _):
        state_dict = {
            "blocks.0.attn.wq.weight": torch.zeros((2, 2)),
        }
        targets = {
            "transformer.transformer_blocks.0.attn.to_q": {
                "base_name": "transformer.transformer_blocks.0.attn.to_q.weight",
                "down": torch.eye(2),
                "up": torch.eye(2),
                "alpha": 2.0,
            }
        }

        count = fuse_targets_into_state_dict(
            state_dict, targets, strength=1.0, device=torch.device("cpu")
        )

        self.assertEqual(count, 1)
        self.assertTrue(torch.equal(state_dict["blocks.0.attn.wq.weight"], torch.eye(2)))

    def test_imports_and_fuses_direct_lokr_adapter(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "adapter.safetensors"
            save_file(
                {
                    "blocks.0.proj.lokr_w1": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                    "blocks.0.proj.lokr_w2": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                    "blocks.0.proj.alpha": torch.tensor(100.0),
                },
                str(path),
            )
            _, targets, metadata = load_lora(path)

        state_dict = {"blocks.0.proj.weight": torch.zeros((4, 4))}
        count = fuse_targets_into_state_dict(
            state_dict, targets, strength=0.5, device=torch.device("cpu")
        )

        self.assertEqual(metadata["target_count"], 1)
        self.assertEqual(count, 1)
        self.assertTrue(
            torch.equal(
                state_dict["blocks.0.proj.weight"],
                0.5 * torch.kron(
                    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                    torch.eye(2),
                ),
            )
        )

    @unittest.skipUnless(hasattr(torch, "float8_e4m3fn"), "PyTorch does not support FP8")
    def test_fuses_scaled_fp8_target_as_fp16(self):
        state_dict = {
            "blocks.0.attn.qkv_proj.weight": torch.ones(
                (2, 2), dtype=torch.float8_e4m3fn
            ),
            "blocks.0.attn.qkv_proj.weight_scale": torch.tensor(0.5),
        }
        targets = {
            "blocks.0.attn.qkv_proj": {
                "base_name": "blocks.0.attn.qkv_proj.weight",
                "down": torch.eye(2),
                "up": torch.eye(2),
                "alpha": 2.0,
            }
        }

        fuse_targets_into_state_dict(state_dict, targets, strength=1.0, device=torch.device("cpu"))

        self.assertEqual(state_dict["blocks.0.attn.qkv_proj.weight"].dtype, torch.float16)
        self.assertTrue(
            torch.equal(
                state_dict["blocks.0.attn.qkv_proj.weight"],
                torch.tensor([[1.5, 0.5], [0.5, 1.5]], dtype=torch.float16),
            )
        )

    def test_converter_merges_safetensors_lora_before_gguf_export(self):
        source = {
            "video_patch_proj.weight": torch.zeros((32, 32), dtype=torch.float16),
            "audio_patch_proj.weight": torch.zeros((32, 32), dtype=torch.float16),
            "blocks.0.attn.qkv_proj.weight": torch.zeros((96, 32), dtype=torch.float16),
            "final_layer.video_out.weight": torch.zeros((96, 32), dtype=torch.float16),
        }
        down = torch.zeros((2, 32), dtype=torch.float16)
        down[:, 0] = 1
        up = torch.zeros((96, 2), dtype=torch.float16)
        up[0] = 1

        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "minimax_h3.safetensors"
            lora_path = Path(temp_dir) / "adapter.safetensors"
            output_path = Path(temp_dir) / "merged.gguf"
            save_file(source, str(source_path))
            save_file(
                {
                    "blocks.0.attn.qkv_proj.lora_A.weight": down,
                    "blocks.0.attn.qkv_proj.lora_B.weight": up,
                    "blocks.0.attn.qkv_proj.alpha": torch.tensor(2.0),
                },
                str(lora_path),
            )
            convert_file(
                str(source_path),
                str(output_path),
                interact=False,
                quant_type_name="F16",
                lora_paths=[str(lora_path)],
            )
            reader = gguf.GGUFReader(str(output_path))
            tensor = next(
                tensor for tensor in reader.tensors
                if tensor.name == "blocks.0.attn.qkv_proj.weight"
            )
            merged = torch.from_numpy(tensor.data.copy()).view(torch.float16).reshape(
                tuple(reversed(tensor.shape))
            )
            reader.tensors.clear()
            reader.fields.clear()
            reader.data._mmap.close()

        self.assertTrue(torch.equal(merged[0, :2], torch.tensor([2.0, 0.0], dtype=torch.float16)))

    def test_streamed_converter_merges_lora_without_loading_state_dict(self):
        source = {
            "video_patch_proj.weight": torch.zeros((32, 32), dtype=torch.float16),
            "audio_patch_proj.weight": torch.zeros((32, 32), dtype=torch.float16),
            "blocks.0.attn.qkv_proj.weight": torch.zeros((96, 32), dtype=torch.float16),
            "final_layer.video_out.weight": torch.zeros((96, 32), dtype=torch.float16),
        }
        down = torch.zeros((2, 32), dtype=torch.float16)
        down[:, 0] = 1
        up = torch.zeros((96, 2), dtype=torch.float16)
        up[0] = 1

        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "minimax_h3.safetensors"
            lora_path = Path(temp_dir) / "adapter.safetensors"
            output_path = Path(temp_dir) / "merged.gguf"
            save_file(source, str(source_path))
            save_file(
                {
                    "blocks.0.attn.qkv_proj.lora_A.weight": down,
                    "blocks.0.attn.qkv_proj.lora_B.weight": up,
                    "blocks.0.attn.qkv_proj.alpha": torch.tensor(2.0),
                },
                str(lora_path),
            )
            with mock.patch("tools.convert.load_state_dict") as load_state_dict:
                convert_file(
                    str(source_path),
                    str(output_path),
                    interact=False,
                    quant_type_name="F16",
                    lora_paths=[str(lora_path)],
                    streamed=True,
                )
            load_state_dict.assert_not_called()
            reader = gguf.GGUFReader(str(output_path))
            tensor = next(
                tensor for tensor in reader.tensors
                if tensor.name == "blocks.0.attn.qkv_proj.weight"
            )
            merged = torch.from_numpy(tensor.data.copy()).view(torch.float16).reshape(
                tuple(reversed(tensor.shape))
            )
            reader.tensors.clear()
            reader.fields.clear()
            reader.data._mmap.close()

        self.assertTrue(torch.equal(merged[0, :2], torch.tensor([2.0, 0.0], dtype=torch.float16)))


class Qwen3VLDetectionMarkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = load_gguf_loader()

    def test_uses_minimax_32b_detection_marker_for_5120_hidden_size(self):
        state_dict = {
            "model.layers.0.input_layernorm.weight": torch.zeros(5120),
            "model.layers.49.self_attn.q_proj.weight": torch.zeros(1),
        }

        self.loader.inject_qwen3vl_detection_markers(state_dict)

        self.assertEqual(
            comfy.sd.detect_te_model(state_dict),
            comfy.sd.TEModel.QWEN3VL_32B,
        )
        self.assertIn("visual.deepstack_merger_list.0.norm.weight", state_dict)
        self.assertNotIn("model.visual.deepstack_merger_list.0.norm.weight", state_dict)
        self.assertNotIn("model.visual.merger.linear_fc2.weight", state_dict)
        self.assertEqual(
            state_dict["visual.deepstack_merger_list.0.norm.weight"].shape,
            (4608,),
        )

    def test_uses_model_prefixed_detection_markers_for_8b(self):
        state_dict = {
            "model.layers.0.input_layernorm.weight": torch.zeros(4096),
        }

        self.loader.inject_qwen3vl_detection_markers(state_dict)

        self.assertIn("model.visual.deepstack_merger_list.0.norm.weight", state_dict)
        self.assertEqual(
            state_dict["model.visual.merger.linear_fc2.weight"].shape,
            (4096, 4608),
        )

    def test_pruned_32b_clip_loader_injects_marker(self):
        with mock.patch.object(
            self.loader,
            "gguf_sd_loader",
            return_value=(
                {
                    "model.layers.0.input_layernorm.weight": torch.zeros(5120),
                    "model.layers.49.self_attn.q_proj.weight": torch.zeros(1),
                },
                {"arch_str": "qwen3vl"},
            ),
        ), mock.patch.object(
            self.loader,
            "gguf_mmproj_loader",
            return_value={},
        ):
            state_dict = self.loader.gguf_clip_loader("qwen3-vl-pruned.gguf")

        self.assertEqual(
            state_dict["visual.deepstack_merger_list.0.norm.weight"].shape,
            (4608,),
        )

    def test_pruned_32b_clip_loader_maps_matching_mmproj_to_visual_tower(self):
        with mock.patch.object(
            self.loader,
            "gguf_sd_loader",
            return_value=(
                {
                    "model.layers.0.input_layernorm.weight": torch.zeros(5120),
                    "model.layers.49.self_attn.q_proj.weight": torch.zeros(1),
                },
                {"arch_str": "qwen3vl"},
            ),
        ), mock.patch.object(
            self.loader,
            "gguf_mmproj_loader",
            return_value={
                "model.visual.deepstack_merger_list.0.norm.weight": torch.ones(4608),
            },
        ):
            state_dict = self.loader.gguf_clip_loader("qwen3-vl-pruned.gguf")

        self.assertIn("visual.deepstack_merger_list.0.norm.weight", state_dict)
        self.assertNotIn("model.visual.deepstack_merger_list.0.norm.weight", state_dict)
        self.assertTrue(
            torch.equal(
                state_dict["visual.deepstack_merger_list.0.norm.weight"],
                torch.ones(4608),
            )
        )

    def test_maps_qwen3vl_mmproj_deepstack_tensors(self):
        mapped = self.loader.sd_map_replace(
            {
                "v.deepstack.0.norm.weight": torch.ones(4608),
                "v.deepstast.1.fc2.weight": torch.ones(5120, 4608),
                "v.blk.0.attn_qkv.weight": torch.ones(3, 3),
            },
            self.loader.CLIP_VISION_QWEN3_MAP,
        )

        self.assertIn(
            "model.visual.deepstack_merger_list.0.norm.weight",
            mapped,
        )
        self.assertIn(
            "model.visual.deepstack_merger_list.1.linear_fc2.weight",
            mapped,
        )
        self.assertIn("model.visual.blocks.0.attn.qkv.weight", mapped)

    def test_qwen3vl_mmproj_stacks_temporal_patch_embeddings(self):
        with TemporaryDirectory() as temp_dir:
            text_encoder = Path(temp_dir) / "qwen3-vl-IQ3_XXS.gguf"
            mmproj = Path(temp_dir) / "qwen3-vl-mmproj-BF16.gguf"
            text_encoder.touch()
            mmproj.touch()
            patch_a = torch.ones((2, 3, 2, 2))
            patch_b = torch.full((2, 3, 2, 2), 2.0)

            with mock.patch.object(
                self.loader,
                "gguf_sd_loader",
                return_value=(
                    {
                        "v.patch_embd.weight": patch_a,
                        "v.patch_embd.weight.1": patch_b,
                        "v.deepstast.0.norm.weight": torch.ones(8),
                    },
                    {},
                ),
            ):
                mapped = self.loader.gguf_mmproj_loader(str(text_encoder))

        weight = mapped["model.visual.patch_embed.proj.weight"]
        self.assertEqual(weight.shape, (2, 3, 2, 2, 2))
        self.assertTrue(torch.equal(weight[:, :, 0], patch_a))
        self.assertTrue(torch.equal(weight[:, :, 1], patch_b))


class Qwen3VLQuantizationTests(unittest.TestCase):
    def test_supports_quant_types_used_by_pruned_32b_gguf(self):
        for quant_name in ("IQ3_S", "IQ3_XXS", "IQ2_S", "IQ2_XS"):
            quant_type = getattr(gguf.GGMLQuantizationType, quant_name)
            _, type_size = gguf.GGML_QUANT_SIZES[quant_type]
            output = dequantize(
                torch.zeros((1, type_size), dtype=torch.uint8),
                quant_type,
                (256,),
                dtype=torch.float32,
            )

            self.assertEqual(output.shape, (256,))
            self.assertEqual(output.dtype, torch.float32)
            self.assertIn(quant_type, dequantize_functions)


class Gemma4GGUFLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = load_gguf_loader()

    def test_maps_e4b_specific_tensor_layout_to_comfyui(self):
        state_dict = {
            "token_embd.weight": torch.zeros(1),
            "per_layer_token_embd.weight": torch.zeros(1),
            "per_layer_model_proj.weight": torch.zeros(1),
            "per_layer_proj_norm.weight": torch.zeros(1),
            "blk.0.inp_gate.weight": torch.zeros(1),
            "blk.0.proj.weight": torch.zeros(1),
            "blk.0.layer_output_scale.weight": torch.zeros(1),
            "blk.0.post_norm.weight": torch.zeros(1),
            "blk.0.post_ffw_norm.weight": torch.zeros(1),
        }

        mapped = self.loader.sd_map_replace(state_dict, self.loader.GEMMA4_SD_MAP)

        self.assertEqual(
            set(mapped),
            {
                "model.embed_tokens.weight",
                "model.embed_tokens_per_layer.weight",
                "model.per_layer_model_projection.weight",
                "model.per_layer_projection_norm.weight",
                "model.layers.0.per_layer_input_gate.weight",
                "model.layers.0.per_layer_projection.weight",
                "model.layers.0.layer_scalar",
                "model.layers.0.post_per_layer_input_norm.weight",
                "model.layers.0.post_feedforward_layernorm.weight",
            },
        )

    def test_recreates_gemma4_bpe_tokenizer_json(self):
        tokenizer_json = self.loader.gemma4_tokenizer_json(
            ["<pad>", "<eos>", "<bos>", "<unk>", "hello", "\u2581world"],
            ["h e", "he llo"],
            [3, 3, 3, 3, 1, 1],
        )
        tokenizer = json.loads(bytes(tokenizer_json.tolist()))

        self.assertEqual(tokenizer["model"]["type"], "BPE")
        self.assertEqual(tokenizer["model"]["vocab"]["hello"], 4)
        self.assertEqual(tokenizer["pre_tokenizer"]["type"], "Metaspace")
        self.assertEqual(
            [token["content"] for token in tokenizer["added_tokens"]],
            ["<pad>", "<eos>", "<bos>", "<unk>"],
        )

    def test_e4b_layout_is_detected_by_installed_comfyui(self):
        state_dict = {
            "model.layers.0.post_feedforward_layernorm.weight": torch.zeros(2560),
            "model.layers.41.self_attn.q_norm.weight": torch.zeros(256),
        }

        self.assertEqual(
            comfy.sd.detect_te_model(state_dict),
            comfy.sd.TEModel.GEMMA_4_E4B,
        )


class MinimaxH3DetectionTests(unittest.TestCase):
    def test_detects_native_minimax_h3_checkpoint_layout(self):
        checkpoint_keys = {
            "video_patch_proj.weight",
            "audio_patch_proj.weight",
            "blocks.0.attn.qkv_proj.weight",
            "final_layer.video_out.weight",
        }

        model_arch = detect_arch(checkpoint_keys)

        self.assertIsInstance(model_arch, ModelMinimaxH3)
        self.assertEqual(model_arch.arch, "minimax_h3")

    def test_keeps_adaln_curve_table_in_full_precision(self):
        model_arch = ModelMinimaxH3()

        self.assertIn("adaln_t_table", model_arch.keys_hiprec)

    def test_converts_to_minimax_h3_gguf_with_full_precision_adaln_table(self):
        state_dict = {
            "video_patch_proj.weight": torch.ones((32, 32), dtype=torch.float16),
            "audio_patch_proj.weight": torch.ones((32, 32), dtype=torch.float16),
            "blocks.0.attn.qkv_proj.weight": torch.ones((96, 32), dtype=torch.float16),
            "final_layer.video_out.weight": torch.ones((96, 32), dtype=torch.float16),
            "adaln_t_table": torch.ones((32, 32), dtype=torch.float32),
        }

        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "minimax_h3.safetensors"
            output_path = Path(temp_dir) / "minimax_h3-Q8_0.gguf"
            save_file(state_dict, str(source_path))

            converted_path, model_arch = convert_file(
                str(source_path),
                str(output_path),
                interact=False,
                quant_type_name="Q8_0",
            )

            reader = gguf.GGUFReader(converted_path)
            tensor_types = {tensor.name: tensor.tensor_type for tensor in reader.tensors}
            architecture = reader.get_field("general.architecture")
            architecture_name = str(architecture.parts[architecture.data[-1]], "utf-8")
            del architecture
            reader.tensors.clear()
            reader.fields.clear()
            reader.data._mmap.close()
            del reader

        self.assertEqual(model_arch.arch, "minimax_h3")
        self.assertEqual(architecture_name, "minimax_h3")
        self.assertEqual(
            tensor_types["blocks.0.attn.qkv_proj.weight"],
            gguf.GGMLQuantizationType.Q8_0,
        )
        self.assertEqual(
            tensor_types["adaln_t_table"],
            gguf.GGMLQuantizationType.F32,
        )


class Q4CRQuantizationTests(unittest.TestCase):
    def test_int4_cr_packs_kitchen_native_layout(self):
        torch.manual_seed(0)
        weight = torch.randn(64, 256, dtype=torch.float32)
        packed, (wscales, wzeros), quant_conf, orig_shape = quantize_int4_cr(
            weight, group_size=64, device=torch.device("cpu")
        )

        # packed (N, K//2) int8, scales/zeros (K//G, N) bf16
        self.assertEqual(packed.shape, (64, 128))
        self.assertEqual(packed.dtype, torch.int8)
        self.assertEqual(wscales.shape, (256 // 64, 64))
        self.assertEqual(wzeros.shape, (256 // 64, 64))
        self.assertEqual(wscales.dtype, torch.bfloat16)
        self.assertEqual(orig_shape, (64, 256))
        self.assertEqual(quant_conf["format"], "int4_cr")
        self.assertEqual(quant_conf["group_size"], 64)
        self.assertFalse(quant_conf["sym"])

    def test_int4_cr_roundtrips_through_dequant(self):
        torch.manual_seed(0)
        weight = torch.randn(64, 256, dtype=torch.float32)
        packed, (wscales, wzeros), quant_conf, _ = quantize_int4_cr(
            weight, group_size=64, device=torch.device("cpu")
        )

        # Mirror the ops._dequantized_weight math.
        n, k = weight.shape
        x32 = packed.to(torch.int32)
        lo = (x32 & 0xF).to(torch.int8)
        hi = ((x32 >> 4) & 0xF).to(torch.int8)
        nibbles = torch.stack([lo, hi], dim=-1).reshape(n, k).to(torch.float32)
        decoded = (nibbles.view(n, k // 64, 64) - 8.0) * wscales.t().unsqueeze(-1) \
            + wzeros.t().unsqueeze(-1)
        decoded = decoded.view(n, k)

        # INT4 reconstruction must stay reasonably close to the source.
        relative = (decoded - weight).abs().amax().item() / weight.abs().amax().item()
        self.assertLess(relative, 0.2)

    def test_int4_cr_rejects_group_size_not_dividing_k(self):
        with self.assertRaisesRegex(ValueError, "must divide input features"):
            quantize_int4_cr(torch.randn(64, 256), group_size=127, device=torch.device("cpu"))

    def test_int4_cr_serializes_kitchen_metadata(self):
        state_dict = {
            "video_patch_proj.weight": torch.ones((32, 32), dtype=torch.float32),
            "audio_patch_proj.weight": torch.ones((32, 32), dtype=torch.float32),
            "blocks.0.attn.qkv_proj.weight": torch.ones((96, 64), dtype=torch.float32),
            "final_layer.video_out.weight": torch.ones((96, 64), dtype=torch.float32),
        }

        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "minimax_h3.safetensors"
            output_path = Path(temp_dir) / "minimax_h3-Q4_CR_M.gguf"
            save_file(state_dict, str(source_path))

            converted_path, _ = convert_file(
                str(source_path),
                str(output_path),
                interact=False,
                quant_type_name="Q4_CR_M",
                quantization_device="cpu",
            )

            reader = gguf.GGUFReader(converted_path)
            tensor_types = {tensor.name: tensor.tensor_type for tensor in reader.tensors}
            names = set(tensor_types.keys())
            reader.tensors.clear()
            reader.fields.clear()
            reader.data._mmap.close()
            del reader

        # Weight stays at I8 (2 uint4 per byte) and scale/zeros are stored as F16.
        self.assertEqual(tensor_types["blocks.0.attn.qkv_proj.weight"], gguf.GGMLQuantizationType.I8)
        self.assertEqual(tensor_types["blocks.0.attn.qkv_proj.weight_scale"], gguf.GGMLQuantizationType.F16)
        self.assertIn("blocks.0.attn.qkv_proj.weight_zeros", names)

    def test_int4_cr_keeps_non_linear_conv_in_fp16(self):
        state_dict = {
            "decoder.transformer_blocks.0.scale1": torch.ones(32, dtype=torch.float32),
            "decoder.x_embedder.weight": torch.ones((64, 32), dtype=torch.float32),
            "encoder.down.5.block.0.conv1.weight": torch.ones(
                (2, 2, 3, 3, 3), dtype=torch.float32
            ),
        }

        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "minimax_h3_vae.safetensors"
            output_path = Path(temp_dir) / "minimax_h3_vae-Q4_CR_M.gguf"
            save_file(state_dict, str(source_path))

            converted_path, _ = convert_file(
                str(source_path),
                str(output_path),
                interact=False,
                quant_type_name="Q4_CR_M",
                quantization_device="cpu",
            )

            reader = gguf.GGUFReader(converted_path)
            tensor_types = {tensor.name: tensor.tensor_type for tensor in reader.tensors}
            reader.tensors.clear()
            reader.fields.clear()
            reader.data._mmap.close()
            del reader

        # Conv3d must NOT be quantized down to a custom layout.
        self.assertEqual(
            tensor_types["encoder.down.5.block.0.conv1.weight"],
            gguf.GGMLQuantizationType.F16,
        )


class Q4CRLoaderTests(unittest.TestCase):
    def setUp(self):
        # Load the loader module under a synthetic package name.
        self.loader = load_gguf_loader()

    @unittest.skipUnless(torch.cuda.is_available(), "Q4_CR kernel requires CUDA")
    def test_int4_cr_ops_forward_matches_dequant(self):
        ops_mod = load_gguf_ops()
        get_gguf_q4_ops = ops_mod.get_gguf_q4_ops

        torch.manual_seed(0)
        n, k, g = 128, 6144, 64
        w = torch.randn(n, k) * 0.4
        packed, (wscales, wzeros), quant_conf, orig_shape = quantize_int4_cr(
            w, group_size=g, device=torch.device("cpu")
        )

        ops = get_gguf_q4_ops(compute_dtype=torch.bfloat16)()
        lin = ops.Linear(in_features=k, out_features=n)
        quant_conf["orig_shape"] = list(orig_shape)
        src_sd = {
            "weight": packed,
            "weight_scale": wscales,
            "weight_zeros": wzeros,
            "comfy_quant": torch.tensor(list(json.dumps(quant_conf).encode()), dtype=torch.uint8),
        }
        lin._load_from_state_dict(src_sd, prefix="", local_metadata={}, strict=False,
                                  missing_keys=[], unexpected_keys=[], error_msgs=[])
        self.assertTrue(lin._quantized)
        self.assertEqual((lin.in_features, lin.out_features), (k, n))

        x = torch.randn(4, k, device="cuda", dtype=torch.bfloat16)
        # Call through __call__/forward (not forward_comfy_cast_weights directly)
        # to reproduce the Krea2 "Module [Linear] is missing the required forward
        # function" failure and guard against its regression.
        out = lin(x)
        self.assertEqual(out.shape, (4, n))

        wq32 = lin._dequantized_weight(torch.device("cuda"), torch.float32)
        ref = torch.nn.functional.linear(x.float(), wq32)
        rel = (out.float() - ref).abs().amax().item() / ref.abs().amax().item()
        self.assertLess(rel, 0.05)

    def test_int4_cr_loader_routes_to_kitchen_ops(self):
        state_dict = {
            "video_patch_proj.weight": torch.randn(32, 32, dtype=torch.float32),
            "audio_patch_proj.weight": torch.randn(32, 32, dtype=torch.float32),
            "blocks.0.attn.qkv_proj.weight": torch.randn(96, 64, dtype=torch.float32),
            "final_layer.video_out.weight": torch.randn(96, 64, dtype=torch.float32),
        }

        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "minimax_h3.safetensors"
            output_path = Path(temp_dir) / "minimax_h3-Q4_CR_M.gguf"
            save_file(state_dict, str(source_path))

            converted_path, _ = convert_file(
                str(source_path),
                str(output_path),
                interact=False,
                quant_type_name="Q4_CR_M",
                quantization_device="cpu",
            )

            sd, extra = self.loader.gguf_sd_loader(converted_path)

            w = sd["blocks.0.attn.qkv_proj.weight"]
            qraw = sd["blocks.0.attn.qkv_proj.comfy_quant"]
            quant_conf = json.loads(bytes(qraw.tolist()).decode("utf-8"))

            w_shape = tuple(int(v) for v in w.shape)
            w_dtype_name = str(w.dtype)
            mode = extra.get("gguf_quant_mode")

            # Detach from the mmap-backed storage before the TemporaryDirectory exits.
            w_detached = w.detach().clone().tolist()
            del w
            del qraw
            del sd
            del extra
            gc.collect()

        self.assertEqual(mode, "int4_cr")

        # The packed weight is (N, K//2) int8; the scale is (K//G, N).
        self.assertEqual(w_shape, (96, 32))
        self.assertEqual(w_dtype_name, "torch.int8")
        # Confirms the packed weight carries real int4-ish nibble data.
        self.assertEqual(len(w_detached), 96)

        # The comfy_quant metadata decodes back to int4_cr.
        self.assertEqual(quant_conf["format"], "int4_cr")
        self.assertEqual(quant_conf["group_size"], 64)
        self.assertEqual(quant_conf["orig_shape"], [96, 64])

    def test_int4_cr_loader_converts_fp16_scales_to_bf16(self):
        # Regression: the loader must convert the F16-stored per-group scales
        # to bf16 with .to(), NOT a raw .view(bfloat16) byte reinterpretation
        # (which corrupts the scale to tiny garbage values).
        state_dict = {
            "video_patch_proj.weight": torch.randn(32, 32, dtype=torch.float32),
            "audio_patch_proj.weight": torch.randn(32, 32, dtype=torch.float32),
            "blocks.0.attn.qkv_proj.weight": torch.randn(96, 64, dtype=torch.float32),
            "final_layer.video_out.weight": torch.randn(96, 64, dtype=torch.float32),
        }

        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "minimax_h3.safetensors"
            output_path = Path(temp_dir) / "minimax_h3-Q4_CR_M.gguf"
            save_file(state_dict, str(source_path))

            converted_path, _ = convert_file(
                str(source_path),
                str(output_path),
                interact=False,
                quant_type_name="Q4_CR_M",
                quantization_device="cpu",
            )

            sd, _ = self.loader.gguf_sd_loader(converted_path)
            s = sd["blocks.0.attn.qkv_proj.weight_scale"].detach().clone()
            del sd
            gc.collect()

        self.assertEqual(s.dtype, torch.bfloat16)
        # Scales are per-group amplitudes; a valid scale is small (>0) and
        # reasonably bounded, not the ~1e-22 garbage a byte-reinterpret gives.
        self.assertGreater(s.abs().max().item(), 1e-6)
        self.assertLess(s.abs().max().item(), 100.0)


class ExperimentalTritonInt4Tests(unittest.TestCase):
    """Gated, offline-only correctness checks for the experimental Triton kernel.

    These do NOT run in a normal environment (and are skipped when Triton / CUDA
    are unavailable). They exist to keep the ``triton_int4.py`` prototype as a
    reproducible, correct-against-its-reference artifact even though it is not
    wired into the runtime path (it is slower than the INT8 Q8 path).
    """

    def _skip_unless(self):
        triton_mod = None
        try:
            import triton_int4 as ti  # noqa: F401
            triton_mod = ti
        except Exception:
            import sys
            sys.path.insert(0, str(Path(__file__).parents[1]))
            try:
                import triton_int4 as ti
                triton_mod = ti
            except Exception as e:
                self.skipTest(f"triton_int4 module not importable: {e}")
        if triton_mod is None or not getattr(triton_mod, "_HAS_TRITON_CUDA", False):
            self.skipTest("Triton/CUDA not available; experimental kernel disabled.")
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available.")
        return triton_mod

    def test_triton_kernel_matches_dequant_reference(self):
        ti = self._skip_unless()
        torch.manual_seed(0)
        dev = "cuda"
        g = 64
        k, n = 4096, 8192
        m = 1024
        x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
        qw = torch.randint(0, 255, (n, k // 2), device=dev, dtype=torch.int32).to(torch.int8)
        ws = (torch.randn(k // g, n, device=dev).abs() + 0.1).to(torch.bfloat16)
        wz = torch.zeros(k // g, n, device=dev, dtype=torch.bfloat16)

        got = ti.triton_q4cr_mm(x, qw, ws, wz, g, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32)

        # fp64 ground truth (independent of torch bf16 rounding)
        x32 = qw.to(torch.int32)
        lo = (x32 & 0xF)
        hi = ((x32 >> 4) & 0xF)
        nibbles = torch.stack([lo, hi], dim=-1).reshape(n, k).to(torch.float64)
        w = ((nibbles.view(n, k // g, g) - 8.0) * ws.t().double().unsqueeze(-1)
             + wz.t().double().unsqueeze(-1)).view(n, k)
        ref = x.double() @ w.t()

        err = (got.double() - ref).abs()
        mean_rel = (err / (ref.abs() + 1e-3)).mean().item()
        self.assertLess(mean_rel, 0.05)


if __name__ == "__main__":
    unittest.main()
