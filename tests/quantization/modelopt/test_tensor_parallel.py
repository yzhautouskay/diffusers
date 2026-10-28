# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import save_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from diffusers import Cosmos3OmniTransformer, NVIDIAModelOptConfig
from diffusers.hooks.tensor_parallel import _check_tp_supported, resolve_tp_shard_specs
from diffusers.models._modeling_parallel import TensorParallelConfig
from diffusers.models.model_loading_utils import _load_shard_file_tp, load_model_dict_into_meta
from diffusers.utils import SAFE_WEIGHTS_INDEX_NAME

from ...models.transformers.test_models_transformer_cosmos3 import Cosmos3OmniTransformerTesterConfig
from ...testing_utils import is_modelopt, is_tensor_parallel, require_torch_multi_accelerator, torch_device


@pytest.fixture
def cpu_tp_config(tmp_path):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("This loader test requires Gloo.")
    dist.init_process_group("gloo", init_method=(tmp_path / "rendezvous").as_uri(), rank=0, world_size=1)
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("tp",))
        config = TensorParallelConfig(mesh=mesh)
        config.setup(rank=0, world_size=1, device=torch.device("cpu"), mesh=mesh)
        yield config
    finally:
        dist.destroy_process_group()


@is_modelopt
class TestModelOptTensorParallelLoading:
    @pytest.fixture
    def quantized_checkpoint(self, tmp_path):
        mtq = pytest.importorskip("modelopt.torch.quantization")
        mto = pytest.importorskip("modelopt.torch.opt")
        torch.manual_seed(0)
        source = torch.nn.ModuleDict(
            {
                "projection": torch.nn.Linear(16, 32, bias=False, dtype=torch.bfloat16),
                "time_embedder": torch.nn.Linear(16, 16, dtype=torch.float32),
            }
        )
        quant_config = {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "quantizer_name": "projection.weight_quantizer",
                    "cfg": {"num_bits": (4, 3), "axis": None, "fake_quant": False},
                },
            ],
            "algorithm": "max",
        }
        with torch.no_grad():
            mtq.quantize(source, quant_config)
            mtq.compress(source)
        for module in source.modules():
            if isinstance(module, mtq.nn.TensorQuantizer):
                module.float()
        checkpoint = tmp_path / "fp8.safetensors"
        weights = source.state_dict()
        save_file(weights, checkpoint)
        with torch.device("meta"):
            model = torch.nn.ModuleDict(
                {
                    "projection": torch.nn.Linear(16, 32, bias=False, dtype=torch.bfloat16),
                    "time_embedder": torch.nn.Linear(16, 16, dtype=torch.bfloat16),
                }
            )
            mto.restore_from_modelopt_state(model, mto.modelopt_state(source))
        return model, checkpoint, weights, quant_config

    @pytest.mark.parametrize("style", ["colwise", "rowwise"])
    def test_preserves_fp8_storage_and_scales(self, quantized_checkpoint, cpu_tp_config, style):
        from diffusers import NVIDIAModelOptConfig
        from diffusers.quantizers.modelopt import NVIDIAModelOptQuantizer

        model, checkpoint, weights, quant_config = quantized_checkpoint
        quantizer = NVIDIAModelOptQuantizer(
            NVIDIAModelOptConfig(quant_type="FP8", modelopt_config=quant_config), pre_quantized=True
        )
        specs = resolve_tp_shard_specs(model, {"projection": style}, 1)
        _load_shard_file_tp(
            str(checkpoint),
            model,
            model.state_dict(),
            specs,
            cpu_tp_config,
            dtype=torch.bfloat16,
            hf_quantizer=quantizer,
            keep_in_fp32_modules=["time_embedder"],
            unexpected_keys=[],
        )
        weight = model["projection"].weight
        assert isinstance(weight, DTensor)
        assert weight.dtype == torch.float8_e4m3fn
        torch.testing.assert_close(
            weight.to_local().view(torch.uint8), weights["projection.weight"].view(torch.uint8), atol=0, rtol=0
        )
        for name, buffer in model.named_buffers():
            assert buffer.dtype == torch.float32
            torch.testing.assert_close(buffer, weights[name], atol=0, rtol=0)
        for name, parameter in model["time_embedder"].named_parameters():
            torch.testing.assert_close(parameter, weights[f"time_embedder.{name}"], atol=0, rtol=0)

    def test_regular_loading_preserves_fp32_quantizer_state(self, quantized_checkpoint):
        from diffusers import NVIDIAModelOptConfig
        from diffusers.quantizers.modelopt import NVIDIAModelOptQuantizer

        model, _, weights, quant_config = quantized_checkpoint
        quantizer = NVIDIAModelOptQuantizer(
            NVIDIAModelOptConfig(quant_type="FP8", modelopt_config=quant_config), pre_quantized=True
        )
        load_model_dict_into_meta(
            model,
            weights,
            dtype=torch.bfloat16,
            hf_quantizer=quantizer,
            keep_in_fp32_modules=["time_embedder"],
            unexpected_keys=[],
        )
        assert model["projection"].weight.dtype == torch.float8_e4m3fn
        for name, buffer in model.named_buffers():
            assert buffer.dtype == torch.float32
            torch.testing.assert_close(buffer, weights[name], atol=0, rtol=0)

    @pytest.mark.parametrize(
        "config_kwargs,pre_quantized",
        [
            ({"quant_type": "FP8_FP8"}, False),
            ({"quant_type": "FP8"}, True),
            ({"quant_type": "NVFP4"}, True),
            ({"quant_type": "FP8_FP8", "channel_quantize": 0}, True),
            ({"quant_type": "FP8_FP8", "block_quantize": 16}, True),
        ],
    )
    def test_rejects_unsupported_quantization(self, config_kwargs, pre_quantized):
        mtq = pytest.importorskip("modelopt.torch.quantization")
        from diffusers import NVIDIAModelOptConfig
        from diffusers.quantizers.modelopt import NVIDIAModelOptQuantizer

        modelopt_config = mtq.NVFP4_DEFAULT_CFG if config_kwargs["quant_type"] == "NVFP4" else mtq.FP8_DEFAULT_CFG
        quantizer = NVIDIAModelOptQuantizer(
            NVIDIAModelOptConfig(**config_kwargs, modelopt_config=modelopt_config), pre_quantized=pre_quantized
        )
        with pytest.raises(ValueError, match="does not support tensor-parallel checkpoint loading"):
            _check_tp_supported(
                "ModuleDict",
                {"projection": "colwise"},
                None,
                TensorParallelConfig(tp_degree=2),
                hf_quantizer=quantizer,
            )

    def test_requires_restored_modelopt_state(self, quantized_checkpoint, cpu_tp_config):
        from diffusers import NVIDIAModelOptConfig
        from diffusers.quantizers.modelopt import NVIDIAModelOptQuantizer

        _, checkpoint, _, quant_config = quantized_checkpoint
        with torch.device("meta"):
            model = torch.nn.ModuleDict({"projection": torch.nn.Linear(16, 32, bias=False)})
        quantizer = NVIDIAModelOptQuantizer(
            NVIDIAModelOptConfig(quant_type="FP8", modelopt_config=quant_config), pre_quantized=True
        )
        specs = resolve_tp_shard_specs(model, {"projection": "colwise"}, 1)
        with pytest.raises(ValueError, match="requires a restored ModelOpt FP8 Linear"):
            _load_shard_file_tp(
                str(checkpoint),
                model,
                model.state_dict(),
                specs,
                cpu_tp_config,
                dtype=torch.bfloat16,
                hf_quantizer=quantizer,
                unexpected_keys=[],
            )


class Cosmos3FP8TensorParallelTesterConfig(Cosmos3OmniTransformerTesterConfig):
    def get_init_dict(self):
        return {
            **super().get_init_dict(),
            "hidden_size": 256,
            "intermediate_size": 512,
            "head_dim": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "rope_axes_dim": [12, 10, 10],
            "num_hidden_layers": 1,
            "attention_bias": False,
        }

    def get_dummy_inputs(self):
        inputs = super().get_dummy_inputs(width=16)
        device = inputs["input_ids"].device
        indexes = torch.arange(16, 32, device=device)
        inputs.update(
            input_ids=torch.arange(16, device=device) % self.get_init_dict()["vocab_size"],
            text_indexes=torch.arange(16, device=device),
            position_ids=torch.arange(32, device=device).expand(3, -1),
            und_len=16,
            sequence_length=32,
            vision_tokens=[tokens.to(dtype=torch.bfloat16) for tokens in inputs["vision_tokens"]],
            vision_sequence_indexes=indexes,
            vision_mse_loss_indexes=indexes,
        )
        return inputs


def _cosmos_fp8_tp_worker(rank, world_size, rendezvous, checkpoint):
    import modelopt.torch.quantization.backends.fp8_per_tensor_gemm  # noqa: F401
    from modelopt.torch.opt import enable_huggingface_checkpointing

    from diffusers.pipelines.cosmos.mixed_precision import (
        Cosmos3MixedPrecisionConfig,
        apply_cosmos3_mixed_precision_step,
    )

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    enable_huggingface_checkpointing()
    dist.init_process_group(
        "nccl", init_method=rendezvous, rank=rank, world_size=world_size, timeout=timedelta(minutes=5)
    )
    try:
        reference = Cosmos3OmniTransformer.from_pretrained(checkpoint, dtype=torch.bfloat16).to(device).eval()
        native = Cosmos3OmniTransformer.from_pretrained(
            checkpoint, dtype=torch.bfloat16, parallel_config=TensorParallelConfig(tp_degree=world_size)
        ).eval()
        specs = resolve_tp_shard_specs(reference, reference._get_tp_plan(), world_size)
        for name, parameter in native.named_parameters():
            expected = reference.get_parameter(name)
            if name in specs:
                assert isinstance(parameter, DTensor)
                assert parameter.dtype == torch.float8_e4m3fn
                expected = expected.chunk(world_size, dim=specs[name].dim)[rank].contiguous()
                torch.testing.assert_close(
                    parameter.to_local().view(torch.uint8), expected.view(torch.uint8), atol=0, rtol=0
                )
            else:
                torch.testing.assert_close(parameter, expected, atol=0, rtol=0)
        for name, buffer in native.named_buffers():
            assert not buffer.is_meta
            if isinstance(buffer, DTensor):
                assert all(placement.is_replicate() for placement in buffer.placements)
                buffer = buffer.to_local()
            if "quantizer" in name:
                assert buffer.device == device
            torch.testing.assert_close(buffer.to(device), reference.get_buffer(name), atol=0, rtol=0)

        policy = Cosmos3MixedPrecisionConfig.resolve(reference, mixed_precision_format="fp8")
        assert policy.enabled
        tester = Cosmos3FP8TensorParallelTesterConfig()
        inputs = tester.get_dummy_inputs()
        with torch.no_grad(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            for step in (0, 4, 9):
                for model in (reference, native):
                    apply_cosmos3_mixed_precision_step(model, policy, step, 10)
                expected = reference(**inputs).sample[0]
                actual = native(**inputs).sample[0]
                torch.testing.assert_close(actual, expected, atol=0.04, rtol=0.04)
        assert native.layers[0].self_attn.to_q._real_quant_gemm_impl is not None
    finally:
        dist.destroy_process_group()


@is_modelopt
@is_tensor_parallel
@require_torch_multi_accelerator
class TestCosmos3FP8TensorParallelLoading(Cosmos3FP8TensorParallelTesterConfig):
    @pytest.fixture
    def fp8_checkpoint(self, tmp_path):
        mtq = pytest.importorskip("modelopt.torch.quantization")
        from modelopt.torch.opt import enable_huggingface_checkpointing

        if torch_device != "cuda" or torch.cuda.get_device_capability()[0] < 9:
            pytest.skip("This test requires NCCL and FP8-capable CUDA devices.")
        enable_huggingface_checkpointing()
        torch.manual_seed(42)
        model = self.model_class(**self.get_init_dict()).to(device=torch_device, dtype=torch.bfloat16).eval()
        model.time_embedder.float()
        inputs = self.get_dummy_inputs()
        quant_config = {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "quantizer_name": "layers.*weight_quantizer",
                    "cfg": {"num_bits": (4, 3), "axis": None, "fake_quant": False},
                },
                {"quantizer_name": "layers.*input_quantizer", "cfg": {"num_bits": (4, 3), "axis": None}},
            ],
            "algorithm": "max",
        }
        with torch.no_grad():
            mtq.quantize(model, quant_config, forward_loop=lambda module: module(**inputs))
            mtq.compress(model)
        model.register_to_config(
            quantization_config=NVIDIAModelOptConfig(quant_type="FP8_FP8", modelopt_config=quant_config).to_dict()
        )
        checkpoint = tmp_path / "fp8-checkpoint"
        model.save_pretrained(checkpoint, max_shard_size="256KB")
        return checkpoint

    def test_sharded_loading_and_mixed_precision_inference(self, tmp_path, fp8_checkpoint):
        assert (fp8_checkpoint / SAFE_WEIGHTS_INDEX_NAME).is_file()
        mp.spawn(
            _cosmos_fp8_tp_worker,
            args=(2, (tmp_path / "rendezvous").as_uri(), str(fp8_checkpoint)),
            nprocs=2,
            join=True,
        )
