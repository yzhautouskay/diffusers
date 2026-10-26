# Copyright 2026 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Cosmos native parallelism contracts; distributed checks are in cosmos3_parallel_worker.py."""

import pytest
import torch

from diffusers import Cosmos3OmniTransformer
from diffusers.hooks.tensor_parallel import resolve_tp_shard_specs
from diffusers.models._modeling_parallel import TensorParallelConfig
from diffusers.models.transformers.transformer_cosmos3 import Cosmos3PackedMoTAttention


@pytest.mark.parametrize("activation", ["silu", "relu2"])
def test_cosmos_tp_plan_covers_both_streams(activation):
    model = Cosmos3OmniTransformer(
        hidden_size=32,
        intermediate_size=64,
        head_dim=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        vocab_size=16,
        hidden_act=activation,
    )
    specs = resolve_tp_shard_specs(model, model._tp_plan, 2)
    for name, parameter in model.named_parameters():
        if name.startswith("layers.") and parameter.ndim == 2:
            assert name in specs
    assert "embed_tokens.weight" not in specs
    assert any("gate_proj" in key for key in specs) == (activation == "silu")


def test_cosmos_rejects_partial_kv_heads_before_distributed_setup():
    model = Cosmos3OmniTransformer(
        hidden_size=32,
        intermediate_size=64,
        head_dim=8,
        num_attention_heads=4,
        num_key_value_heads=1,
        num_hidden_layers=1,
        vocab_size=16,
    )
    with pytest.raises(ValueError, match="both query and KV"):
        model._resolve_parallel_config(TensorParallelConfig(tp_degree=2))
    assert model._parallel_config is None


def test_understanding_is_causal_and_generation_uses_understanding():
    torch.manual_seed(17)
    attn = Cosmos3PackedMoTAttention(32, 8, 4, 2, False, 1e-6).eval()
    und, gen = torch.randn(5, 32), torch.randn(7, 32)
    rope = (torch.ones(5, 8), torch.zeros(5, 8), torch.ones(7, 8), torch.zeros(7, 8))
    with torch.no_grad():
        und_out, gen_out = attn(und, gen, rope)
        changed = und.clone()
        changed[-1] += torch.randn(32) * 10
        new_und, new_gen = attn(changed, gen, rope)
    torch.testing.assert_close(und_out[:-1], new_und[:-1], atol=0, rtol=0)
    assert not torch.allclose(gen_out, new_gen)


def test_tp_loader_preserves_replicated_fp32_parameters(tmp_path):
    from types import SimpleNamespace

    from safetensors.torch import save_file

    from diffusers.models.model_loading_utils import _load_shard_file_tp

    with torch.device("meta"):
        model = torch.nn.ModuleDict({"time_embedder": torch.nn.Linear(4, 4, dtype=torch.bfloat16)})
    weights = {name: torch.randn(parameter.shape) for name, parameter in model.named_parameters()}
    checkpoint = tmp_path / "weights.safetensors"
    save_file(weights, checkpoint)
    config = SimpleNamespace(_mesh=SimpleNamespace(device_type="cuda"), _device=torch.device("cpu"))
    _load_shard_file_tp(
        str(checkpoint),
        model,
        model.state_dict(),
        {},
        config,
        dtype=torch.bfloat16,
        keep_in_fp32_modules=["time_embedder"],
        unexpected_keys=[],
    )
    for name, parameter in model.named_parameters():
        assert parameter.dtype == torch.float32
        torch.testing.assert_close(parameter, weights[name], atol=0, rtol=0)
