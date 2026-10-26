"""Distributed edge cases: torchrun --standalone --nproc_per_node=4 this_file.py."""

import copy
import json
import os
from contextlib import nullcontext

import torch
import torch.distributed as dist

from diffusers import Cosmos3OmniTransformer
from diffusers.models._modeling_parallel import ContextParallelConfig
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_cosmos3 import _cosmos3_ulysses_attention


rank = int(os.environ["RANK"])
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group("nccl")
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
degree = dist.get_world_size()
torch.backends.cuda.matmul.allow_tf32 = False


def check(a, b):
    torch.testing.assert_close(
        a, b, atol=0.025 if a.dtype == torch.bfloat16 else 3e-5, rtol=0.025 if a.dtype == torch.bfloat16 else 3e-5
    )


with torch.no_grad():
    for production in (False, True):
        dtype = torch.bfloat16 if production else torch.float32
        with nullcontext() if production else torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            for heads, kv in ((8, 4), (8, 1), (12, 3)):
                for act in ("silu", "relu2"):
                    torch.manual_seed(42)
                    model = (
                        Cosmos3OmniTransformer(
                            hidden_size=64,
                            intermediate_size=128,
                            head_dim=8,
                            num_attention_heads=heads,
                            num_key_value_heads=kv,
                            num_hidden_layers=2,
                            vocab_size=32,
                            hidden_act=act,
                            use_und_k_norm_for_gen=act == "relu2",
                        )
                        .to(device=device, dtype=dtype)
                        .eval()
                    )
                    ref = copy.deepcopy(model)
                    ref.set_attention_backend("native")
                    model.enable_parallelism(config=ContextParallelConfig(ulysses_degree=degree))
                    for u, g in ((5, 7), (8, 12), (1, 2), (0, 3), (3, 0), (0, 0)):
                        und, gen = (
                            torch.randn(u, 64, device=device, dtype=dtype),
                            torch.randn(g, 64, device=device, dtype=dtype),
                        )
                        au, ag = (
                            torch.randn(u, 4, device=device, dtype=dtype).repeat(1, 2),
                            torch.randn(g, 4, device=device, dtype=dtype).repeat(1, 2),
                        )
                        rope = (au.cos(), au.sin(), ag.cos(), ag.sin())
                        expected = ref._run_decoder_stack(und, gen, rope)
                        actual = model._run_decoder_stack(und, gen, rope)
                        for a, b in zip(actual, expected):
                            check(a, b)
                        if rank == 0:
                            print(
                                json.dumps(
                                    {
                                        "test": "decoder",
                                        "production": production,
                                        "heads": heads,
                                        "kv": kv,
                                        "act": act,
                                        "u": u,
                                        "g": g,
                                    }
                                ),
                                flush=True,
                            )
                    del model, ref
    # A global nontrivial mask detects incorrect [U_rank,G_rank] KV ordering.
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        for heads, kv in ((8, 1), (12, 3)):
            torch.manual_seed(8)
            u, g = 5, 7
            tensors = [
                torch.randn(n, h, 8, device=device)
                for n, h in ((u, heads), (u, kv), (u, kv), (g, heads), (g, kv), (g, kv), (u, kv))
            ]

            def shard(t):
                n = ((t.shape[0] + degree - 1) // degree) * degree
                return torch.nn.functional.pad(t, (0, 0, 0, 0, 0, n - t.shape[0])).chunk(degree)[rank].contiguous()

            mask = torch.ones(1, 1, g, u + g, device=device, dtype=torch.bool)
            mask[..., 1] = False
            mask[..., 4, 8:] = False
            for m in (mask, torch.zeros_like(mask, dtype=torch.float32).masked_fill(~mask, float("-inf"))):
                qu, ku, vu, qg, kg, vg, extra = tensors
                expected = dispatch_attention_fn(
                    qg[None],
                    torch.cat((extra, kg))[None],
                    torch.cat((vu, vg))[None],
                    attn_mask=m,
                    enable_gqa=True,
                    backend="native",
                )[0].flatten(-2)
                actual = _cosmos3_ulysses_attention(
                    dist.group.WORLD, *[shard(t) for t in tensors[:6]], (u, g), shard(extra), m
                )[1]
                padded = torch.nn.functional.pad(expected, (0, 0, 0, 1))
                check(actual, padded.chunk(degree)[rank])
                if rank == 0:
                    print(json.dumps({"test": "mask", "heads": heads, "kv": kv, "dtype": str(m.dtype)}), flush=True)
if rank == 0:
    print("ALL CP EDGE CHECKS PASSED", flush=True)
dist.destroy_process_group()
