"""Run with torchrun --standalone --nproc_per_node=4 cosmos3_parallel_worker.py --tp 4."""

import argparse
import copy
import json
import os
import resource
import sys
from pathlib import Path

import torch
import torch.distributed as dist

from diffusers import Cosmos3OmniTransformer
from diffusers.models._modeling_parallel import TensorParallelConfig


parser = argparse.ArgumentParser()
parser.add_argument("--tp", type=int, default=1)
parser.add_argument("--cp", type=int, default=1)
args = parser.parse_args()
if args.cp != 1:
    parser.error("Native CP is experimental and excluded from this TP worker.")
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
rank = int(os.environ["RANK"])
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group("nccl")
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))


def parallel():
    return TensorParallelConfig(tp_degree=args.tp)

def inputs(u, g, dtype):
    idx = torch.arange(u, u + g, device=device)
    return {
        "input_ids": torch.arange(u, device=device) % 32,
        "text_indexes": torch.arange(u, device=device),
        "position_ids": torch.arange(u + g, device=device).expand(3, -1),
        "und_len": u,
        "sequence_length": u + g,
        "vision_tokens": [torch.randn(1, 2, 1, 1, g, device=device, dtype=dtype)],
        "vision_token_shapes": [(1, 1, g)],
        "vision_sequence_indexes": idx,
        "vision_mse_loss_indexes": idx,
        "vision_timesteps": torch.ones(g, device=device),
        "vision_noisy_frame_indexes": [torch.tensor([0], device=device)],
    }


def check(actual, expected, label, dtype):
    err = (actual.float() - expected.float()).abs()
    result = {
        "test": label,
        "tp": args.tp,
        "cp": args.cp,
        "dtype": str(dtype),
        "max_abs": err.max().item(),
        "rms": err.square().mean().sqrt().item(),
    }
    torch.testing.assert_close(
        actual,
        expected,
        atol=0.025 if dtype == torch.bfloat16 else 3e-5,
        rtol=0.025 if dtype == torch.bfloat16 else 3e-5,
    )
    if rank == 0:
        print(json.dumps(result), flush=True)


sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples/cosmos3"))
from cosmos_parallel import (  # noqa: E402
    enable_cosmos3_flash_attention,
    enable_cosmos3_tensor_parallel,
)


with torch.no_grad(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
    for act in ("silu", "relu2"):
        for dtype in (torch.float32, torch.bfloat16):
            torch.manual_seed(2026)
            model = Cosmos3OmniTransformer(
                head_dim=8,
                hidden_size=64,
                intermediate_size=128,
                latent_channel=2,
                latent_patch_size=1,
                num_attention_heads=8,
                num_key_value_heads=4,
                num_hidden_layers=2,
                patch_latent_dim=2,
                vocab_size=32,
                rope_axes_dim=[2, 1, 1],
                hidden_act=act,
                qk_norm_for_text=act == "silu",
                use_und_k_norm_for_gen=act == "relu2",
                attention_bias=True,
            ).eval()
            freqs = model.rotary_emb.inv_freq.clone()
            model.to(device=device, dtype=dtype)
            model.rotary_emb.inv_freq = freqs.to(device)
            ref = copy.deepcopy(model)
            # Keep the documented fp32 time embedding behavior even in BF16 tests.
            model.time_embedder.float()
            ref.time_embedder.float()
            model.set_attention_backend("native")
            ref.set_attention_backend("native")
            model.enable_parallelism(config=parallel())
            for u, g in ((5, 7), (8, 12), (7, 5)):
                data = inputs(u, g, dtype)
                check(model(**data).sample[0], ref(**data).sample[0], f"full_forward_{act}_u{u}_g{g}", dtype)
            # Compare decoder stack against the unmodified example, including its separate padding.
            if act == "silu":
                legacy = copy.deepcopy(ref)
                config = model._parallel_config
                if args.tp > 1:
                    enable_cosmos3_tensor_parallel(legacy, config.tensor_parallel_config._mesh)
                enable_cosmos3_flash_attention(legacy)
                # Example uses flash-attn; set backend native on installed processors for portable comparison.
                for proc in legacy.attn_processors.values():
                    proc._attention_backend = "native"
                data = inputs(5, 7, dtype)
                check(model(**data).sample[0], legacy(**data).sample[0], f"legacy_{act}", dtype)
                del legacy
            # Compare native rank-local checkpoint loading with sharding after loading.
            checkpoint = (
                Path(os.environ.get("VALIDATION_ROOT", "/tmp/cosmos3-parallel-tests"))
                / f"tiny-{args.tp}-{args.cp}-{act}-{str(dtype).split('.')[-1]}"
            )
            if rank == 0:
                ref.save_pretrained(checkpoint, max_shard_size="100KB")
            dist.barrier()
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            loaded = Cosmos3OmniTransformer.from_pretrained(
                checkpoint, torch_dtype=dtype, parallel_config=parallel()
            ).eval()
            torch.cuda.synchronize()
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "test": "loading",
                            "act": act,
                            "tp": args.tp,
                            "cp": args.cp,
                            "dtype": str(dtype),
                            "gpu_delta": torch.cuda.memory_allocated() - before,
                            "gpu_peak": torch.cuda.max_memory_allocated(),
                            "cpu_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        }
                    ),
                    flush=True,
                )
            # CP-only loader follows normal CPU placement.
            if args.tp == 1:
                loaded.to(device)
            loaded.set_attention_backend("native")
            for name, param in loaded.named_parameters():
                other = dict(model.named_parameters())[name]
                local = param.to_local() if hasattr(param, "to_local") else param
                other = other.to_local() if hasattr(other, "to_local") else other
                torch.testing.assert_close(local, other, atol=0, rtol=0, msg=lambda msg: name + ": " + msg)
            for name, buf in loaded.named_buffers():
                other = dict(model.named_buffers())[name]
                if not torch.equal(buf.to(other.device), other):
                    print("BUFFER DIFFERENCE", name, buf, other, flush=True)
            check(loaded(**data).sample[0], model(**data).sample[0], f"load_vs_enable_{act}", dtype)
            check(loaded(**data).sample[0], ref(**data).sample[0], f"checkpoint_{act}", dtype)
            del loaded, ref, model
            torch.cuda.empty_cache()
if rank == 0:
    print("ALL CHECKS PASSED", flush=True)
dist.destroy_process_group()
