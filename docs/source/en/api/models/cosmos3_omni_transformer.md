<!-- Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# Cosmos3OmniTransformer

A Mixture-of-Transformer (MoT) joint vision-language transformer introduced as part of NVIDIA's Cosmos3 world foundation model family. The model runs two parallel computation pathways over a packed joint sequence:

- a **causal "understanding" pathway** that self-attends over text tokens with causal masking, and
- a **bi-directional "generation" pathway** that cross-attends from generation tokens (vision + optional sound latents) over the full understanding-plus-generation key/value set.

The two pathways share the same hidden size and number of layers but maintain **separate Q/K/V/O projections, MLPs, and RMSNorm parameters**, which is what makes the architecture a Mixture-of-Transformer rather than a standard Mixture-of-Experts. Position information is supplied through a 3D multimodal RoPE (mRoPE) that interleaves temporal / height / width frequencies for video latents and reuses the temporal axis for text and audio.

The model can be loaded as follows.

```python
import torch
from diffusers import Cosmos3OmniTransformer

transformer = Cosmos3OmniTransformer.from_pretrained(
    "nvidia/Cosmos3-Nano", subfolder="transformer", dtype=torch.bfloat16
)
```

## Cosmos3OmniTransformer

[[autodoc]] Cosmos3OmniTransformer

## Native context parallelism

Initialize `torch.distributed`, select the local CUDA device, and pass a native configuration:

```python
from diffusers import ContextParallelConfig

transformer.enable_parallelism(config=ContextParallelConfig(ulysses_degree=4))
with torch.inference_mode():
    output = transformer(**inputs)
```

`from_pretrained(..., parallel_config=ContextParallelConfig(ulysses_degree=4))` is also supported; move the CP-only model to the local CUDA device after loading. CP replicates weights and partitions both token streams. The degree must divide the query head count. This inference implementation supports Ulysses with the `native` attention backend; ring attention and training are not supported.

The model installs a standalone `Cosmos3UlyssesAttnProcessor`. Its `_cp_plan` splits hidden states at the first layer, splits each layer's RoPE, and gathers final normalized outputs. Each stream is padded to equal rank lengths, including empty streams; padding is removed before attention and after the final gather. Causal understanding retains global token order. Generation attends to understanding followed by generation keys.

The processor packs each stream's Q/K/V into an existing Ulysses collective and reverses the output exchange. Compact GQA heads are retained wherever possible; KV groups are replicated only as needed for CP head ownership. No segmented input type or new shared attention dispatcher argument is introduced. The standard single-GPU/TP processor remains unaware of CP. SeaCache retains its existing request/CFG state and synchronizes compute decisions for native CP.

Understanding prefill is deliberately not part of this integration. Causal understanding can be reused when conditioning is fixed, but transparent reuse requires a request-scoped cache for every CFG branch and layer. Compact BF16 KV costs `2 * layers * understanding_tokens * KV_heads * head_dim * 2` bytes before TP partitioning. Long video conditioning can therefore retain gigabytes per branch. This implementation shards understanding computation alongside generation, adds no persistent KV cache, and recomputes changed conditioning on every forward.

## Combining tensor and context parallelism

```python
from diffusers import ParallelConfig, TensorParallelConfig

config = ParallelConfig(
    tensor_parallel_config=TensorParallelConfig(tp_degree=2),
    context_parallel_config=ContextParallelConfig(ulysses_degree=2),
)
transformer = Cosmos3OmniTransformer.from_pretrained(
    "nvidia/Cosmos3-Nano", subfolder="transformer", torch_dtype=torch.bfloat16,
    parallel_config=config,
)
```

Run this on four distributed ranks. The framework creates a `(ring, ulysses, tp)` mesh, loads TP-local weight slices, and attaches CP hooks after loading. CP exchanges only heads owned by the TP group. TP and CP degrees must multiply to the world size; custom combined meshes, ring attention, and unvalidated model classes are rejected. CLI `--parallelism auto` selects native execution for all supported degree combinations. Select `--parallelism example` explicitly for the previous implementation.
