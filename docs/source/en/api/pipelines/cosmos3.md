<!-- Copyright 2025 The HuggingFace Team. All rights reserved.
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
# limitations under the License. -->

# Cosmos 3

NVIDIA Cosmos 3 is a unified world foundation model (WFM) for Physical AI — a single omni-model that combines world generation, physical reasoning, and action generation. It replaces the separate Predict, Reason, and Transfer models from earlier Cosmos releases: whether you're building for robotics, autonomous vehicles, or smart spaces, Cosmos 3 gives you one foundation to simulate and understand the physical world.

What's shipping with this release:

- Models on the Hugging Face Hub with model cards and licensing
- Cosmos 3 Diffusers integration for generation pipelines (this page)
- Post-training scripts for fine-tuning Cosmos 3 on your own data
- Open synthetic data generation (SDG) datasets for Physical AI

## What's new in Cosmos 3

The biggest change from previous Cosmos releases is that Cosmos 3 is an *omni-model*, built on a Mixture-of-Transformers (MoT) architecture. Previously, developers worked with separate models for world generation (Predict), controlled generation (Transfer), scene understanding (Reason), and action-policy generation. Cosmos 3 unifies all of these in one model that reasons and generates across modalities in a single forward pass.

From one model you can:

- Generate physically plausible video worlds from text, images, or action inputs (image-to-video, text-to-video, action-conditioned video generation).
- Reason about physical properties like motion, causality, and spatial relationships.
- Predict future video and action sequences from the current state.
- Transfer scenes across viewpoints and conditions with structural control *(coming soon)*.

Under the hood, a single `Cosmos3OmniTransformer` runs a Qwen-style language model in parallel with a diffusion generation pathway: text tokens flow through a causal "understanding" stream while video and sound latents flow through a bi-directionally-attended "generation" stream, joined by a 3D multimodal RoPE. See the [Cosmos World Foundation Model Platform paper](https://huggingface.co/papers/2501.03575) for the architectural background.

## Available checkpoints

Two checkpoints are released on the Hub — [`nvidia/Cosmos3-Nano`](https://huggingface.co/nvidia/Cosmos3-Nano) (smaller, faster) and [`nvidia/Cosmos3-Super`](https://huggingface.co/nvidia/Cosmos3-Super) (larger, higher quality). The same pipeline class supports text-to-image, text-to-video, image-to-video, and (with a sound-capable checkpoint) text+image-to-video-with-sound — pick a repo and use the per-model tab in each workflow below.

> [!TIP]
> Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reusing-models-in-multiple-pipelines) section to learn how to efficiently load the same components into multiple pipelines.

## FP8 mixed W8A8/W8A16 denoising

Official ModelOpt FP8 checkpoints live on the Hub `fp8` revision (for example [`nvidia/Cosmos3-Nano`](https://huggingface.co/nvidia/Cosmos3-Nano) with `revision="fp8"`).

All of these checkpoints are quantized the same way. **W8A8** uses 8-bit weights and 8-bit activations (the restored ModelOpt GEMM). **W8A16** reuses those same 8-bit weights but skips activation quantization: the FP8 weight is dequantized and a standard linear runs on BF16/FP16/FP32 activations.

Running W8A8 on every step can produce visible flickering in multi-step video generation. The video Nano / Super / Super-I2V FP8 checkpoints therefore declare a schedule in `transformer/config.json`: **W8A16 on the first 3 and last 3 steps**, **W8A8 in the middle**. Diffusers reads those counts from the checkpoint rather than hardcoding them. Precision is chosen once per scheduler step so classifier-free guidance cond/uncond calls match.

Image generation and few-step distilled checkpoints do not show that flickering, so Super-T2I and the distilled 4-step FP8 repos declare no schedule and stay W8A8 on every step. The schedule is also **ModelOpt FP8 only**: other quantization backends (for example TorchAO) keep their native forwards.

Load the `fp8` revision with the same restore path as the [ModelOpt guide](../../quantization/modelopt) (`revision="fp8"` already carries the quantization config). Mixed precision then follows the checkpoint automatically:

```python
import torch
from diffusers import Cosmos3OmniPipeline

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano",
    revision="fp8",
    dtype=torch.bfloat16,
    device_map="cuda",
)
result = pipe(prompt="...", num_inference_steps=35)
```

Two generate-time choices:

- **Default** (`mixed_precision_format=None`): if the checkpoint declares `diffusion_step_policy`, run W8A16 on the first/last N steps and native W8A8 in the middle. That is the intended recipe for multi-step **video** FP8 (less flickering than all-W8A8). Distilled 4-step and Super-T2I FP8 omit the policy, so the default is already all W8A8.
- **`mixed_precision_format="none"`**: keep every step on native W8A8. Faster, because W8A16 is dequant + `torch.nn.functional.linear` rather than the restored FP8 GEMM, but multi-step video can flicker. Use this to A/B the schedule or to match a fully quantized baseline.

On one Blackwell workstation, Cosmos3-Nano `@fp8` at 720×1280 / 35 steps was about **27% slower** (T2I) and **13% slower** (49-frame T2V) with the default mixed schedule than with `"none"`. Those numbers are not a throughput guarantee. Pass `"fp8"` only to force the first/last-N schedule on a ModelOpt FP8 checkpoint that has no policy.

## Prompt upsampling

Cosmos 3 was trained on long, highly descriptive captions. For optimal quality, short text prompts should be **upsampled into a specific JSON structure** before they are passed to the pipeline. The upsampler lives in the [cosmos-framework](https://github.com/NVIDIA/cosmos-framework) package.

Start from a short, plain-text prompt and save it to `assets/prompt.txt`. For the text-to-video example below, the original prompt is *"A robotic arm is cleaning a plate in a kitchen"*:

```bash
mkdir -p assets
echo "A robotic arm is cleaning a plate in a kitchen" > assets/prompt.txt
```

Then install the framework and run the upsampler. The example below upsamples for text-to-video using Opus-4.6:

```bash
git clone https://github.com/NVIDIA/cosmos-framework.git packages/cosmos-framework
pip install -e packages/cosmos-framework

export PROMPT_UPSAMPLER_ENDPOINT_URL="https://api.anthropic.com/v1/"
export PROMPT_UPSAMPLER_MODEL_NAME="claude-opus-4-6"
export PROMPT_UPSAMPLER_API_TOKEN="<your_token>"

python -m cosmos_framework.inference.prompt_upsampling \
    --input assets/prompt.txt \
    --output assets/example_t2v_prompt.json \
    --mode text2video \
    --endpoint-url "${PROMPT_UPSAMPLER_ENDPOINT_URL}" \
    --model "${PROMPT_UPSAMPLER_MODEL_NAME}" \
    --api-token "${PROMPT_UPSAMPLER_API_TOKEN}" \
    --resolution 720 \
    --aspect-ratio "16,9"
```

Switch `--mode` to match the workflow you are targeting (`text2image`, `text2video`, `image2video`). The command writes the upsampled prompt(s) to the `--output` file as a JSON array (one object per non-empty line in `--input`); pass a `.jsonl` path instead to get one JSON object per line. For `image2video`, you must also supply the conditioning image via `--image-url` (a URL or local path) or `--image-list` (one image per prompt).

<!-- TODO: Add prompt upsampling support for video inputs (video-to-video) to the upsampler CLI. -->

A pre-upsampled positive prompt (`assets/example_t2v_prompt.json`) and negative prompt (`assets/negative_prompt.json`) are provided for convenience, and are used by the generation examples below. The examples load these JSON files and pass them to the pipeline as JSON strings via `json.dumps(...)`.

## Text-to-video

Multi-frame generation conditioned on text alone. Pick `num_frames` based on the target duration — the default `num_frames=189` produces ≈ 7.9 s at 24 FPS. The prompt and negative prompt are read from the JSON-upsampled files described in [Prompt upsampling](#prompt-upsampling).

<hfoptions id="model">
<hfoption id="Nano">

```python
import json
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video

# JSON-upsampled positive and negative prompts (see "Prompt upsampling" above).
json_prompt = json.load(open("assets/example_t2v_prompt.json"))
negative_prompt = json.load(open("assets/negative_prompt.json"))

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano", dtype=torch.bfloat16, device_map="cuda"  # or "mps", "xpu", "cpu"
)
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=10.0, use_karras_sigmas=False
)

result = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    num_frames=189,
    height=720,
    width=1280,
    num_inference_steps=35,
    guidance_scale=6.0,
    fps=24.0,
)
# macro_block_size=1 allows arbitrary frame sizes (Cosmos3 outputs are not always divisible by 16).
export_to_video(result.video, "cosmos3_t2v.mp4", fps=24, macro_block_size=1)
```

</hfoption>
<hfoption id="Super">

```python
import json
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video

# JSON-upsampled positive and negative prompts (see "Prompt upsampling" above).
json_prompt = json.load(open("assets/example_t2v_prompt.json"))
negative_prompt = json.load(open("assets/negative_prompt.json"))

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Super", dtype=torch.bfloat16, device_map="cuda"  # or "mps", "xpu", "cpu"
)
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=10.0, use_karras_sigmas=False
)

result = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    num_frames=189,
    height=720,
    width=1280,
    num_inference_steps=35,
    guidance_scale=6.0,
    fps=24.0,
)
# macro_block_size=1 allows arbitrary frame sizes (Cosmos3 outputs are not always divisible by 16).
export_to_video(result.video, "cosmos3_t2v.mp4", fps=24, macro_block_size=1)
```

</hfoption>
</hfoptions>

## Text-to-image

Single-frame generation. The model is conditioned only on the text prompt; pass `num_frames=1`. Upsample with `--mode text2image` to produce the JSON prompt.

<hfoptions id="model">
<hfoption id="Nano">

```python
import json
import torch
from diffusers import Cosmos3OmniPipeline

# JSON-upsampled prompt (see "Prompt upsampling" above).
json_prompt = json.load(open("assets/example_t2i_prompt.json"))

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano", dtype=torch.bfloat16, device_map="cuda"  # or "mps", "xpu", "cpu"
)

result = pipe(prompt=json.dumps(json_prompt), num_frames=1, height=720, width=1280)
result.video[0].save("cosmos3_t2i.jpg", format="JPEG", quality=85)
```

</hfoption>
<hfoption id="Super">

```python
import json
import torch
from diffusers import Cosmos3OmniPipeline

# JSON-upsampled prompt (see "Prompt upsampling" above).
json_prompt = json.load(open("assets/example_t2i_prompt.json"))

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Super", dtype=torch.bfloat16, device_map="cuda"  # or "mps", "xpu", "cpu"
)

result = pipe(prompt=json.dumps(json_prompt), num_frames=1, height=720, width=1280)
result.video[0].save("cosmos3_t2i.jpg", format="JPEG", quality=85)
```

</hfoption>
</hfoptions>

## Image-to-video

Pass a conditioning image via `image=`. The pipeline anchors frame 0 to the supplied image and denoises the rest. The image is resized while preserving its aspect ratio, center-cropped to the requested output size, and normalized with uint8-equivalent rounding. Upsample with `--mode image2video` to produce the JSON prompt.

<hfoptions id="model">
<hfoption id="Nano">

```python
import json
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.utils import export_to_video, load_image

# JSON-upsampled positive and negative prompts (see "Prompt upsampling" above).
json_prompt = json.load(open("assets/example_i2v_prompt.json"))
negative_prompt = json.load(open("assets/negative_prompt_i2v.json"))

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano", dtype=torch.bfloat16, device_map="cuda"  # or "mps", "xpu", "cpu"
)

image = load_image(
    "https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/assets/robot_153.jpg"
)

result = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    image=image,
    num_frames=189,
    height=720,
    width=1280,
    fps=24.0,
)
# macro_block_size=1 allows arbitrary frame sizes (Cosmos3 outputs are not always divisible by 16).
export_to_video(result.video, "cosmos3_i2v.mp4", fps=24, macro_block_size=1)
```

</hfoption>
<hfoption id="Super">

```python
import json
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.utils import export_to_video, load_image

# JSON-upsampled positive and negative prompts (see "Prompt upsampling" above).
json_prompt = json.load(open("assets/example_i2v_prompt.json"))
negative_prompt = json.load(open("assets/negative_prompt_i2v.json"))

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Super", dtype=torch.bfloat16, device_map="cuda"  # or "mps", "xpu", "cpu"
)

image = load_image(
    "https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/assets/robot_153.jpg"
)

result = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    image=image,
    num_frames=189,
    height=720,
    width=1280,
    fps=24.0,
)
# macro_block_size=1 allows arbitrary frame sizes (Cosmos3 outputs are not always divisible by 16).
export_to_video(result.video, "cosmos3_i2v.mp4", fps=24, macro_block_size=1)
```

</hfoption>
</hfoptions>

## Video-to-video

Pass a conditioning clip via `video=` (e.g. from `load_video`). The pipeline anchors the leading latent frames given by `condition_frame_indexes_vision` (default `[0, 1]`) to the clip and denoises the rest. Use `condition_video_keep` (`"first"` or `"last"`) to choose which end of a longer source clip the conditioning frames are taken from. As with the other modes, the prompt should follow the descriptive JSON structure described in [Prompt upsampling](#prompt-upsampling).

<!-- TODO: Add prompt upsampling support for video inputs (video-to-video) to the upsampler CLI. -->

<hfoptions id="model">
<hfoption id="Nano">

```python
import json
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_video

# JSON-upsampled positive and negative prompts (see "Prompt upsampling" above).
json_prompt = json.load(open("assets/example_v2v_prompt.json"))
negative_prompt = json.load(open("assets/negative_prompt_i2v.json"))

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano", dtype=torch.bfloat16, device_map="cuda"  # or "mps", "xpu", "cpu"
)
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=10.0, use_karras_sigmas=False
)

video = load_video(
    "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/vision/robot_pouring.mp4"
)

result = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    video=video,
    condition_frame_indexes_vision=[0, 1],
    condition_video_keep="first",
    num_frames=189,
    height=720,
    width=1280,
    num_inference_steps=35,
    guidance_scale=6.0,
    fps=24.0,
)
# macro_block_size=1 allows arbitrary frame sizes (Cosmos3 outputs are not always divisible by 16).
export_to_video(result.video, "cosmos3_v2v.mp4", fps=24, macro_block_size=1)
```

</hfoption>
<hfoption id="Super">

```python
import json
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_video

# JSON-upsampled positive and negative prompts (see "Prompt upsampling" above).
json_prompt = json.load(open("assets/example_v2v_prompt.json"))
negative_prompt = json.load(open("assets/negative_prompt_i2v.json"))

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Super", dtype=torch.bfloat16, device_map="cuda"  # or "mps", "xpu", "cpu"
)
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=10.0, use_karras_sigmas=False
)

video = load_video(
    "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/vision/robot_pouring.mp4"
)

result = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    video=video,
    condition_frame_indexes_vision=[0, 1],
    condition_video_keep="first",
    num_frames=189,
    height=720,
    width=1280,
    num_inference_steps=35,
    guidance_scale=6.0,
    fps=24.0,
)
# macro_block_size=1 allows arbitrary frame sizes (Cosmos3 outputs are not always divisible by 16).
export_to_video(result.video, "cosmos3_v2v.mp4", fps=24, macro_block_size=1)
```

</hfoption>
</hfoptions>

## Video-to-video with sound

When the checkpoint carries a `sound_tokenizer`, add `enable_sound=True` to the video-to-video call to jointly generate a synchronized audio track. The waveform is returned alongside the video and can be muxed into the MP4 with [`~utils.encode_video`].

<hfoptions id="model">
<hfoption id="Nano">

```python
import json
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import encode_video, load_video

# JSON-upsampled positive and negative prompts (see "Prompt upsampling" above).
json_prompt = json.load(open("assets/example_v2v_prompt.json"))
negative_prompt = json.load(open("assets/negative_prompt_i2v.json"))

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano", dtype=torch.bfloat16, device_map="cuda"  # or "mps", "xpu", "cpu"
)
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=10.0, use_karras_sigmas=False
)

video = load_video(
    "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/vision/robot_pouring.mp4"
)

result = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    video=video,
    condition_frame_indexes_vision=[0, 1],
    condition_video_keep="first",
    num_frames=189,
    height=720,
    width=1280,
    fps=24.0,
    enable_sound=True,
)

encode_video(
    result.video,
    fps=24,
    audio=result.sound,
    audio_sample_rate=pipe.sound_tokenizer.config.sampling_rate,
    output_path="cosmos3_v2v_with_sound.mp4",
)
```

</hfoption>
<hfoption id="Super">

```python
import json
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import encode_video, load_video

# JSON-upsampled positive and negative prompts (see "Prompt upsampling" above).
json_prompt = json.load(open("assets/example_v2v_prompt.json"))
negative_prompt = json.load(open("assets/negative_prompt_i2v.json"))

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Super", dtype=torch.bfloat16, device_map="cuda"  # or "mps", "xpu", "cpu"
)
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=10.0, use_karras_sigmas=False
)

video = load_video(
    "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/vision/robot_pouring.mp4"
)

result = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    video=video,
    condition_frame_indexes_vision=[0, 1],
    condition_video_keep="first",
    num_frames=189,
    height=720,
    width=1280,
    fps=24.0,
    enable_sound=True,
)

encode_video(
    result.video,
    fps=24,
    audio=result.sound,
    audio_sample_rate=pipe.sound_tokenizer.config.sampling_rate,
    output_path="cosmos3_v2v_with_sound.mp4",
)
```

</hfoption>
</hfoptions>

## Text-to-video with sound

When the checkpoint carries a `sound_tokenizer`, pass `enable_sound=True` to jointly generate a synchronized audio track. The waveform is returned alongside the video and can be muxed into the MP4 with [`~utils.encode_video`].

This is the same call as the text-to-video example above with `enable_sound=True` added:

<hfoptions id="model">
<hfoption id="Nano">

```python
import json
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.utils import encode_video

# JSON-upsampled positive and negative prompts (see "Prompt upsampling" above).
json_prompt = json.load(open("assets/example_t2v_sound_prompt.json"))
negative_prompt = json.load(open("assets/negative_prompt.json"))

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano", dtype=torch.bfloat16, device_map="cuda"  # or "mps", "xpu", "cpu"
)

result = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    num_frames=189,
    height=720,
    width=1280,
    fps=24.0,
    enable_sound=True,
)

encode_video(
    result.video,
    fps=24,
    audio=result.sound,
    audio_sample_rate=pipe.sound_tokenizer.config.sampling_rate,
    output_path="cosmos3_with_sound.mp4",
)
```

</hfoption>
<hfoption id="Super">

```python
import json
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.utils import encode_video

# JSON-upsampled positive and negative prompts (see "Prompt upsampling" above).
json_prompt = json.load(open("assets/example_t2v_sound_prompt.json"))
negative_prompt = json.load(open("assets/negative_prompt.json"))

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Super", dtype=torch.bfloat16, device_map="cuda"  # or "mps", "xpu", "cpu"
)

result = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    num_frames=189,
    height=720,
    width=1280,
    fps=24.0,
    enable_sound=True,
)

encode_video(
    result.video,
    fps=24,
    audio=result.sound,
    audio_sample_rate=pipe.sound_tokenizer.config.sampling_rate,
    output_path="cosmos3_with_sound.mp4",
)
```

</hfoption>
</hfoptions>

## Action-conditioned generation

Action runs group every action-specific input into a [`CosmosActionCondition`] passed via the `action` argument instead of the top-level `image` / `video` / `height` / `width` arguments. Set `resolution_tier` (`256`/`480`/`704`/`720`) close to the input video's native resolution; it selects the conditioning canvas. Cosmos 3 supports three action modes — `policy`, `forward_dynamics`, and `inverse_dynamics`. `policy` and `forward_dynamics` condition only on the first frame (so an `image` or a `video` both work), while `inverse_dynamics` requires a `video`. The conditioning video for an action run is set on `action.video` (or `action.image`), not on the pipeline's top-level `video` argument.

Pass a plain task description as `prompt` and pick the camera with `action.view_point` (default `"ego_view"`; also `"third_person_view"`, `"wrist_view"`, `"concat_view"`). The pipeline turns these into the structured JSON caption the model was trained on, so action prompts should not be LLM-upsampled.

### Action policy

Action policy generation predicts future video and action tokens from the first observation frame, text prompt, and action domain metadata. The example below uses the Bridge robot domain and writes the predicted action chunk to JSON in model-normalized action space.

<hfoptions id="model">
<hfoption id="Nano">

```python
import json

import torch
from diffusers import Cosmos3OmniPipeline, CosmosActionCondition
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_video

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano", dtype=torch.bfloat16, device_map="cuda"  # or "mps", "xpu", "cpu"
)
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=10.0, use_karras_sigmas=False
)

prompt = "Put the pot to the left of the purple item."
video = load_video(
    "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/action/bridge_20260501_0.mp4"
)

result = pipe(
    prompt=prompt,
    action=CosmosActionCondition(
        mode="policy",
        chunk_size=16,
        domain_name="bridge_orig_lerobot",
        resolution_tier=480,
        video=video,
        view_point="ego_view",
    ),
    fps=5,
    num_inference_steps=30,
    guidance_scale=1.0,
    use_system_prompt=False,
)

# macro_block_size=1 allows arbitrary frame sizes (Cosmos3 outputs are not always divisible by 16).
export_to_video(result.video, "sample.mp4", fps=5, macro_block_size=1)

if result.action is not None:
    with open("sample_action.json", "w") as f:
        json.dump(result.action[0].tolist(), f)
```

</hfoption>
<hfoption id="Super">

```python
import json

import torch
from diffusers import Cosmos3OmniPipeline, CosmosActionCondition
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_video

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Super", dtype=torch.bfloat16, device_map="cuda"  # or "mps", "xpu", "cpu"
)
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=10.0, use_karras_sigmas=False
)

prompt = "Put the pot to the left of the purple item."
video = load_video(
    "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/action/bridge_20260501_0.mp4"
)

result = pipe(
    prompt=prompt,
    action=CosmosActionCondition(
        mode="policy",
        chunk_size=16,
        domain_name="bridge_orig_lerobot",
        resolution_tier=480,
        video=video,
        view_point="ego_view",
    ),
    fps=5,
    num_inference_steps=30,
    guidance_scale=1.0,
    use_system_prompt=False,
)

# macro_block_size=1 allows arbitrary frame sizes (Cosmos3 outputs are not always divisible by 16).
export_to_video(result.video, "sample.mp4", fps=5, macro_block_size=1)

if result.action is not None:
    with open("sample_action.json", "w") as f:
        json.dump(result.action[0].tolist(), f)
```

</hfoption>
</hfoptions>

## SeaCache

SeaCache is disabled by default. Cosmos 3 supports enabling it explicitly with [`SeaCacheConfig`]. SeaCache reuses
transformer residuals when the Spectral-Evolution-Aware indicator changes slowly, reducing the number of full
transformer executions. It computes the indicator from the raw vision latents, including clean conditioning frames for
image-to-video generation. Enable it on the transformer with scheduler metadata callbacks from the pipeline:

```python
import torch
from diffusers import Cosmos3OmniPipeline, SeaCacheConfig

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano", dtype=torch.bfloat16, device_map="cuda"
)

pipe.transformer.enable_cache(SeaCacheConfig(threshold=0.2, max_consecutive_cached=2))
```

The same model-level API works with [`Cosmos3OmniPipeline`], [`Cosmos3OmniModularPipeline`], and
[`Cosmos3DistilledModularPipeline`]. SeaCache is approximate and can change generated outputs. Disable it with
`pipe.transformer.disable_cache()` when you need every denoising step to execute the full transformer. Cache state is
reset after each pipeline call, and conditional and unconditional guidance branches keep independent histories.

Cosmos 3 keeps the SeaCache gate outside its repeated decoder layers, so it is compatible with regional compilation.
Compile the layers after enabling the cache:

```python
pipe.transformer.compile_repeated_blocks(fullgraph=True)
```

SeaCache also supports the Cosmos 3 Ulysses context-parallel and DTensor-based tensor-parallel helpers documented
below. Cache decisions are synchronized across ranks. Full-model compilation and other model-sharding strategies are not claimed.

## Tensor parallelism

Cosmos 3 supports native tensor parallelism (TP) through `TensorParallelConfig` and its model `_tp_plan`. Both streams' Q/K/V and MLP input projections use column sharding; attention output and MLP down projections use row sharding and reduction. Embeddings, final output heads, modality projections, and normalization weights remain replicated. Head reshaping uses the projection output's local width without changing checkpoint configuration.

The TP degree must divide both query and KV head counts and the sharded MLP dimensions. Released Nano and Super have 8 KV heads. SiLU and ReLU2 MLP variants have separate plans because ReLU2 has no gate projection.

### Run it

The CLI uses `--parallelism auto` by default: TP, CP, and combined TP+CP all select native execution. Launch one process per GPU with `--tp-degree` equal to the world size. All ranks must receive identical prompts, conditioning inputs, and seeds.

```bash
torchrun --nproc_per_node=4 examples/cosmos3/inference_cosmos3.py \
    --model super --tp-degree 4 --prompt "A robot reading a book."
```

`--model-path` selects an existing local checkpoint snapshot. Loading uses native rank-local safetensors slices rather than materializing the full transformer on each GPU. `enable_parallelism(...)` can also shard an already loaded model but incurs the original loading memory cost.

### Use it in your own modular pipeline

```python
import os

import torch
import torch.distributed as dist
from diffusers import Cosmos3OmniModularPipeline, Cosmos3OmniTransformer
from diffusers.models._modeling_parallel import TensorParallelConfig

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl")

model_id = "nvidia/Cosmos3-Nano"
transformer = Cosmos3OmniTransformer.from_pretrained(
    model_id,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    parallel_config=TensorParallelConfig(tp_degree=dist.get_world_size()),
)
pipe = Cosmos3OmniModularPipeline.from_pretrained(model_id)
pipe.update_components(transformer=transformer)
pipe.load_components(torch_dtype=torch.bfloat16)
pipe.to(f"cuda:{local_rank}")
pipe.transformer.set_attention_backend("native")
frames = pipe(
    prompt="A robot reading a book.",
    num_frames=1,
    height=512,
    width=512,
    generator=torch.Generator().manual_seed(42),
    output="videos",
)
if dist.get_rank() == 0:
    frames[0].save("sample.png")
dist.destroy_process_group()
```

Released Super's transformer contains approximately 64.02B BF16 parameters: 128.04 GB (119.25 GiB) of stored tensor data. Other components and runtime memory are additional. This exceeds a 96 GB GPU's capacity, but single-GPU execution can fit on a larger GPU such as GB200, depending on workload. These figures describe released Cosmos 3, not upcoming Cosmos 3.5 MoE models.

Native TP retains the framework's restrictions on quantization, CPU offloading, LoRA, and saving sharded models. Multi-node execution and HSDP/FSDP composition require separate validation.

## Context parallelism

Native context parallelism (CP) uses `ContextParallelConfig(ulysses_degree=4)` and installs a standalone Cosmos processor with the existing Ulysses collectives. It shards both streams and RoPE through `_cp_plan`, handles short/empty streams by bounded padding, and preserves compact GQA where possible. See [the model documentation](../models/cosmos3_omni_transformer#native-context-parallelism).

The previous implementation remains available through the standalone processors and helpers in [`examples/cosmos3/cosmos_parallel.py`](https://github.com/huggingface/diffusers/blob/main/examples/cosmos3/cosmos_parallel.py). Existing command-line degrees continue to work with the native path. `--parallelism example` selects them explicitly:

```bash
# Sequence sharding; every rank holds the full model's weights.
torchrun --nproc_per_node=4 examples/cosmos3/inference_cosmos3.py \
    --model nano --parallelism example --cp-degree 4 --prompt "A robot reading a book."

# Existing example's orthogonal weight and sequence sharding.
torchrun --nproc_per_node=4 examples/cosmos3/inference_cosmos3.py \
    --model super --parallelism example --tp-degree 2 --cp-degree 2 --prompt "A robot reading a book."
```

The example shards understanding and generation streams and their RoPE separately, pads ragged streams, and handles causal understanding, generation cross-attention, and GQA in its own processor. Its TP degree must divide the KV head count, and `tp_degree * cp_degree` must divide the query head count. Use the `native` attention backend. The helper expands KV heads before calling SDPA so flash kernels can be selected. CP alone does not reduce the weight footprint. Do not apply example helpers to a model already using native TP.

Native TP+CP constructs orthogonal CP and TP submeshes and installs CP hooks after rank-local checkpoint loading. The degree product must equal the world size; TP must divide query and KV heads, and CP must divide TP-local query heads. Custom meshes and composition for models without explicit support are rejected. Native CP does not require a segmented-KV dispatcher API: the standalone processor owns causal understanding attention, GQA mapping, and key ordering while reusing existing tensor collectives. The standard attention API is unchanged.

## Metadata templates

`tokenize_prompt` appends short metadata sentences inside the user message so the LLM sees the conditioning the model was trained with. The positive prompt gets sentences like *"The video is 7.9 seconds long and is of 24 FPS."* and *"This video is of 720x1280 resolution."*; the negative prompt gets the inverse (*"… is not …"*).

Both are on by default. Disable either pair through `__call__`:

```python
result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_frames=189,
    height=720,
    width=1280,
    fps=24.0,
    add_duration_template=False,    # skip the duration sentence on both prompts
    add_resolution_template=False,  # skip the resolution sentence on both prompts
)
```

`add_duration_template` has no effect when `num_frames == 1` (image mode); only the resolution sentence is appended in that case.

## Safety checker

Cosmos3 wires up the [`cosmos_guardrail`](https://pypi.org/project/cosmos-guardrail/) `CosmosSafetyChecker` and runs it **by default**. The text guardrail rejects unsafe prompts before generation (`ValueError`); the video guardrail runs on the decoded frames and either pixelates detected faces or rejects the output. Audio output is not guardrailed.

Install the optional dependency to enable the default checker:

```
pip install cosmos_guardrail
```

The checker is mandatory under the NVIDIA Open Model License Agreement. The two flags below exist for tests and development workflows where the guardrail would be redundant (e.g., the input has already been cleared, or you are intentionally exercising the pipeline on edge inputs).

**Disable at construction** (no checker is instantiated, so no guardrail models are downloaded or loaded into memory):

```python
import torch
from diffusers import Cosmos3OmniPipeline

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano",
    dtype=torch.bfloat16,
    device_map="cuda",  # or "mps", "xpu", "cpu"
    enable_safety_checker=False,
)
```

**Disable for a single call** (checker stays loaded — useful for one-off bypass while keeping the default on for subsequent calls):

```python
result = pipe(
    prompt=prompt,
    num_frames=189,
    height=720,
    width=1280,
    fps=24.0,
    enable_safety_check=False,
)
```

To supply a custom checker (e.g., a no-op subclass for fast tests), pass it as `safety_checker=`:

```python
pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano",
    dtype=torch.bfloat16,
    device_map="cuda",  # or "mps", "xpu", "cpu"
    safety_checker=MyCustomSafetyChecker(),
)
```

## Cosmos3OmniPipeline

[[autodoc]] Cosmos3OmniPipeline

- all
- __call__

## Cosmos3OmniModularPipeline

Cosmos 3 is also available as a Modular Diffusers pipeline. The task-based [`Cosmos3OmniPipeline`] remains available; the modular pipeline coexists with it and covers the same modes (`text2image`, `text2video`, `image2video`, `video2video`, action-conditioned generation, and `transfer` (structural control), with optional sound when supported by the checkpoint).

```python
import torch
from diffusers import Cosmos3OmniModularPipeline

pipe = Cosmos3OmniModularPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano", dtype=torch.bfloat16
)
pipe.load_components(dtype=torch.bfloat16)
pipe.enable_safety_checker()

videos = pipe(
    prompt='{"scene":"A robot arm in a kitchen"}',
    num_frames=1,
    height=720,
    width=1280,
    output="videos",
)

# Modular pipelines expose declared outputs directly instead of using the task pipeline's
# `return_dict`/`Cosmos3OmniPipelineOutput` API.
image = videos[0]
```

You can also load through [`ModularPipeline`] and let the repository config select the blocks class:

```python
import torch
from diffusers import ModularPipeline

pipe = ModularPipeline.from_pretrained("nvidia/Cosmos3-Nano", dtype=torch.bfloat16)
pipe.load_components(dtype=torch.bfloat16)
pipe.enable_safety_checker()
videos = pipe(
    prompt='{"scene":"A robot arm in a kitchen"}', num_frames=1, height=720, width=1280, output="videos"
)
```

To inspect or customize a specific Cosmos modular workflow, use `available_workflows` + `get_workflow()`:

```python
available = pipe.blocks.available_workflows
image2video_blocks = pipe.blocks.get_workflow("image2video")
```

### Modular examples for all existing workflows

The modular pipeline supports the same call signatures as the task pipeline. The snippets below mirror every generation example shown above (`text2video`, `text2image`, `image2video`, `video2video`, `video2video_sound`, `text2video_sound`, and `action_policy`). Transfer (structural control) has its own inputs and is shown separately in [Modular transfer](#modular-transfer-structural-control) below.

```python
import json
import torch
from diffusers import Cosmos3OmniModularPipeline, CosmosActionCondition
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import encode_video, export_to_video, load_image, load_video

pipe = Cosmos3OmniModularPipeline.from_pretrained("nvidia/Cosmos3-Nano", dtype=torch.bfloat16)
pipe.load_components(dtype=torch.bfloat16)
pipe.enable_safety_checker()
pipe.to("cuda")  # or "mps", "xpu", "cpu"
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=10.0, use_karras_sigmas=False
)

# text2video
json_prompt = json.load(open("assets/example_t2v_prompt.json"))
negative_prompt = json.load(open("assets/negative_prompt.json"))
videos = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    num_frames=189,
    height=720,
    width=1280,
    num_inference_steps=35,
    guidance_scale=6.0,
    fps=24.0,
    output="videos",
)
export_to_video(videos, "cosmos3_modular_t2v.mp4", fps=24, macro_block_size=1)

# text2image
json_prompt = json.load(open("assets/example_t2i_prompt.json"))
videos = pipe(prompt=json.dumps(json_prompt), num_frames=1, height=720, width=1280, output="videos")
videos[0].save("cosmos3_modular_t2i.jpg", format="JPEG", quality=85)

# image2video
json_prompt = json.load(open("assets/example_i2v_prompt.json"))
negative_prompt = json.load(open("assets/negative_prompt_i2v.json"))
image = load_image("https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/assets/robot_153.jpg")
videos = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    image=image,
    num_frames=189,
    height=720,
    width=1280,
    fps=24.0,
    output="videos",
)
export_to_video(videos, "cosmos3_modular_i2v.mp4", fps=24, macro_block_size=1)

# video2video
json_prompt = json.load(open("assets/example_v2v_prompt.json"))
negative_prompt = json.load(open("assets/negative_prompt_i2v.json"))
video = load_video(
    "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/vision/robot_pouring.mp4"
)
videos = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    video=video,
    condition_frame_indexes_vision=[0, 1],
    condition_video_keep="first",
    num_frames=189,
    height=720,
    width=1280,
    num_inference_steps=35,
    guidance_scale=6.0,
    fps=24.0,
    output="videos",
)
export_to_video(videos, "cosmos3_modular_v2v.mp4", fps=24, macro_block_size=1)

# video2video_sound
outputs = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    video=video,
    condition_frame_indexes_vision=[0, 1],
    condition_video_keep="first",
    num_frames=189,
    height=720,
    width=1280,
    fps=24.0,
    enable_sound=True,
    output=["videos", "sound", "sampling_rate"],
)
encode_video(
    outputs["videos"],
    fps=24,
    audio=outputs["sound"],
    audio_sample_rate=outputs["sampling_rate"],
    output_path="cosmos3_modular_v2v_with_sound.mp4",
)

# text2video_sound
json_prompt = json.load(open("assets/example_t2v_sound_prompt.json"))
negative_prompt = json.load(open("assets/negative_prompt.json"))
outputs = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    num_frames=189,
    height=720,
    width=1280,
    fps=24.0,
    enable_sound=True,
    output=["videos", "sound", "sampling_rate"],
)
encode_video(
    outputs["videos"],
    fps=24,
    audio=outputs["sound"],
    audio_sample_rate=outputs["sampling_rate"],
    output_path="cosmos3_modular_t2v_with_sound.mp4",
)

# action_policy
prompt = "Put the pot to the left of the purple item."
action_video = load_video(
    "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/action/bridge_20260501_0.mp4"
)
outputs = pipe(
    prompt=prompt,
    action=CosmosActionCondition(
        mode="policy",
        chunk_size=16,
        domain_name="bridge_orig_lerobot",
        resolution_tier=480,
        video=action_video,
        view_point="ego_view",
    ),
    fps=5,
    num_inference_steps=30,
    guidance_scale=1.0,
    use_system_prompt=False,
    output=["videos", "action"],
)
export_to_video(outputs["videos"], "cosmos3_modular_action_policy.mp4", fps=5, macro_block_size=1)
if outputs["action"] is not None:
    with open("cosmos3_modular_action_policy.json", "w") as f:
        json.dump(outputs["action"][0].tolist(), f)
```

### Modular transfer (structural control)

Transfer follows a **precomputed control video** (edge, blur, depth, segmentation, or a world-scenario map) passed through `control_videos=` as a `{hint: video}` mapping. It is video-only (no `image` / `video` / `action` / `enable_sound`), the prompt is a pre-upsampled JSON caption (see [Prompt upsampling](#prompt-upsampling)), and long clips are generated autoregressively in chunks of `num_video_frames_per_chunk` and stitched automatically. `guidance_scale` is the usual text CFG; `control_guidance` (`!= 1.0`) additionally amplifies the control signal. Recommended starting values per hint:

| Hint | `guidance_scale` | `control_guidance` | `flow_shift` | Geometry |
| --- | --- | --- | --- | --- |
| Edge / Blur / Depth | 3.0 | 1.5 | 10.0 | 121 frames @ 30 FPS |
| Segmentation | 3.0 | 2.0 | 10.0 | 121 frames @ 30 FPS |
| World scenario (WSM) | 1.0 | 3.0 | 10.0 | 101 frames @ 10 FPS |

Diffusers does not ship the control assets. Ready-made ones (a control video + matching `prompt.json` per hint, plus a shared `negative_prompt.json`) live in the [Cosmos cookbook](https://github.com/NVIDIA/cosmos/tree/main/cookbooks/cosmos3/generator/transfer/assets). For the edge example below, download them into a local `assets/` folder:

```bash
base=https://github.com/NVIDIA/cosmos/raw/refs/heads/main/cookbooks/cosmos3/generator/transfer/assets
mkdir -p assets/edge
curl -sL "$base/edge/control_edge.mp4" -o assets/edge/control_edge.mp4
curl -sL "$base/edge/prompt.json"      -o assets/edge/prompt.json
curl -sL "$base/negative_prompt.json"  -o assets/negative_prompt.json
```

```python
import json
import torch
from diffusers import Cosmos3OmniModularPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_video

pipe = Cosmos3OmniModularPipeline.from_pretrained("nvidia/Cosmos3-Nano", dtype=torch.bfloat16)
pipe.load_components(dtype=torch.bfloat16)
pipe.to("cuda")  # or "mps", "xpu", "cpu"
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=10.0, use_karras_sigmas=False
)

# Downloaded into assets/ from the Cosmos cookbook (see the curl snippet above).
json_prompt = json.load(open("assets/edge/prompt.json"))
negative_prompt = json.load(open("assets/negative_prompt.json"))
control_edge = load_video("assets/edge/control_edge.mp4")

videos = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    control_videos={"edge": control_edge},
    num_frames=121,
    height=720,
    width=1280,
    fps=30.0,
    num_inference_steps=35,
    guidance_scale=3.0,
    control_guidance=1.5,
    output="videos",
)
export_to_video(videos, "cosmos3_modular_transfer_edge.mp4", fps=30, macro_block_size=1)
```

### Distilled (few-step) text-to-image and image-to-video

Few-step distilled checkpoints are served by [`Cosmos3DistilledModularPipeline`] (blocks:
`Cosmos3DistilledBlocks`); the base [`Cosmos3OmniModularPipeline`] and [`Cosmos3OmniPipeline`] do
not support them. `num_inference_steps` is fixed to the length of the `distilled_sigmas` pipeline
config (from the checkpoint's `modular_model_index.json`) and `guidance_scale` is forced to
1.0 since guidance is baked into the weights — passing any other value for either raises an error,
and `negative_prompt` is warned about and ignored.

FP8 distilled checkpoints (`revision="fp8"`) do not declare a mixed-precision policy, so every
step stays native W8A8.

Prompts follow the same descriptive JSON structure as the non-distilled models, so short text
must be upsampled first — use `--mode text2image` (T2I) or `--mode image2video` (I2V) as
described in [Prompt upsampling](#prompt-upsampling), then pass the JSON via `json.dumps(...)`.

```python
import json
import torch
from diffusers import Cosmos3DistilledModularPipeline
from diffusers.utils import export_to_video, load_image

# JSON-upsampled prompt (see "Prompt upsampling" above).
json_prompt = json.load(open("assets/example_t2i_prompt.json"))

repo = "nvidia/Cosmos3-Super-Text2Image-4Step"
pipe = Cosmos3DistilledModularPipeline.from_pretrained(repo, dtype=torch.bfloat16)
pipe.load_components(dtype=torch.bfloat16)
pipe.to("cuda")  # or "mps", "xpu", "cpu"

# text-to-image (distilled)
videos = pipe(
    prompt=json.dumps(json_prompt),
    num_frames=1,
    height=720,
    width=1280,
    output="videos",
)
videos[0].save("cosmos3_distilled_t2i.jpg", format="JPEG", quality=85)

# image-to-video (distilled) — load the I2V repo instead
# JSON-upsampled prompt (see "Prompt upsampling" above); upsampled from the source prompt
# "The right robotic hand picks up the red sphere on the shelf."
json_prompt_i2v = json.load(open("assets/example_i2v_prompt.json"))

repo_i2v = "nvidia/Cosmos3-Super-Image2Video-4Step"
pipe = Cosmos3DistilledModularPipeline.from_pretrained(repo_i2v, dtype=torch.bfloat16)
pipe.load_components(dtype=torch.bfloat16)
pipe.to("cuda")

image = load_image(
    "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/vision/robot_153.jpg"
)
videos = pipe(
    prompt=json.dumps(json_prompt_i2v),
    image=image,
    num_frames=189,
    height=720,
    width=1280,
    output="videos",
)
export_to_video(videos, "cosmos3_distilled_i2v.mp4", fps=24, macro_block_size=1)
```

[[autodoc]] Cosmos3OmniModularPipeline

- all
- __call__

## Cosmos3DistilledModularPipeline

[[autodoc]] Cosmos3DistilledModularPipeline

- all
- __call__

## CosmosActionCondition

[[autodoc]] CosmosActionCondition

## Cosmos3OmniPipelineOutput

[[autodoc]] pipelines.cosmos.pipeline_cosmos3_omni.Cosmos3OmniPipelineOutput
