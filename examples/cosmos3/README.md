# Cosmos3 — smoke-test runners

The canonical reference for `Cosmos3OmniPipeline` lives in the diffusers docs:
[`docs/source/en/api/pipelines/cosmos3.md`](../../docs/source/en/api/pipelines/cosmos3.md). Use the
examples there as the source of truth for application code — they cover text-to-image,
text-to-video, image-to-video, and text+sound modes.

This directory provides small CLI wrappers that exercise the full load → encode → denoise → decode
path against either the Hub release or a local checkpoint during development.

| Script | Pipeline | Use case |
|---|---|---|
| `inference_cosmos3.py` | task-based [`Cosmos3OmniPipeline`] | Base checkpoints (Nano/Super): T2I, T2V, I2V, V2V, action, sound |
| `inference_cosmos3_modular.py` | modular [`Cosmos3OmniModularPipeline`] / [`Cosmos3DistilledModularPipeline`] | Base and distilled T2I/I2V, transfer, and E2E benchmarking |

Distilled models are **not** supported through `inference_cosmos3.py`. Use the modular script
above. It selects the dedicated [`Cosmos3DistilledModularPipeline`] for few-step checkpoints
and [`Cosmos3OmniModularPipeline`] for base checkpoints and transfer.

## Setup

```bash
pip install -r examples/cosmos3/requirements.txt
```

## Usage

Text-to-image:

```bash
python examples/cosmos3/inference_cosmos3.py \
    --prompt "A medium shot of a modern robotics research laboratory…" \
    --num-frames 1
```

Text-to-video:

```bash
python examples/cosmos3/inference_cosmos3.py \
    --prompt "A waterfall cascading down a rocky cliff in a lush forest."
```

Image-to-video:

```bash
python examples/cosmos3/inference_cosmos3.py \
    --prompt "The right robotic hand picks up the red sphere…" \
    --vision-path https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/assets/robot_153.jpg
```

Video-to-video (condition on the leading frames of a clip and continue it):

```bash
python examples/cosmos3/inference_cosmos3.py \
    --prompt "A robotic arm finishes pouring liquid into the glass." \
    --video-path "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/vision/robot_pouring.mp4" \
    --condition-frame-indexes-vision 0,1 \
    --condition-video-keep first
```

Text-to-video-with-sound (sound-capable checkpoint only):

```bash
python examples/cosmos3/inference_cosmos3.py \
    --prompt "A waterfall in a lush forest." \
    --enable-sound
```

Action forward dynamics, robot domain (predict video from an observation video and a provided action chunk):

```bash
python examples/cosmos3/inference_cosmos3.py \
    --model nano \
    --prompt "Put the pot to the left of the purple item." \
    --vision-path "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/action/bridge_0.mp4" \
    --action-mode forward_dynamics \
    --action-path "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/action/bridge_0.json" \
    --action-chunk-size 16 \
    --domain-name bridge_orig_lerobot \
    --resolution-tier 480 --fps 5 \
    --num-inference-steps 30 --guidance-scale 1.0 --flow-shift 10.0 --seed 0 \
    --output results/cosmos3_forward_dynamics_robot
```

Action forward dynamics, autonomous-vehicle domain:

```bash
python examples/cosmos3/inference_cosmos3.py \
    --model nano \
    --prompt "You are an autonomous vehicle planning system." \
    --vision-path "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/action/av_vision_25_73d01c91-51f0-46cf-9b76-5682a76fb349.mp4" \
    --action-mode forward_dynamics \
    --action-path "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/action/av_action_25.json" \
    --action-chunk-size 60 \
    --domain-name av \
    --resolution-tier 480 --fps 10 \
    --num-inference-steps 30 --guidance-scale 1.0 --flow-shift 10.0 --seed 0 \
    --output results/cosmos3_forward_dynamics_av
```

Action inverse dynamics, robot domain (predict actions from an observed video):

```bash
python examples/cosmos3/inference_cosmos3.py \
    --model nano \
    --prompt "Put the pot to the left of the purple item." \
    --vision-path "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/action/bridge_0.mp4" \
    --action-mode inverse_dynamics \
    --action-chunk-size 16 \
    --domain-name bridge_orig_lerobot \
    --resolution-tier 480 --fps 5 \
    --num-inference-steps 30 --guidance-scale 1.0 --flow-shift 10.0 --seed 0 \
    --output results/cosmos3_inverse_dynamics_robot
```

Action inverse dynamics, autonomous-vehicle domain:

```bash
python examples/cosmos3/inference_cosmos3.py \
    --model nano \
    --prompt "You are an autonomous vehicle planning system." \
    --vision-path "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/action/av_vision_25_73d01c91-51f0-46cf-9b76-5682a76fb349.mp4" \
    --action-mode inverse_dynamics \
    --action-chunk-size 60 \
    --domain-name av \
    --resolution-tier 480 --fps 10 \
    --num-inference-steps 30 --guidance-scale 1.0 --flow-shift 10.0 --seed 0 \
    --output results/cosmos3_inverse_dynamics_av
```

Action policy, robot domain (predict both future video and actions from the first observation frame):

```bash
python examples/cosmos3/inference_cosmos3.py \
    --model nano \
    --prompt "Put the pot to the left of the purple item." \
    --vision-path "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/action/bridge_0.mp4" \
    --action-mode policy \
    --action-chunk-size 16 \
    --domain-name bridge_orig_lerobot \
    --resolution-tier 480 --fps 5 \
    --num-inference-steps 30 --guidance-scale 1.0 --flow-shift 10.0 --seed 0 \
    --output results/cosmos3_policy_robot
```

Action policy, autonomous-vehicle domain:

```bash
python examples/cosmos3/inference_cosmos3.py \
    --model nano \
    --prompt "You are an autonomous vehicle planning system. Please go backward." \
    --vision-path "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/action/av_vision_25_73d01c91-51f0-46cf-9b76-5682a76fb349.mp4" \
    --action-mode policy \
    --action-chunk-size 60 \
    --domain-name av \
    --resolution-tier 480 --fps 10 \
    --num-inference-steps 30 --guidance-scale 1.0 --flow-shift 10.0 --seed 0 \
    --output results/cosmos3_policy_av
```

Action modes use `action_chunk_size + 1` conditioning frames. `forward_dynamics` consumes `--action-path`; `inverse_dynamics` and `policy` write predicted actions to `sample_action.json` in model-normalized action space. This script loads `--vision-path` as a video for all action modes; `policy` and `forward_dynamics` condition only on the first frame, while `inverse_dynamics` uses the whole video.

Pass `--prompt` as a plain task description and select the camera perspective with `--view-point` (default `ego_view`); the pipeline builds the structured action caption (task, viewpoint, duration, FPS, resolution) the model was trained on. Do not hand-write the viewpoint sentence into `--prompt`.

`--resolution-tier` is a resolution *tier* (`256`/`480`/`704`/`720`). The tier keys a table of predefined aspect-ratio canvases; the one closest to the input aspect ratio becomes the padded conditioning canvas. It is not the output frame size: the input is downscaled (never upscaled) and padded to fill the canvas, then the padding is cropped from the latents so the decoded output follows the downscaled input content. `--height` / `--width` (and `--num-frames`) are ignored for action modes.

Pick the tier that matches the native resolution of your conditioning input (`480` for ~480p, `720` for ~720p). A tier below your input downscales it and discards detail; a tier above your input gains no resolution (content is never upscaled), wastes compute on padding, and is a train/inference distribution mismatch that can degrade quality.

## Modular pipeline benchmarks

Few-step distilled Cosmos3 checkpoints (`nvidia/Cosmos3-Super-Text2Image-4Step`,
`nvidia/Cosmos3-Super-Image2Video-4Step`) are served by the dedicated
[`Cosmos3DistilledModularPipeline`]. Use `inference_cosmos3_modular.py` — not
`inference_cosmos3.py`.

Distilled inference reads `fixed_step_sampler_config` from the scheduler, fixes the step
count, and forces `guidance_scale=1.0`, so do **not** pass `num_inference_steps`,
`guidance_scale`, or `negative_prompt`.

```bash
# T2I distilled (benchmarking)
python examples/cosmos3/inference_cosmos3_modular.py \
    --model super-t2i-4step \
    --prompt "A medium shot of a modern robotics research laboratory …" \
    --num-frames 1 --height 720 --width 1280 \
    --disable-safety-checker --warmup 2 --num-iterations 3 \
    --output outputs/text2image/

# I2V distilled (benchmarking)
python examples/cosmos3/inference_cosmos3_modular.py \
    --model super-i2v-4step \
    --prompt "The video opens with a view of a testing environment …" \
    --vision-path https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/vision/robot_153.jpg \
    --num-frames 189 --height 720 --width 1280 \
    --disable-safety-checker --warmup 2 --num-iterations 3 \
    --output outputs/image2video/
```

Transfer uses a base Nano/Super checkpoint and one or more precomputed control videos. Pass each
control as `HINT=PATH`, where `HINT` is `edge`, `blur`, `depth`, `seg`, or `wsm`. JSON caption files
are serialized for the pipeline automatically. Download the edge control video and captions from
the Cosmos cookbook:

```bash
base=https://github.com/NVIDIA/cosmos/raw/refs/heads/main/cookbooks/cosmos3/generator/transfer/assets
mkdir -p assets/edge
curl -L "$base/edge/control_edge.mp4" -o assets/edge/control_edge.mp4
curl -L "$base/edge/prompt.json" -o assets/edge/prompt.json
curl -L "$base/negative_prompt.json" -o assets/negative_prompt.json
```

Then run the modular transfer benchmark:

```bash
python examples/cosmos3/inference_cosmos3_modular.py \
    --model nano \
    --prompt-path assets/edge/prompt.json \
    --negative-prompt-path assets/negative_prompt.json \
    --control-video edge=assets/edge/control_edge.mp4 \
    --num-frames 121 --height 720 --width 1280 --fps 30 \
    --num-inference-steps 35 --guidance-scale 3.0 \
    --control-guidance 1.5 --flow-shift 10.0 \
    --disable-safety-checker --warmup 1 --num-iterations 2 \
    --output outputs/transfer-edge/
```

Point at a local snapshot with `--model-path`.
See also [`docs/source/en/api/pipelines/cosmos3.md`](../../docs/source/en/api/pipelines/cosmos3.md#cosmos3omnimodularpipeline).

### Useful flags

| Flag | Default | Description |
|---|---|---|
| `--prompt` | (required) | Text prompt. |
| `--vision-path` | `None` | URL or local path for an image-conditioning frame (image-to-video), or the image/video conditioning for action modes. |
| `--num-frames` | `189` | `1` = image, otherwise number of video frames (`189` ≈ 7.9 s @ 24 FPS). Ignored for action modes (derived from `--action-chunk-size`). |
| `--height` / `--width` | `720` / `1280` | Output resolution (must be a multiple of the VAE spatial scale factor). Ignored for action modes; use `--resolution-tier`. |
| `--resolution-tier` | `480` | Action resolution tier (`256`/`480`/`704`/`720`): selects the aspect bin / padded conditioning canvas, not the output size. |
| `--fps` | `24.0` | Frame rate of the generated video. |
| `--flow-shift` | `None` | Override `UniPCMultistepScheduler.flow_shift` (and force `use_karras_sigmas=False`); left at the checkpoint default when unset. Cosmos3 runs use `10.0`. |
| `--enable-sound` | off | Generate a synchronized audio track. |
| `--action-mode` | `None` | Enable action conditioning/generation. One of `forward_dynamics`, `inverse_dynamics`, or `policy`. |
| `--action-path` | `None` | URL or local JSON action path for `forward_dynamics`. |
| `--action-chunk-size` | `None` | Number of action tokens. Action runs generate/use `action_chunk_size + 1` video frames. |
| `--domain-name` | `None` | Action embodiment domain, for example `bridge_orig_lerobot` or `av`. |
| `--view-point` | `ego_view` | Camera perspective for the action caption's framing (`ego_view`, `third_person_view`, `wrist_view`, `concat_view`). Action only. |
| `--no-duration-template` | off | Skip the duration metadata sentence appended to the prompt and negative prompt. Ignored for `--num-frames 1` and for action modes (which build a structured caption instead). |
| `--no-resolution-template` | off | Skip the resolution metadata sentence appended to the prompt and negative prompt. Ignored for action modes. |
| `--output` | `.` | Directory to write `sample.jpg` or `sample.mp4`. |
