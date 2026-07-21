#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark runner for Cosmos3 modular pipelines.

The runner supports base Nano/Super checkpoints through ``Cosmos3OmniModularPipeline``,
distilled few-step T2I/I2V checkpoints through ``Cosmos3DistilledModularPipeline``, and
structurally controlled transfer generation with base checkpoints.

Distilled T2I:
    python inference_cosmos3_modular.py --model super-t2i-4step \
        --prompt "A robot in a laboratory." --num-frames 1

Transfer with an edge control video and JSON captions:
    python inference_cosmos3_modular.py --model nano \
        --prompt-path assets/edge/prompt.json \
        --negative-prompt-path assets/negative_prompt.json \
        --control-video edge=assets/edge/control_edge.mp4 \
        --num-frames 121 --fps 30 --guidance-scale 3.0 \
        --control-guidance 1.5 --flow-shift 10.0
"""

import argparse
import json
import pathlib
import time

import torch

from diffusers import Cosmos3DistilledModularPipeline, Cosmos3OmniModularPipeline, UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image, load_video


HF_REPOS = {
    "nano": "nvidia/Cosmos3-Nano",
    "super": "nvidia/Cosmos3-Super",
    "super-i2v-4step": "nvidia/Cosmos3-Super-Image2Video-4Step",
    "super-t2i-4step": "nvidia/Cosmos3-Super-Text2Image-4Step",
}
DISTILLED_MODELS = {"super-i2v-4step", "super-t2i-4step"}
CONTROL_HINTS = {"blur", "depth", "edge", "seg", "wsm"}


def _load_json_caption(path: str) -> str:
    with pathlib.Path(path).expanduser().open() as f:
        return json.dumps(json.load(f))


def _parse_control_videos(values: list[str] | None) -> dict[str, list] | None:
    if not values:
        return None

    control_videos = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --control-video {value!r}; expected HINT=PATH.")
        hint, path = value.split("=", 1)
        if hint not in CONTROL_HINTS:
            raise ValueError(f"Unknown control hint {hint!r}; choose from {sorted(CONTROL_HINTS)}.")
        if hint in control_videos:
            raise ValueError(f"Control hint {hint!r} was provided more than once.")
        control_videos[hint] = load_video(path)
    return control_videos


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", help="Text prompt, or a pre-serialized JSON caption for transfer.")
    prompt_group.add_argument("--prompt-path", help="Path to a JSON caption file (recommended for transfer).")
    parser.add_argument(
        "--model",
        choices=sorted(HF_REPOS),
        default="nano",
        help="Cosmos3 checkpoint to load.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional local diffusers checkpoint path. Overrides the repository selected by --model.",
    )
    parser.add_argument(
        "--vision-path",
        default=None,
        help="URL or local image path for image-to-video generation.",
    )
    parser.add_argument(
        "--control-video",
        action="append",
        metavar="HINT=PATH",
        help="Transfer control video. Repeat for multiple hints: edge, blur, depth, seg, or wsm.",
    )
    negative_prompt_group = parser.add_mutually_exclusive_group()
    negative_prompt_group.add_argument("--negative-prompt", default=None, help="Negative prompt text or JSON.")
    negative_prompt_group.add_argument(
        "--negative-prompt-path",
        default=None,
        help="Path to a negative-prompt JSON file (recommended for transfer).",
    )
    parser.add_argument("--output", default=".", help="Directory to save generated video/image files.")
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument(
        "--num-frames",
        type=int,
        default=189,
        help="Number of frames to generate. Use 1 for text-to-image; defaults to 189 for video.",
    )
    parser.add_argument(
        "--num-video-frames-per-chunk",
        type=int,
        default=None,
        help="Transfer frames generated per autoregressive chunk (defaults to all requested frames in one chunk).",
    )
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--control-guidance", type=float, default=1.5)
    parser.add_argument("--num-inference-steps", type=int, default=35)
    parser.add_argument(
        "--flow-shift",
        type=float,
        default=None,
        help="Override the base pipeline scheduler flow shift (10.0 is recommended for transfer).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for latent initialization.")
    parser.add_argument(
        "--no-duration-template",
        dest="add_duration_template",
        action="store_false",
        default=True,
        help="Skip the duration metadata sentence appended to prompts (video only).",
    )
    parser.add_argument(
        "--no-resolution-template",
        dest="add_resolution_template",
        action="store_false",
        default=True,
        help="Skip the resolution metadata sentence appended to prompts.",
    )
    parser.add_argument(
        "--disable-safety-checker",
        action="store_true",
        help="Disable the Cosmos Guardrail safety checker.",
    )
    parser.add_argument("--warmup", type=int, default=0, help="Number of untimed warmup generations.")
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=1,
        help="Number of timed generations used to calculate average end-to-end time.",
    )
    args = parser.parse_args()

    is_distilled = args.model in DISTILLED_MODELS
    is_transfer = bool(args.control_video)
    if is_distilled and is_transfer:
        raise ValueError("Transfer requires a base Nano/Super checkpoint; distilled checkpoints are unsupported.")
    if is_transfer and args.vision_path is not None:
        raise ValueError("--vision-path and --control-video cannot be used together.")
    if not is_transfer and args.num_video_frames_per_chunk is not None:
        raise ValueError("--num-video-frames-per-chunk is only supported with --control-video.")
    if args.num_video_frames_per_chunk is not None and args.num_video_frames_per_chunk < 1:
        raise ValueError("--num-video-frames-per-chunk must be at least 1.")
    if args.model == "super-i2v-4step" and args.vision_path is None:
        raise ValueError("--vision-path is required for the distilled image-to-video checkpoint.")
    if args.num_iterations < 1:
        raise ValueError("--num-iterations must be at least 1.")
    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative.")

    prompt = _load_json_caption(args.prompt_path) if args.prompt_path else args.prompt
    if args.negative_prompt_path:
        negative_prompt = _load_json_caption(args.negative_prompt_path)
    else:
        negative_prompt = args.negative_prompt
    control_videos = _parse_control_videos(args.control_video)

    if args.model_path is None:
        pipeline_source = HF_REPOS[args.model]
    else:
        pipeline_source = str(pathlib.Path(args.model_path).expanduser().resolve())

    pipeline_cls = Cosmos3DistilledModularPipeline if is_distilled else Cosmos3OmniModularPipeline
    print(f"Loading {pipeline_cls.__name__} from {pipeline_source} …")
    pipeline = pipeline_cls.from_pretrained(pipeline_source, torch_dtype=torch.bfloat16)
    pipeline.load_components(torch_dtype=torch.bfloat16)
    missing_components = [
        name for name in ("text_tokenizer", "vae", "transformer", "scheduler") if getattr(pipeline, name, None) is None
    ]
    if missing_components:
        raise RuntimeError(f"Failed to load required pipeline component(s): {', '.join(missing_components)}")
    if args.disable_safety_checker:
        pipeline.disable_safety_checker()
    else:
        pipeline.enable_safety_checker()
    if args.flow_shift is not None:
        if is_distilled:
            raise ValueError("--flow-shift cannot be overridden for distilled checkpoints.")
        pipeline.scheduler = UniPCMultistepScheduler.from_config(
            pipeline.scheduler.config, flow_shift=args.flow_shift, use_karras_sigmas=False
        )
    pipeline.to("cuda")
    print("Pipeline loaded successfully.")

    output_dir = pathlib.Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    image = load_image(args.vision_path) if args.vision_path is not None else None
    generator = torch.Generator(device="cpu").manual_seed(args.seed) if args.seed is not None else None

    def generate():
        kwargs = {
            "prompt": prompt,
            "num_frames": args.num_frames,
            "height": args.height,
            "width": args.width,
            "fps": args.fps,
            "generator": generator,
            "add_resolution_template": args.add_resolution_template,
            "add_duration_template": args.add_duration_template,
            "output": "videos",
        }
        if is_distilled:
            # Fixed-step scheduler configuration supplies the step count and guidance scale.
            kwargs["image"] = image
        else:
            kwargs.update(
                {
                    "negative_prompt": negative_prompt,
                    "num_inference_steps": args.num_inference_steps,
                    "guidance_scale": args.guidance_scale,
                }
            )
            if is_transfer:
                kwargs["control_videos"] = control_videos
                kwargs["control_guidance"] = args.control_guidance
                kwargs["num_video_frames_per_chunk"] = args.num_video_frames_per_chunk
            else:
                kwargs["image"] = image
        return pipeline(**kwargs)

    for i in range(args.warmup):
        print(f"[warmup {i + 1}/{args.warmup}]")
        generate()

    timings = []
    videos = None
    for i in range(args.num_iterations):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        videos = generate()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        timings.append(elapsed)
        print(f"[iter {i + 1}/{args.num_iterations}] E2E generation time: {elapsed:.3f}s")

    print(f"Average E2E generation time over {len(timings)} iteration(s): {sum(timings) / len(timings):.3f}s")

    if args.num_frames == 1:
        save_path = output_dir / "sample.jpg"
        videos[0].save(save_path, format="JPEG", quality=85)
    else:
        save_path = output_dir / "sample.mp4"
        export_to_video(videos, str(save_path), fps=int(args.fps), quality=10, macro_block_size=1)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
