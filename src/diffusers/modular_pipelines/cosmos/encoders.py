import torch
from transformers import AutoTokenizer

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...models.autoencoders.autoencoder_cosmos3_audio import Cosmos3AVAEAudioTokenizer
from ...models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from ...models.transformers.transformer_cosmos3 import Cosmos3OmniTransformer
from ...utils import logging
from ...pipelines.cosmos.pipeline_cosmos3_omni import (
    _ACTION_RESOLUTION_BINS,
    CosmosActionCondition,
    CosmosSafetyChecker,
)
from ...video_processor import VideoProcessor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import Cosmos3OmniModularPipeline


logger = logging.get_logger(__name__)


class Cosmos3TextEncoderStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return "Validates inputs, tokenizes prompts, and packs text conditioning."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("transformer", Cosmos3OmniTransformer),
            ComponentSpec("text_tokenizer", AutoTokenizer),
            ComponentSpec("vae", AutoencoderKLWan),
            ComponentSpec("sound_tokenizer", Cosmos3AVAEAudioTokenizer),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(name="prompt", type_hint=str, required=True),
            InputParam(name="negative_prompt", default=None),
            InputParam(name="image", default=None),
            InputParam(name="video", default=None),
            InputParam(name="condition_frame_indexes_vision", default=(0, 1)),
            InputParam(name="condition_video_keep", default="first"),
            InputParam(name="num_frames", default=None),
            InputParam(name="height", default=None),
            InputParam(name="width", default=None),
            InputParam(name="fps", type_hint=float, default=24.0),
            InputParam(name="num_inference_steps", type_hint=int, default=35),
            InputParam(name="guidance_scale", type_hint=float, default=6.0),
            InputParam(name="enable_sound", type_hint=bool, default=False),
            InputParam(name="action", type_hint=CosmosActionCondition, default=None),
            InputParam(name="use_system_prompt", type_hint=bool, default=True),
            InputParam(name="callback_on_step_end", default=None),
            InputParam(name="callback_on_step_end_tensor_inputs", default=["latents"]),
            InputParam(name="add_resolution_template", type_hint=bool, default=True),
            InputParam(name="add_duration_template", type_hint=bool, default=True),
            InputParam(name="enable_safety_check", type_hint=bool, default=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("action_mode"),
            OutputParam("device"),
            OutputParam("dtype"),
            OutputParam("cond_text_segment"),
            OutputParam("uncond_text_segment"),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        if isinstance(block_state.callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            block_state.callback_on_step_end_tensor_inputs = block_state.callback_on_step_end.tensor_inputs

        if block_state.action is None:
            if block_state.num_frames is None:
                block_state.num_frames = 189
            if block_state.height is None:
                block_state.height = 720
            if block_state.width is None:
                block_state.width = 1280

        components.check_inputs(
            block_state.prompt,
            block_state.negative_prompt,
            block_state.image,
            block_state.height,
            block_state.width,
            block_state.num_frames,
            block_state.guidance_scale,
            block_state.enable_sound,
            block_state.callback_on_step_end_tensor_inputs,
            block_state.action,
            video=block_state.video,
            condition_frame_indexes_vision=block_state.condition_frame_indexes_vision,
        )

        block_state.action_mode = block_state.action.mode if block_state.action is not None else None
        if block_state.action is not None:
            block_state.num_frames = block_state.action.chunk_size + 1
            conditioning_clip = (
                [block_state.action.image] if block_state.action.image is not None else block_state.action.video
            )
            probe = components.video_processor.preprocess_video(conditioning_clip)
            source_h, source_w = int(probe.shape[-2]), int(probe.shape[-1])
            resolution_key = str(block_state.action.resolution_tier)
            block_state.height, block_state.width = VideoProcessor.classify_height_width_bin(
                source_h, source_w, ratios=_ACTION_RESOLUTION_BINS[resolution_key]
            )

        components._current_timestep = None
        components._interrupt = False
        components._guidance_scale = block_state.guidance_scale

        if isinstance(block_state.prompt, list):
            block_state.prompt = block_state.prompt[0]
        if isinstance(block_state.negative_prompt, list):
            block_state.negative_prompt = block_state.negative_prompt[0]

        block_state.device = components._get_execution_device()
        block_state.dtype = components.transformer.dtype

        if block_state.enable_safety_check and getattr(components, "safety_checker", None) is None:
            try:
                components._ensure_safety_checker()
            except ImportError:
                pass

        if block_state.enable_safety_check and isinstance(components.safety_checker, CosmosSafetyChecker):
            components.safety_checker.to(block_state.device)
            try:
                if not components.safety_checker.check_text_safety(block_state.prompt):
                    raise ValueError(
                        f"Cosmos Guardrail detected unsafe text in the prompt: {block_state.prompt}. "
                        "Please ensure that the prompt abides by the NVIDIA Open Model License Agreement."
                    )
            finally:
                components.safety_checker.to("cpu")

        cond_input_ids, uncond_input_ids = components.tokenize_prompt(
            block_state.prompt,
            block_state.negative_prompt,
            num_frames=block_state.num_frames,
            height=block_state.height,
            width=block_state.width,
            fps=block_state.fps,
            use_system_prompt=block_state.use_system_prompt,
            add_resolution_template=block_state.add_resolution_template,
            add_duration_template=block_state.add_duration_template,
            action_mode=block_state.action_mode,
            action_view_point=block_state.action.view_point if block_state.action is not None else None,
        )
        block_state.cond_text_segment = components._prepare_text_segment(cond_input_ids, device=block_state.device)
        block_state.uncond_text_segment = components._prepare_text_segment(uncond_input_ids, device=block_state.device)

        self.set_block_state(state, block_state)
        return components, state
