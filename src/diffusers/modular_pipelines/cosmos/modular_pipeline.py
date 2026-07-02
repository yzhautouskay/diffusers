import torch

from ...pipelines.cosmos.pipeline_cosmos3_omni import Cosmos3OmniPipeline, CosmosSafetyChecker
from ...utils import logging
from ...video_processor import VideoProcessor
from ..modular_pipeline import ModularPipeline, PipelineState


logger = logging.get_logger(__name__)


class Cosmos3OmniModularPipeline(ModularPipeline):
    """
    A ModularPipeline for Cosmos 3 omni generation.
    """

    default_blocks_name = "Cosmos3OmniBlocks"
    _callback_tensor_inputs = ["latents"]
    _exclude_from_cpu_offload = ["safety_checker"]
    model_cpu_offload_seq = "transformer->vae->sound_tokenizer"

    def _ensure_runtime_attributes(self):
        if getattr(self, "vae", None) is not None:
            self._vae_latents_mean = torch.tensor(self.vae.config.latents_mean, dtype=self.vae.dtype)
            self._vae_latents_inv_std = 1.0 / torch.tensor(self.vae.config.latents_std, dtype=self.vae.dtype)
            self.vae_scale_factor_spatial = int(self.vae.config.scale_factor_spatial)
        elif not hasattr(self, "vae_scale_factor_spatial"):
            self.vae_scale_factor_spatial = 16

        if getattr(self, "text_tokenizer", None) is not None:
            self.llm_special_tokens = {
                "start_of_generation": self.text_tokenizer.convert_tokens_to_ids("<|vision_start|>"),
                "eos_token_id": self.text_tokenizer.eos_token_id,
            }

        if getattr(self, "video_processor", None) is None:
            self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial, resample="bilinear")

        self.duration_template = "The video is {duration:.1f} seconds long and is of {fps:.0f} FPS."
        self.image_resolution_template = "This image is of {height}x{width} resolution."
        self.video_resolution_template = "This video is of {height}x{width} resolution."
        self.inverse_duration_template = "The video is not {duration:.1f} seconds long and is not of {fps:.0f} FPS."
        self.inverse_image_resolution_template = "This image is not of {height}x{width} resolution."
        self.inverse_video_resolution_template = "This video is not of {height}x{width} resolution."

        if not hasattr(self, "safety_checker"):
            self.safety_checker = None
        if not hasattr(self, "_current_timestep"):
            self._current_timestep = None
        if not hasattr(self, "_interrupt"):
            self._interrupt = False
        if not hasattr(self, "_guidance_scale"):
            self._guidance_scale = 1.0

    def _ensure_safety_checker(self):
        if getattr(self, "safety_checker", None) is None:
            self.safety_checker = CosmosSafetyChecker()

    def __call__(self, state: PipelineState = None, output: str | list[str] | None = None, **kwargs):
        self._ensure_runtime_attributes()
        if output is None:
            output = "result"
        return super().__call__(state=state, output=output, **kwargs)

    def _get_execution_device(self):
        return Cosmos3OmniPipeline._get_execution_device(self)

    def _encode_video(self, x):
        return Cosmos3OmniPipeline._encode_video(self, x)

    def decode_sound(self, latent):
        return Cosmos3OmniPipeline.decode_sound(self, latent)

    def _prepare_text_segment(self, input_ids, device):
        return Cosmos3OmniPipeline._prepare_text_segment(self, input_ids, device)

    def _prepare_vision_segment(self, *args, **kwargs):
        return Cosmos3OmniPipeline._prepare_vision_segment(self, *args, **kwargs)

    def _prepare_sound_segment(self, *args, **kwargs):
        return Cosmos3OmniPipeline._prepare_sound_segment(self, *args, **kwargs)

    def _prepare_action_segment(self, *args, **kwargs):
        return Cosmos3OmniPipeline._prepare_action_segment(self, *args, **kwargs)

    def _prepare_action_video_conditioning(self, *args, **kwargs):
        return Cosmos3OmniPipeline._prepare_action_video_conditioning(self, *args, **kwargs)

    def _remove_action_video_padding_from_latent(self, *args, **kwargs):
        return Cosmos3OmniPipeline._remove_action_video_padding_from_latent(self, *args, **kwargs)

    def prepare_latents(self, *args, **kwargs):
        return Cosmos3OmniPipeline.prepare_latents(self, *args, **kwargs)

    def check_inputs(self, *args, **kwargs):
        return Cosmos3OmniPipeline.check_inputs(self, *args, **kwargs)

    @staticmethod
    def _build_action_json_prompt(*args, **kwargs):
        return Cosmos3OmniPipeline._build_action_json_prompt(*args, **kwargs)

    def tokenize_prompt(self, *args, **kwargs):
        return Cosmos3OmniPipeline.tokenize_prompt(self, *args, **kwargs)

    @staticmethod
    def _mask_velocity_predictions(*args, **kwargs):
        return Cosmos3OmniPipeline._mask_velocity_predictions(*args, **kwargs)

    def _apply_video_safety_check(self, *args, **kwargs):
        return Cosmos3OmniPipeline._apply_video_safety_check(self, *args, **kwargs)

    def maybe_free_model_hooks(self):
        for component in self.components.values():
            if hasattr(component, "_reset_stateful_cache"):
                component._reset_stateful_cache()

        model_hooks = getattr(self._components_manager, "model_hooks", None)
        if not model_hooks:
            return

        for hook in model_hooks:
            hook.offload()

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale != 1.0
