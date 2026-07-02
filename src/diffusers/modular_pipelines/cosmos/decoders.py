import torch

from ...models.autoencoders.autoencoder_cosmos3_audio import Cosmos3AVAEAudioTokenizer
from ...models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from ...pipelines.cosmos.pipeline_cosmos3_omni import Cosmos3OmniPipelineOutput, CosmosSafetyChecker
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import Cosmos3OmniModularPipeline


logger = logging.get_logger(__name__)


class Cosmos3DecodeStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return "Decodes denoised latents into video/sound/action outputs."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLWan),
            ComponentSpec("sound_tokenizer", Cosmos3AVAEAudioTokenizer),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(name="latents", required=True),
            InputParam(name="sound_latents", default=None),
            InputParam(name="action_latents", default=None),
            InputParam(name="action_mode", default=None),
            InputParam(name="raw_action_dim_resolved", default=None),
            InputParam.template("output_type", default="pil"),
            InputParam(name="enable_safety_check", default=True),
            InputParam(name="device", required=True),
            InputParam(name="return_dict", default=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("videos"),
            OutputParam("sound"),
            OutputParam("action"),
            OutputParam("result"),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        sound = components.decode_sound(block_state.sound_latents) if block_state.sound_latents is not None else None
        action_output = None
        if block_state.action_mode in {"inverse_dynamics", "policy"} and block_state.action_latents is not None:
            action_output = block_state.action_latents
            if block_state.raw_action_dim_resolved is not None:
                action_output = action_output[:, : block_state.raw_action_dim_resolved]
            action_output = [action_output.detach().cpu()]

        if block_state.output_type == "latent":
            video = block_state.latents
        else:
            in_dtype = block_state.latents.dtype
            vae_dtype = components.vae.dtype
            mean = components._vae_latents_mean.to(device=block_state.latents.device, dtype=vae_dtype)
            inv_std = components._vae_latents_inv_std.to(device=block_state.latents.device, dtype=vae_dtype)
            z_raw = block_state.latents.to(vae_dtype) / inv_std.view(1, -1, 1, 1, 1) + mean.view(1, -1, 1, 1, 1)
            decoded = components.vae.decode(z_raw).sample.to(in_dtype)
            video = components.video_processor.postprocess_video(decoded, output_type=block_state.output_type)[0]

        if (
            block_state.enable_safety_check
            and isinstance(components.safety_checker, CosmosSafetyChecker)
            and block_state.output_type != "latent"
        ):
            video = components._apply_video_safety_check(
                video, output_type=block_state.output_type, device=block_state.device
            )

        components.maybe_free_model_hooks()

        if not block_state.return_dict:
            if block_state.action_mode is not None:
                result = (video, sound, action_output)
            else:
                result = (video, sound)
        else:
            result = Cosmos3OmniPipelineOutput(video=video, sound=sound, action=action_output)

        block_state.videos = video
        block_state.sound = sound
        block_state.action = action_output
        block_state.result = result

        self.set_block_state(state, block_state)
        return components, state
