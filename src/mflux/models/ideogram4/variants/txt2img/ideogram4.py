from typing import Any

import mlx.core as mx
from mlx import nn

from mflux.models.common.config import Config, ModelConfig
from mflux.models.common.pid_decoder.pid_decoder import pid_decode_latents
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.models.ideogram4.ideogram4_initializer import Ideogram4Initializer
from mflux.models.ideogram4.latent_creator import Ideogram4LatentCreator
from mflux.models.ideogram4.model.ideogram4_scheduler import Ideogram4Scheduler
from mflux.models.ideogram4.model.ideogram4_text_encoder import Ideogram4PromptEncoder, Qwen3TextEncoder
from mflux.models.ideogram4.model.ideogram4_transformer import Ideogram4Transformer
from mflux.models.ideogram4.weights import Ideogram4WeightDefinition
from mflux.utils.apple_silicon import AppleSiliconUtil
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil


class Ideogram4(nn.Module):
    vae: Flux2VAE
    text_encoder: Qwen3TextEncoder | None
    conditional_transformer: Ideogram4Transformer
    unconditional_transformer: Ideogram4Transformer

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        model_config: ModelConfig = ModelConfig.ideogram4_fp8(),
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        bake_lora: bool = True,
    ):
        super().__init__()
        Ideogram4Initializer.init(
            model=self,
            quantize=quantize,
            model_path=model_path,
            model_config=model_config,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            bake_lora=bake_lora,
        )

    def generate_image(
        self,
        seed: int,
        prompt: str | dict[str, Any],
        num_inference_steps: int | None = None,
        height: int = 1024,
        width: int = 1024,
        guidance: float | None = None,
        preset: str | None = None,
        strict_caption_validation: bool = False,
        warn_on_caption_issues: bool = True,
        cfg_end: float | None = None,
        pid_decode: bool = False,
        pid_degrade_sigma: float = 0.0,
    ) -> GeneratedImage:
        prompt = Ideogram4PromptEncoder.resolve_prompt(
            prompt,
            strict_caption_validation=strict_caption_validation,
            warn_on_caption_issues=warn_on_caption_issues,
        )
        Ideogram4LatentCreator.validate_dimensions(width=width, height=height)

        sampler = Ideogram4Scheduler.get_preset(preset)
        if num_inference_steps is None:
            num_steps = sampler.num_steps
            guidance_values = sampler.guidance_schedule
        else:
            if num_inference_steps < 1:
                raise ValueError(f"num_inference_steps must be >= 1, got {num_inference_steps}")
            num_steps = num_inference_steps
            guidance_values = (float(guidance if guidance is not None else 7.0),) * num_steps
        config = Config(
            width=width,
            height=height,
            guidance=guidance if guidance is not None else guidance_values[-1],
            scheduler="linear",
            model_config=self.model_config,
            num_inference_steps=num_steps,
        )

        inputs = Ideogram4PromptEncoder.build_inputs(
            self.tokenizers["ideogram4"],
            [prompt],
            height=config.height,
            width=config.width,
        )
        llm_features = Ideogram4PromptEncoder.encode_prompt(
            prompt=prompt,
            width=config.width,
            height=config.height,
            inputs=inputs,
            text_encoder=self.text_encoder,
            prompt_cache=self.prompt_cache,
        )
        z = Ideogram4LatentCreator.create_noise(
            seed=seed,
            width=config.width,
            height=config.height,
            latent_dim=self.conditional_transformer.config.in_channels,
        )
        text_z_padding = mx.zeros(
            (1, int(inputs["max_text_tokens"]), self.conditional_transformer.config.in_channels),
            dtype=mx.float32,
        )
        negative_inputs = Ideogram4PromptEncoder.negative_inputs(inputs, llm_features)
        t_values, s_values = Ideogram4Scheduler.make_timesteps(
            num_steps=num_steps,
            height=config.height,
            width=config.width,
            mu=sampler.mu,
            std=sampler.std,
        )

        # CFG truncation: guidance shapes the result in the early steps (composition); the
        # late steps mostly refine. cfg_end = fraction of steps that run CFG; the remaining
        # steps run cond-only (guidance 1.0, which skips the unconditional forward entirely).
        cfg_cutoff = num_steps if cfg_end is None else max(1, int(round(float(cfg_end) * num_steps)))

        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(z)
        predict_conditional = self._predict_conditional(self.conditional_transformer)
        predict_unconditional = self._predict_unconditional(self.unconditional_transformer)
        time_steps = config.time_steps
        for step_index in time_steps:
            try:
                schedule_index = num_steps - 1 - step_index
                z = self._denoise_step(
                    z=z,
                    t_value=float(t_values[schedule_index]),
                    s_value=float(s_values[schedule_index]),
                    guidance_value=float(guidance_values[schedule_index]) if step_index < cfg_cutoff else 1.0,
                    text_z_padding=text_z_padding,
                    llm_features=llm_features,
                    inputs=inputs,
                    negative_inputs=negative_inputs,
                    predict_conditional=predict_conditional,
                    predict_unconditional=predict_unconditional,
                )
                ctx.in_loop(step_index, z, time_steps=time_steps)
                mx.eval(z)
            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(step_index, z, time_steps=time_steps)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {step_index + 1}/{config.num_inference_steps}"
                )
        ctx.after_loop(z)

        decoded = self._decode_latents(
            z=z, config=config, prompt=prompt, seed=seed, pid_decode=pid_decode, degrade_sigma=pid_degrade_sigma
        )
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            generation_time=time_steps.format_dict["elapsed"],
            pid_decode=pid_decode,
            pid_degrade_sigma=pid_degrade_sigma,
        )

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=Ideogram4WeightDefinition,
        )

    def _decode_latents(
        self,
        *,
        z: mx.array,
        config: Config,
        prompt: str,
        seed: int,
        pid_decode: bool = False,
        degrade_sigma: float = 0.0,
    ) -> mx.array:
        latents = Ideogram4LatentCreator.unpack_latents(z, config.height, config.width)
        if pid_decode:
            return pid_decode_latents(
                vae=self.vae, latent=latents, caption=prompt, seed=seed, degrade_sigma=degrade_sigma
            )
        return self.vae.decode(latents)

    @staticmethod
    def _predict_conditional(transformer: Ideogram4Transformer):
        def predict(
            *,
            z: mx.array,
            t: mx.array,
            text_z_padding: mx.array,
            llm_features: mx.array,
            inputs: dict[str, Any],
        ) -> mx.array:
            pos_z = mx.concatenate([text_z_padding, z], axis=1)
            pos_out = transformer(
                llm_features=llm_features,
                x=pos_z,
                t=t,
                position_ids=inputs["position_ids"],
                segment_ids=inputs["segment_ids"],
                indicator=inputs["indicator"],
            )
            return pos_out[:, int(inputs["max_text_tokens"]) :, :]

        if AppleSiliconUtil.is_m1_or_m2():
            return predict
        return mx.compile(predict)

    @staticmethod
    def _predict_unconditional(transformer: Ideogram4Transformer):
        def predict(
            *,
            z: mx.array,
            t: mx.array,
            negative_inputs: dict[str, mx.array],
        ) -> mx.array:
            return transformer(
                llm_features=negative_inputs["llm_features"],
                x=z,
                t=t,
                position_ids=negative_inputs["position_ids"],
                segment_ids=negative_inputs["segment_ids"],
                indicator=negative_inputs["indicator"],
            )

        if AppleSiliconUtil.is_m1_or_m2():
            return predict
        return mx.compile(predict)

    def _denoise_step(
        self,
        *,
        z: mx.array,
        t_value: float,
        s_value: float,
        guidance_value: float,
        text_z_padding: mx.array,
        llm_features: mx.array,
        inputs: dict[str, Any],
        negative_inputs: dict[str, mx.array],
        predict_conditional,
        predict_unconditional,
    ) -> mx.array:
        t = mx.full((1,), t_value, dtype=mx.float32)
        pos_v = predict_conditional(
            z=z,
            t=t,
            text_z_padding=text_z_padding,
            llm_features=llm_features,
            inputs=inputs,
        )
        # guidance == 1.0 makes the negative term vanish (v = pos_v exactly) — skip the
        # whole unconditional forward, halving the cost of CFG-free steps.
        if abs(guidance_value - 1.0) < 1e-6:
            return z + pos_v * (s_value - t_value)
        neg_v = predict_unconditional(z=z, t=t, negative_inputs=negative_inputs)
        v = guidance_value * pos_v + (1.0 - guidance_value) * neg_v
        return z + v * (s_value - t_value)
