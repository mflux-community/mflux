from __future__ import annotations

import mlx.core as mx
from mlx import nn

from mflux.models.boogu.boogu_initializer import BooguInitializer
from mflux.models.boogu.model.boogu_text_encoder import BooguTextEncoder
from mflux.models.boogu.model.boogu_transformer.boogu_transformer import BooguImageTransformer
from mflux.models.boogu.weights.boogu_weight_definition import BooguWeightDefinition
from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.flux.model.flux_vae.vae import VAE
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil

# Reference system prompt for the pure text-to-image path (SYSTEM_PROMPT_4_T2I).
SYSTEM_PROMPT_T2I = (
    "You are a helpful assistant that generates high-quality images based on user instructions. "
    "The instructions are as follows."
)


class BooguImage(nn.Module):
    """Boogu-Image-0.1-Turbo: 4-step DMD text-to-image (no CFG, no scheduler).

    Pipeline: Qwen3-VL instruction encoding -> mixed single/double-stream
    transformer -> FLUX.1 VAE decode. The Turbo path builds its own linspace
    sigma schedule and alternates a predict (``x0`` estimate) and renoise step.

    ``num_inference_steps`` defaults to 4 (Turbo). 4 steps resolves detail well
    up to ~768px; at 1024px and above use ~8 steps, since few-step DMD
    under-resolves fine detail as the token count grows.
    """

    vae: VAE
    transformer: BooguImageTransformer
    text_encoder: BooguTextEncoder

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        model_config: ModelConfig | None = None,
    ) -> None:
        super().__init__()
        BooguInitializer.init(
            model=self,
            quantize=quantize,
            model_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            model_config=model_config or ModelConfig.boogu_image_turbo(),
        )

    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int = 4,
        height: int = 1024,
        width: int = 1024,
        conditioning_sigma: float = 0.001,
    ) -> GeneratedImage:
        config = Config(
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance=1.0,
        )

        # 1. Encode the instruction with the Qwen3-VL chat template.
        instruction_embeds = self._encode_instruction(prompt)

        # 2. Pure-noise latents at the VAE-downscaled resolution.
        latent_h, latent_w = config.height // 8, config.width // 8
        key = mx.random.key(seed)
        key, subkey = mx.random.split(key)
        latents = mx.random.normal((1, 16, latent_h, latent_w), key=subkey)

        # 3. DMD few-step schedule: sigma is the signal level (1.0 = clean).
        sigmas = mx.linspace(conditioning_sigma, 1.0, num_inference_steps + 1)[:-1]

        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)
        for i in config.time_steps:
            try:
                sigma = sigmas[i]
                pred = self.transformer(latents, mx.broadcast_to(sigma, (1,)), instruction_embeds)
                latents = latents + (1.0 - sigma) * pred
                if i < num_inference_steps - 1:
                    next_sigma = sigmas[i + 1]
                    key, subkey = mx.random.split(key)
                    noise = mx.random.normal(latents.shape, key=subkey)
                    latents = (1.0 - next_sigma) * noise + next_sigma * latents
                ctx.in_loop(i, latents)
                mx.eval(latents)
            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(i, latents)
                raise StopImageGenerationException(f"Stopping image generation at step {i + 1}/{num_inference_steps}")
        ctx.after_loop(latents)

        # 4. Decode (the FLUX VAE applies the scaling/shift internally).
        decoded = self.vae.decode(latents)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            generation_time=config.time_steps.format_dict["elapsed"],
        )

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=BooguWeightDefinition,
        )

    def _encode_instruction(self, prompt: str) -> mx.array:
        """Tokenize via the Qwen3-VL chat template and return instruction features.

        Mirrors the reference ``_get_instruction_feature_embeds`` T2I path: a
        ``[system, user]`` message pair, ``apply_chat_template`` without a
        generation prompt, then the encoder's ``last_hidden_state``.
        """
        tokenizer = self.tokenizers["qwen3vl"].tokenizer
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_T2I}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        encoded = tokenizer.apply_chat_template(
            [messages],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors=None,
        )
        input_ids = mx.array(encoded["input_ids"])
        attention_mask = mx.array(encoded["attention_mask"])
        return self.text_encoder.get_instruction_features(input_ids, attention_mask)
