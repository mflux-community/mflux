import gc
import time

import mlx.core as mx
from mlx import nn
from PIL import Image
from tqdm import tqdm

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.ming_image.latent_creator.ming_latent_creator import MingLatentCreator
from mflux.models.ming_image.ming_image_initializer import MingImageInitializer
from mflux.models.ming_image.model.ming_text_encoder.ling_moe_encoder import LingMoeEncoder
from mflux.models.ming_image.model.ming_text_encoder.ming_condition_encoder import (
    PROMPT_TEMPLATE,
    MingConditionEncoder,
    MingHeads,
)
from mflux.models.ming_image.model.ming_text_encoder.ming_connector import MingConnector
from mflux.models.ming_image.model.ming_transformer.ming_transformer import MingTransformer
from mflux.models.ming_image.model.ming_vae.ming_vae import MingVAE
from mflux.models.ming_image.weights.ming_image_weight_definition import MingImageWeightDefinition
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.image_util import ImageUtil

SIGMA_SHIFT = 6.0


class MingImage(nn.Module):
    vae: MingVAE
    text_encoder: LingMoeEncoder | None
    connector: MingConnector | None
    heads: MingHeads | None
    transformer: MingTransformer

    def __init__(
        self,
        quantize: int | None = None,
        text_encoder_quantize: int | None = None,
        model_path: str | None = None,
        model_config: ModelConfig | None = None,
        low_ram: bool = False,
    ):
        super().__init__()
        MingImageInitializer.init(
            model=self,
            model_config=model_config or ModelConfig.ming_image_design(),
            quantize=quantize,
            text_encoder_quantize=text_encoder_quantize,
            model_path=model_path,
        )
        # With low_ram the ~16B-parameter text encoder is released after the first prompt is
        # encoded, so the DiT denoises without it resident. Later prompts then need a reload.
        self.low_ram = low_ram
        self._prompt_cache: dict[str, tuple[mx.array, mx.array]] = {}

    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int = 12,
        height: int = 1024,
        width: int = 1024,
        guidance: float | None = 1.0,
    ) -> "GeneratedImage":  # noqa: F821
        guidance = 1.0 if guidance is None else float(guidance)
        config = Config(
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance=guidance,
        )
        start = time.time()
        cap_feats, cap_feats_2 = self.encode_prompt(prompt)

        latents = MingLatentCreator.create_noise(seed=seed, height=config.height, width=config.width)
        sigmas = MingImage.sigmas(num_inference_steps)
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)
        predict = self._predict(self.transformer)

        steps = tqdm(range(num_inference_steps))
        for t in steps:
            try:
                sigma, sigma_next = sigmas[t], sigmas[t + 1]
                # The official schedule ends on a sigma = 0 step whose update is dt = 0; skip its DiT call.
                if sigma.item() > 0:
                    velocity = predict(latents, 1.0 - sigma.reshape(1), cap_feats, cap_feats_2, guidance)
                    latents = latents + (sigma_next - sigma) * velocity.astype(mx.float32)
                ctx.in_loop(t, latents)
                mx.eval(latents)
            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                raise StopImageGenerationException(f"Stopping image generation at step {t + 1}/{num_inference_steps}")
        ctx.after_loop(latents)

        decoded = VAEUtil.decode(vae=self.vae, latent=latents.astype(ModelConfig.precision), tiling_config=self.tiling_config)  # fmt: off
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            generation_time=time.time() - start,
        )

    def encode_prompt(self, prompt: str) -> tuple[mx.array, mx.array]:
        if prompt in self._prompt_cache:
            return self._prompt_cache[prompt]
        if self.text_encoder is None:
            raise RuntimeError(
                "The text encoder was released (low_ram) after the first prompt; create a new MingImage "
                "to encode a different prompt."
            )
        tokenizer = self.tokenizers["ming"].tokenizer
        prompt_ids = tokenizer(PROMPT_TEMPLATE.format(prompt=prompt), add_special_tokens=False)["input_ids"]
        cap_feats, cap_feats_2 = MingConditionEncoder.encode(
            prompt_ids=prompt_ids,
            text_encoder=self.text_encoder,
            connector=self.connector,
            heads=self.heads,
        )
        mx.eval(cap_feats, cap_feats_2)
        self._prompt_cache[prompt] = (cap_feats, cap_feats_2)
        if self.low_ram:
            self.release_text_encoder()
        return cap_feats, cap_feats_2

    def release_text_encoder(self) -> None:
        self.text_encoder = None
        self.connector = None
        self.heads = None
        gc.collect()
        mx.clear_cache()

    @staticmethod
    def sigmas(num_inference_steps: int, shift: float = SIGMA_SHIFT) -> mx.array:
        # FlowMatchEulerDiscreteScheduler as the official pipeline configures it: sigma_min forced
        # to 0 and the checkpoint's static shift of 6 (its use_dynamic_shifting override never
        # reaches the frozen config), plus the terminal zero.
        s = mx.linspace(1.0, 0.0, num_inference_steps, dtype=mx.float32)
        s = shift * s / (1.0 + (shift - 1.0) * s)
        return mx.concatenate([s, mx.zeros((1,), dtype=mx.float32)])

    @staticmethod
    def _predict(transformer: MingTransformer):
        def predict(latents, timestep, cap_feats, cap_feats_2, guidance):
            x = latents[0][:, None]  # (C, 1, H, W)
            out = transformer(x, timestep, cap_feats, cap_feats_2)
            if guidance > 1.0:
                # The official negative condition is both streams zeroed.
                uncond = transformer(x, timestep, mx.zeros_like(cap_feats), mx.zeros_like(cap_feats_2))
                out = out + guidance * (out - uncond)
            return out[:, 0][None]

        return predict

    def save_model(self, base_path: str) -> None:
        if self.text_encoder is None:
            raise RuntimeError("Cannot save after the text encoder was released (low_ram).")
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=MingImageWeightDefinition,
        )

    @staticmethod
    def to_rgb(image: Image.Image, background: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
        if image.mode != "RGBA":
            return image.convert("RGB")
        canvas = Image.new("RGB", image.size, background)
        canvas.paste(image, mask=image.split()[3])
        return canvas
