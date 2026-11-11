import mlx.core as mx
from mlx import nn

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.sensenova.latent_creator.sensenova_latent_creator import SenseNovaLatentCreator
from mflux.models.sensenova.model.sensenova_scheduler.sensenova_scheduler import SenseNovaScheduler
from mflux.models.sensenova.model.sensenova_text_encoder.prompt_encoder import SenseNovaPromptEncoder
from mflux.models.sensenova.model.sensenova_text_encoder.sensenova_text_encoder import SenseNovaTextEncoder
from mflux.models.sensenova.model.sensenova_transformer.sensenova_transformer import SenseNovaTransformer
from mflux.models.sensenova.sensenova_initializer import SenseNovaInitializer
from mflux.models.sensenova.weights.sensenova_weight_definition import SenseNovaWeightDefinition
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil

Prefix = tuple[list[tuple[mx.array, mx.array]], int]


class SenseNova(nn.Module):
    transformer: SenseNovaTransformer
    text_encoder: SenseNovaTextEncoder

    DEFAULT_STEPS = 50
    DEFAULT_GUIDANCE = 4.0
    SHIFT = 3.0
    # Unbaked keeps the full LoRA update. Baking into bfloat16 keeps 82% of it, as the reference's own merge does
    # (36 dB to the reference, 26 dB to unbaked); with -q the bake folds in float32 instead (34 dB to unbaked)
    DEFAULT_BAKE_LORA = False
    UNCOND_KEY = None

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        model_config: ModelConfig | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        bake_lora: bool = DEFAULT_BAKE_LORA,
    ):
        super().__init__()
        SenseNovaInitializer.init(
            model=self,
            model_config=model_config or ModelConfig.sensenova_u15(),
            quantize=quantize,
            model_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            bake_lora=bake_lora,
        )

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=SenseNovaWeightDefinition,
        )

    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int | None = None,
        height: int = 1024,
        width: int = 1024,
        guidance: float | None = None,
        negative_prompt: str | None = None,
    ) -> GeneratedImage:
        if negative_prompt:
            raise ValueError("SenseNova-U1.5 has no negative prompt: its unconditional branch is a fixed empty query.")
        SenseNovaLatentCreator.check_size(height, width)
        config = Config(
            model_config=self.model_config,
            num_inference_steps=num_inference_steps or SenseNova.DEFAULT_STEPS,
            height=height,
            width=width,
            guidance=SenseNova.DEFAULT_GUIDANCE if guidance is None else float(guidance),
        )

        # 1. Encode the prompt into the per-layer prefix KV (and the empty query for CFG)
        use_cfg = config.guidance > 1.0
        cond = self._encode_prompt(prompt)
        uncond = self._encode_uncond() if use_cfg else None

        # 2. Noise
        latents = SenseNovaLatentCreator.create_noise(seed, config.height, config.width)
        noise_scale = SenseNovaLatentCreator.noise_scale(config.height, config.width)

        # 3. Denoising loop; the integrator state stays float32, the network runs in the model precision.
        scheduler = SenseNovaScheduler(config.num_inference_steps, SenseNova.SHIFT)
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)
        for t in config.time_steps:
            try:
                model_time = scheduler.model_time(t)
                v_cond = self._velocity(latents, model_time, noise_scale, cond)
                if use_cfg:
                    v_uncond = self._velocity(latents, model_time, noise_scale, uncond)
                    v_cond = v_uncond + config.guidance * (v_cond - v_uncond)
                latents = scheduler.step(latents, v_cond, t)
                ctx.in_loop(t, latents)
                mx.eval(latents)
            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                )

        ctx.after_loop(latents)

        # 4. No VAE: the sample is already an RGB image in [-1, 1].
        decoded = SenseNovaLatentCreator.unpack_latents(mx.clip(latents, -1, 1), config.height, config.width)
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

    def _encode_prompt(self, prompt: str) -> Prefix:
        if prompt not in self.prompt_cache:
            self.prompt_cache[prompt] = SenseNovaPromptEncoder.encode_prompt(
                prompt=prompt,
                tokenizer=self.tokenizers["sensenova"],
                text_encoder=self.text_encoder,
            )
            mx.eval(self.prompt_cache[prompt][0])
        return self.prompt_cache[prompt]

    def _encode_uncond(self) -> Prefix:
        if SenseNova.UNCOND_KEY not in self.prompt_cache:
            self.prompt_cache[SenseNova.UNCOND_KEY] = SenseNovaPromptEncoder.encode_negative_prompt(
                tokenizer=self.tokenizers["sensenova"],
                text_encoder=self.text_encoder,
            )
            mx.eval(self.prompt_cache[SenseNova.UNCOND_KEY][0])
        return self.prompt_cache[SenseNova.UNCOND_KEY]

    def _velocity(self, latents: mx.array, model_time: mx.array, noise_scale: float, prefix: Prefix) -> mx.array:
        kv, text_len = prefix
        x_pred = self.transformer(latents.astype(ModelConfig.precision), model_time, noise_scale, kv, text_len)
        return SenseNovaTransformer.velocity(x_pred.astype(mx.float32), latents, model_time)
