import mlx.core as mx
from mlx import nn

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.step_cache.step_cache import StepCache
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.common_models.qwen3_5 import Qwen35TextModel
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.models.moss_image.latent_creator.moss_image_latent_creator import MOSSImageLatentCreator
from mflux.models.moss_image.model.moss_image_text_encoder.moss_image_prompt_encoder import MOSSImagePromptEncoder
from mflux.models.moss_image.model.moss_image_transformer.moss_image_transformer import MOSSImageTransformer
from mflux.models.moss_image.moss_image_initializer import MOSSImageInitializer
from mflux.models.moss_image.weights.moss_image_weight_definition import MOSSImageWeightDefinition
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil


class MOSSImage(nn.Module):
    vae: Flux2VAE
    transformer: MOSSImageTransformer
    text_encoder: Qwen35TextModel

    # The reference sampling settings (model card of OpenMOSS-Team/MOSS-Image-SFT).
    DEFAULT_STEPS = 50
    DEFAULT_GUIDANCE = 3.0
    # The negative CFG branch skips this zero-based transformer layer.
    DEFAULT_UNCOND_SKIP_LAYERS = (22,)
    # LoRAs stay separate layers. Most of the update of the step-distilled LoRAs is smaller than half
    # a bfloat16 step of the weights it lands on, so a merge into the weights loses half of it.
    DEFAULT_BAKE_LORA = False

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        model_config: ModelConfig | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        bake_lora: bool = DEFAULT_BAKE_LORA,
        ssd_stream: bool = False,
        stream_lora: bool = False,
    ):
        super().__init__()
        MOSSImageInitializer.init(
            model=self,
            model_config=model_config or ModelConfig.moss_image_sft(),
            quantize=quantize,
            model_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            bake_lora=bake_lora,
            ssd_stream=ssd_stream,
            stream_lora=stream_lora,
        )
        self.bake_lora = bake_lora

    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int = DEFAULT_STEPS,
        height: int = 1024,
        width: int = 1024,
        guidance: float = DEFAULT_GUIDANCE,
        negative_prompt: str | None = None,
        uncond_skip_layers: tuple[int, ...] = DEFAULT_UNCOND_SKIP_LAYERS,
        step_cache_ratio: float | None = None,
    ) -> GeneratedImage:
        skip_layers = self._validate_skip_layers(uncond_skip_layers)
        latents = MOSSImageLatentCreator.create_noise(seed=seed, height=height, width=width)
        config = Config(
            width=width,
            height=height,
            guidance=guidance,
            scheduler="moss_image",
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
        )
        do_cfg = config.guidance != 1.0
        positive = self._encode_prompt(prompt)
        negative = self._encode_prompt(negative_prompt or "") if do_cfg else None
        timesteps = config.scheduler.timesteps.tolist()
        step_cache = StepCache.for_run(config, ratio=step_cache_ratio, signal_fn=self.transformer.time_signal)

        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)
        for t in config.time_steps:
            try:
                noise = step_cache.reuse(t)
                if noise is None:
                    noise = self._predict(latents, timesteps[t], positive, negative, config.guidance, skip_layers)
                    step_cache.store(noise)
                latents = config.scheduler.step(noise=noise, timestep=t, latents=latents)
                ctx.in_loop(t, latents)
                mx.eval(latents)
            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                )
        ctx.after_loop(latents)

        decoded = self.vae.decode_packed_latents(
            latents.astype(ModelConfig.precision), tiling_config=self.tiling_config
        )
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            negative_prompt=negative_prompt,
            generation_time=config.time_steps.format_dict["elapsed"],
            generation_parameters={
                **step_cache.generation_parameters(step_cache_ratio),
                # The layer skip changes the image, so the metadata records it for --config-from-metadata.
                "uncond_skip_layers": sorted(skip_layers),
                # Merged and runtime LoRAs give different images, so a LoRA run records which one it used.
                **({"bake_lora": self.bake_lora} if self.lora_paths else {}),
            },
        )

    def save_model(self, base_path: str) -> None:
        if self.ssd_stream:
            # The streamed layers are placeholders between calls, so a save would write a checkpoint without them.
            raise ValueError("A model loaded with ssd_stream=True cannot be saved. Load it without --ssd-stream.")
        # The weights load lazily. Read them all first: a save over the checkpoint they come from
        # would otherwise truncate a shard before reading it.
        mx.eval(self.parameters())
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=MOSSImageWeightDefinition,
        )

    def _predict(
        self,
        latents: mx.array,
        timestep: float,
        positive: mx.array,
        negative: mx.array | None,
        guidance: float,
        skip_layers: frozenset[int],
    ) -> mx.array:
        # The two CFG branches run as separate passes: their text lengths differ and the
        # negative branch skips layers, so they cannot share one batch.
        cond = self.transformer(latents, timestep, positive).astype(mx.float32)
        if negative is None:
            return cond
        uncond = self.transformer(latents, timestep, negative, skip_layers=skip_layers).astype(mx.float32)
        return uncond + guidance * (cond - uncond)

    def _encode_prompt(self, prompt: str) -> mx.array:
        if prompt not in self.prompt_cache:
            embeds = MOSSImagePromptEncoder.encode(prompt, self.tokenizers["qwen3_5"], self.text_encoder)
            mx.eval(embeds)
            self.prompt_cache[prompt] = embeds
        return self.prompt_cache[prompt]

    def _validate_skip_layers(self, layers: tuple[int, ...]) -> frozenset[int]:
        return MOSSImageTransformer.check_skip_layers(layers, len(self.transformer.layers))
