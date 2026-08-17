"""Microsoft Lens text-to-image (Turbo): 3.8B dual-stream MMDiT over FLUX.2 latents
with GPT-OSS 20B multi-layer text features.

The three components come from three repositories: the DiT from the Comfy-Org
mirror (the Microsoft originals were withdrawn), the encoder from the mlx-format
community conversion (pre-quantized, self-describing), and the FLUX.2 VAE from
the klein repository already used by the flux2 family.
"""

import math
import time

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.config.config import Config
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition
from mflux.models.lens.model.text_encoder.lens_gpt_oss_encoder import (
    DEFAULT_ENCODER_REPO,
    LensGptOssEncoder,
)
from mflux.models.lens.model.transformer.lens_transformer import LensTransformer
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.image_util import ImageUtil

TURBO_WEIGHTS_PATTERN = "diffusion_models/lens_turbo_bf16.safetensors"
VAE_REPO = "black-forest-labs/FLUX.2-klein-4B"


class LensImage:
    def __init__(
        self,
        model_config: ModelConfig | None = None,
        quantize: int | None = None,
        model_path: str | None = None,
        encoder_path: str | None = None,
    ):
        self.model_config = model_config or ModelConfig.lens_turbo()
        self.bits = quantize
        self.callbacks = CallbackRegistry()

        encoder_root = PathResolution.resolve(
            path=encoder_path or DEFAULT_ENCODER_REPO,
            patterns=["*.safetensors", "*.json"],
        )
        self.text_encoder = LensGptOssEncoder(str(encoder_root))

        dit_root = PathResolution.resolve(
            path=model_path or self.model_config.model_name,
            patterns=[TURBO_WEIGHTS_PATTERN],
        )
        self.transformer = LensTransformer()
        weights = mx.load(str(dit_root / TURBO_WEIGHTS_PATTERN))
        self.transformer.update(tree_unflatten([(k, v.astype(mx.bfloat16)) for k, v in weights.items()]))
        if quantize is not None:
            nn.quantize(self.transformer, bits=quantize, class_predicate=lambda p, m: hasattr(m, "to_quantized"))
        mx.eval(self.transformer.parameters())

        vae_component = [c for c in Flux2KleinWeightDefinition.get_components() if c.name == "vae"][0]
        # Names the safetensors, not the whole vae/ directory: the pattern is also what
        # judges a cached snapshot complete, and "vae/*" is satisfied by a stray config.json.
        vae_weights = WeightLoader.load_single(vae_component, VAE_REPO, file_pattern="vae/*.safetensors")
        self.vae = Flux2VAE()
        WeightApplier.apply_and_quantize_single(vae_weights, self.vae, vae_component, None)
        mx.eval(self.vae.parameters())

    def generate_image(
        self,
        seed: int,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 4,
    ):
        start = time.time()
        config = Config(
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
        )

        features = self.text_encoder.encode(prompt).astype(mx.bfloat16)

        latent_height = height // 16
        latent_width = width // 16
        seq = latent_height * latent_width
        sigmas = self._turbo_sigmas(seq, num_inference_steps)

        mx.random.seed(seed)
        latents = mx.random.normal((1, seq, 128), dtype=mx.float32).astype(mx.bfloat16)

        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)
        for i in config.time_steps:
            try:
                velocity = self.transformer(
                    hidden_states=latents,
                    encoder_layers=features,
                    timestep=mx.array([sigmas[i]]),
                    latent_height=latent_height,
                    latent_width=latent_width,
                )
                latents = latents + (sigmas[i + 1] - sigmas[i]) * velocity
                ctx.in_loop(i, latents)
                mx.eval(latents)
            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(i, latents)
                raise StopImageGenerationException(f"Stopping image generation at step {i + 1}/{num_inference_steps}")
        ctx.after_loop(latents)

        packed = latents.reshape(1, latent_height, latent_width, 128).transpose(0, 3, 1, 2)
        decoded = self.vae.decode_packed_latents(packed.astype(mx.float32))

        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            generation_time=time.time() - start,
        )

    def _turbo_sigmas(self, image_seq_len: int, num_steps: int) -> list[float]:
        # FlowMatchEuler dynamic exponential shift, mu from the image sequence
        # length exactly as the reference scheduler config declares.
        if num_steps < 1:
            raise ValueError(f"num_inference_steps must be >= 1, got {num_steps}")
        mc = self.model_config
        mu = mc.sigma_base_shift + (mc.sigma_max_shift - mc.sigma_base_shift) * (
            image_seq_len - mc.sigma_base_seq_len
        ) / (mc.sigma_max_seq_len - mc.sigma_base_seq_len)
        base = [1.0 - i / num_steps for i in range(num_steps)]
        shifted = [math.exp(mu) / (math.exp(mu) + (1.0 / s - 1.0)) for s in base]
        return [float(s) for s in shifted] + [0.0]
