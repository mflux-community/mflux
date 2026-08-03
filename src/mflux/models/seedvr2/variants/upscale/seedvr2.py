from pathlib import Path

import mlx.core as mx
from mlx import nn

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.seedvr2.latent_creator.seedvr2_latent_creator import SeedVR2LatentCreator
from mflux.models.seedvr2.model.seedvr2_text_encoder.text_embeddings import SeedVR2TextEmbeddings
from mflux.models.seedvr2.model.seedvr2_transformer.transformer import SeedVR2Transformer
from mflux.models.seedvr2.model.seedvr2_vae.vae import SeedVR2VAE
from mflux.models.seedvr2.seedvr2_initializer import SeedVR2Initializer
from mflux.models.seedvr2.variants.upscale.seedvr2_util import SeedVR2Util
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.gpu_stream import with_gpu_stream
from mflux.utils.image_util import ImageUtil
from mflux.utils.metadata_reader import MetadataReader
from mflux.utils.scale_factor import ScaleFactor


class SeedVR2(nn.Module):
    vae: SeedVR2VAE
    transformer: SeedVR2Transformer

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        model_config: ModelConfig = ModelConfig.seedvr2_3b(),
    ):
        super().__init__()
        SeedVR2Initializer.init(
            model=self,
            quantize=quantize,
            model_path=model_path,
            model_config=model_config,
        )

    @with_gpu_stream
    def generate_image(
        self,
        seed: int,
        image_path: str | Path,
        resolution: int | ScaleFactor,
        softness: float = 0.0,
    ) -> GeneratedImage:
        # 0. Process and scale the input image
        processed_image, true_height, true_width = SeedVR2Util.preprocess_image(
            image_path=image_path,
            resolution=resolution,
            softness=softness,
        )

        # 1. Create a new config based on the model type and input parameters
        config = Config(
            width=true_width,
            height=true_height,
            guidance=1.0,
            num_inference_steps=1,
            image_path=image_path,
            scheduler="seedvr2_euler",
            model_config=self.model_config,
        )

        # 2. Create the initial latents and conditioning
        initial_latent = VAEUtil.encode(vae=self.vae, image=processed_image, tiling_config=self.tiling_config)
        static_condition = SeedVR2LatentCreator.create_condition(encoded_latent=initial_latent)
        latents = SeedVR2LatentCreator.create_noise_latents(seed=seed, height=initial_latent.shape[-2], width=initial_latent.shape[-1])  # fmt: off

        # 3. Get the pre-computed text embeddings
        txt_pos = SeedVR2TextEmbeddings.load_positive()

        # 4. Create callback context and call before_loop
        ctx = self.callbacks.start(seed=seed, prompt="", config=config)
        ctx.before_loop(latents)

        for t in config.time_steps:
            model_input = mx.concatenate([latents, static_condition], axis=1)

            # 5.t Predict the noise
            noise = self.transformer(
                txt=txt_pos,
                vid=model_input,
                timestep=config.scheduler.timesteps[t],
            )

            # 6.t Take one denoise step
            latents = config.scheduler.step(noise=noise, timestep=t, latents=latents)

            # 7.t Call subscribers in-loop
            ctx.in_loop(t, latents)

            mx.eval(latents)

        # 8. Call subscribers after loop
        ctx.after_loop(latents)

        # 9. Decode the latents and return the image
        decoded = VAEUtil.decode(vae=self.vae, latent=latents, tiling_config=self.tiling_config)
        decoded = decoded[:, :, :true_height, :true_width]
        style = processed_image[:, :, :true_height, :true_width]
        decoded = SeedVR2Util.apply_color_correction(decoded, style)

        # 10. Read metadata from the original image if available
        init_metadata = MetadataReader.read_all_metadata(image_path) if image_path else None

        return ImageUtil.to_image(
            seed=seed,
            prompt="",
            config=config,
            quantization=self.bits,
            decoded_latents=decoded,
            generation_time=config.time_steps.format_dict["elapsed"],
            init_metadata=init_metadata,
        )
