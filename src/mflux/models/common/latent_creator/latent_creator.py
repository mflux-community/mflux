from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

import mlx.core as mx
from mlx import nn

from mflux.models.common.vae.vae_util import VAEUtil
from mflux.utils.image_util import ImageUtil

if TYPE_CHECKING:
    from mflux.models.common.vae.tiling_config import TilingConfig
    from mflux.models.fibo.latent_creator.fibo_latent_creator import FiboLatentCreator
    from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
    from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator
    from mflux.models.qwen21.latent_creator.qwen21_latent_creator import Qwen21LatentCreator
    from mflux.models.z_image.latent_creator.z_image_latent_creator import ZImageLatentCreator

    LatentCreatorType: TypeAlias = type[
        FiboLatentCreator | FluxLatentCreator | QwenLatentCreator | Qwen21LatentCreator | ZImageLatentCreator
    ]


class Img2Img:
    def __init__(
        self,
        vae: nn.Module,
        latent_creator: "LatentCreatorType",
        sigmas: mx.array,
        init_time_step: int,
        image_path: str | Path | None,
        tiling_config: "TilingConfig" | None = None,
    ):
        self.vae = vae
        self.sigmas = sigmas
        self.init_time_step = init_time_step
        self.image_path = image_path
        self.latent_creator = latent_creator
        self.tiling_config = tiling_config


class LatentCreator:
    @staticmethod
    def create_for_txt2img_or_img2img(
        seed: int,
        height: int,
        width: int,
        img2img: Img2Img,
    ) -> mx.array:
        latent_creator = img2img.latent_creator

        if img2img.image_path is None:
            # txt2img: just create noise
            return latent_creator.create_noise(seed, height, width)
        else:
            # img2img: blend encoded image with noise
            pure_noise = latent_creator.create_noise(seed, height, width)
            encoded = LatentCreator.encode_image(
                width=width,
                height=height,
                vae=img2img.vae,
                image_path=img2img.image_path,
                tiling_config=img2img.tiling_config,
            )
            latents = latent_creator.pack_latents(encoded, height, width)
            sigma = img2img.sigmas[img2img.init_time_step]
            return LatentCreator.add_noise_by_interpolation(clean=latents, noise=pure_noise, sigma=sigma)

    @staticmethod
    def encode_image(
        vae: nn.Module,
        image_path: str | Path,
        height: int,
        width: int,
        tiling_config: "TilingConfig" | None = None,
    ) -> mx.array:
        scaled_user_image = ImageUtil.scale_to_dimensions(
            image=ImageUtil.load_image(image_path).convert("RGB"),
            target_width=width,
            target_height=height,
        )
        image_array = ImageUtil.to_array(scaled_user_image)
        return VAEUtil.encode(vae=vae, image=image_array, tiling_config=tiling_config)

    @staticmethod
    def add_noise_by_interpolation(clean: mx.array, noise: mx.array, sigma: float) -> mx.array:
        return (1 - sigma) * clean + sigma * noise
