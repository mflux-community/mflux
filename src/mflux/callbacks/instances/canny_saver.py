import os
from pathlib import Path

import mlx.core as mx
import PIL.Image

from mflux.callbacks.callback import BeforeLoopCallback
from mflux.models.common.config.config import Config
from mflux.utils.image_util import ImageUtil


class CannyImageSaver(BeforeLoopCallback):
    def __init__(self, path: str):
        self.path = Path(path)

    def call_before_loop(
        self,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: Config,
        canny_image: PIL.Image.Image | None = None,
        depth_image: PIL.Image.Image | None = None,
        control_images: list[PIL.Image.Image] | None = None,
    ) -> None:
        base, ext = os.path.splitext(self.path)
        ImageUtil.save_image(
            image=canny_image,
            path=f"{base}_controlnet_canny{ext}",
        )
