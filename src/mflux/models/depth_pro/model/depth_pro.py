import os
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from mflux.models.depth_pro.depth_pro_initializer import DepthProInitializer
from mflux.models.depth_pro.model.depth_pro_model import DepthProModel
from mflux.models.depth_pro.model.depth_pro_util import DepthProUtil
from mflux.utils.exif_orientation import open_oriented
from mflux.utils.image_util import ImageUtil


@dataclass
class DepthResult:
    depth_image: Image.Image
    depth_array: mx.array
    min_depth: float
    max_depth: float


class DepthPro:
    def __init__(self, quantize: int | None = None):
        self._depth_pro_model = DepthProModel()
        DepthProInitializer.init(self._depth_pro_model, quantize=quantize)

    def create_depth_map(self, image_path: str | Path) -> DepthResult:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        input_array, height, width = DepthPro._pre_process(image_path)
        x0, x1, x2 = DepthPro._create_patches(input_array)
        depth = self._depth_pro_model(x0, x1, x2)
        return DepthPro._post_process(depth, height=height, width=width)

    @staticmethod
    def _pre_process(image_path: str | Path) -> tuple[mx.array, int, int]:
        image = open_oriented(image_path).convert("RGB")
        input_array = ImageUtil.preprocess_for_depth_pro(image)
        input_array = DepthPro._resize(input_array)
        return input_array, image.height, image.width

    @staticmethod
    def _create_patches(input_array: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        # 1. Create the image pyramid
        x0 = input_array
        x1 = DepthProUtil.interpolate(x=input_array, scale_factor=0.5)
        x2 = DepthProUtil.interpolate(x=input_array, scale_factor=0.25)

        # 2: Split to create batched overlapped mini-images at the backbone (BeiT/ViT/Dino) resolution.
        x0_patches = DepthProUtil.split(x0, overlap_ratio=0.25)
        x1_patches = DepthProUtil.split(x1, overlap_ratio=0.5)
        x2_patches = x2

        return x0_patches, x1_patches, x2_patches

    @staticmethod
    def _post_process(depth: mx.array, height: int, width: int) -> DepthResult:
        depth_min = mx.min(depth)
        depth_max = mx.max(depth)
        normalized_depth = (depth - depth_min) / (depth_max - depth_min)
        depth_np = np.asarray(normalized_depth.squeeze())
        depth_image = Image.fromarray((depth_np * 255).astype(np.uint8))
        depth_image = depth_image.resize((width, height))
        depth_np = np.array(depth_image) / 255.0
        return DepthResult(
            depth_image=depth_image,
            depth_array=depth,
            min_depth=depth_min.item(),
            max_depth=depth_max.item(),
        )

    @staticmethod
    def _resize(x: mx.array) -> mx.array:
        x = mx.expand_dims(x, 0)
        x = DepthProUtil.interpolate(x=x, size=(1536, 1536))
        return x
