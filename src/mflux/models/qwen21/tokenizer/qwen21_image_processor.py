import math
from typing import Optional, Union

import mlx.core as mx
import numpy as np
from PIL import Image


class Qwen21ImageProcessor:
    # Qwen3-VL vision preprocessing for Qwen-Image-2.1's text encoder: resize, normalize
    # (mean/std 0.5), patchify into 16x16 patches with temporal duplication and 2x2
    # merge layout, matching processor/preprocessor_config.json of the checkpoint.

    def __init__(
        self,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        min_pixels: int = 256 * 256,
        max_pixels: int = 4096 * 4096,
        image_mean: Optional[list[float]] = None,
        image_std: Optional[list[float]] = None,
    ):
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.image_mean = image_mean if image_mean is not None else [0.5, 0.5, 0.5]
        self.image_std = image_std if image_std is not None else [0.5, 0.5, 0.5]

    def smart_resize(
        self,
        height: int,
        width: int,
        factor: Optional[int] = None,
    ) -> tuple[int, int]:
        factor = factor or self.patch_size * self.merge_size
        if max(height, width) / min(height, width) > 200:
            raise ValueError(
                f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
            )
        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor
        if h_bar * w_bar > self.max_pixels:
            beta = math.sqrt((height * width) / self.max_pixels)
            h_bar = max(factor, math.floor(height / beta / factor) * factor)
            w_bar = max(factor, math.floor(width / beta / factor) * factor)
        elif h_bar * w_bar < self.min_pixels:
            beta = math.sqrt(self.min_pixels / (height * width))
            h_bar = math.ceil(height * beta / factor) * factor
            w_bar = math.ceil(width * beta / factor) * factor
        return h_bar, w_bar

    def _preprocess(
        self,
        image: Image.Image,
        resized_height: Optional[int] = None,
        resized_width: Optional[int] = None,
    ) -> tuple[np.ndarray, tuple[int, int, int]]:
        if image.mode != "RGB":
            image = image.convert("RGB")

        height, width = image.size[1], image.size[0]
        if resized_height is None or resized_width is None:
            resized_height, resized_width = self.smart_resize(height, width)
        if (height, width) != (resized_height, resized_width):
            image = image.resize((resized_width, resized_height), Image.BICUBIC)

        image_np = np.array(image).astype(np.float32) / 255.0
        image_np = (image_np - np.array(self.image_mean, dtype=np.float32)) / np.array(self.image_std, dtype=np.float32)
        image_np = image_np.transpose(2, 0, 1)
        patches = image_np[np.newaxis]  # (1, C, H, W)

        if patches.shape[0] % self.temporal_patch_size != 0:
            repeats = np.repeat(
                patches[-1][np.newaxis],
                self.temporal_patch_size - (patches.shape[0] % self.temporal_patch_size),
                axis=0,
            )
            patches = np.concatenate([patches, repeats], axis=0)

        channel = patches.shape[1]
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h = resized_height // self.patch_size
        grid_w = resized_width // self.patch_size

        patches = patches.reshape(
            grid_t,
            self.temporal_patch_size,
            channel,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w,
            channel * self.temporal_patch_size * self.patch_size * self.patch_size,
        )
        return flatten_patches, (grid_t, grid_h, grid_w)

    def preprocess(
        self,
        images: Union[Image.Image, list[Image.Image]],
        resized_height: Optional[int] = None,
        resized_width: Optional[int] = None,
    ) -> tuple[mx.array, mx.array]:
        if isinstance(images, Image.Image):
            images = [images]
        patches_list, grids = [], []
        for image in images:
            patches, grid = self._preprocess(image, resized_height, resized_width)
            patches_list.append(patches)
            grids.append(grid)
        pixel_values = mx.array(np.concatenate(patches_list, axis=0))
        grid_thw = mx.array(np.array(grids, dtype=np.int32))
        return pixel_values, grid_thw
