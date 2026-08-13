import math
from pathlib import Path
from typing import Union

import mlx.core as mx
import numpy as np
from PIL import Image

from mflux.models.qwen.tokenizer.qwen_vision_language_processor import QwenVisionLanguageProcessor
from mflux.utils.exif_orientation import open_oriented


class QwenVisionLanguageTokenizer:
    def __init__(
        self,
        processor: QwenVisionLanguageProcessor,
        max_length: int = 1024,
        use_picture_prefix: bool = True,
    ):
        self.processor = processor
        self.max_length = max_length
        self.use_picture_prefix = use_picture_prefix

        if use_picture_prefix:
            self.edit_template = (
                "<|im_start|>system\n"
                "Describe the key features of the input image (color, shape, size, texture, objects, background), "
                "then explain how the user's text instruction should alter or modify the image. "
                "Generate a new image that meets the user's requirements while maintaining consistency "
                "with the original input where appropriate.<|im_end|>\n"
                "<|im_start|>user\n"
                "{}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        else:
            self.edit_template = (
                "<|im_start|>system\n"
                "Describe the key features of the input image (color, shape, size, texture, objects, background), "
                "then explain how the user's text instruction should alter or modify the image. "
                "Generate a new image that meets the user's requirements while maintaining consistency "
                "with the original input where appropriate.<|im_end|>\n"
                "<|im_start|>user\n"
                "<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        self.edit_template_start_idx = 64

    def tokenize_with_image(
        self,
        prompt: str,
        image: Union[Image.Image, np.ndarray, str, list],
        vl_width: int | None = None,
        vl_height: int | None = None,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        # Normalize image to list format
        if not isinstance(image, list):
            images = [image]
        else:
            images = image

        # Format prompt based on tokenizer mode
        if self.use_picture_prefix:
            # Edit format: Add "Picture N:" prefix for each image
            # For multiple images: "Picture 1: ... Picture 2: ... Picture N: ..."
            img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
            base_img_prompt = ""
            for i in range(len(images)):
                base_img_prompt += img_prompt_template.format(i + 1)
            formatted_text = self.edit_template.format(base_img_prompt + prompt)
        else:
            # Regular Edit format: Vision tokens already in template
            # Just format with user prompt directly
            formatted_text = self.edit_template.format(prompt)

        # Process images: convert to PIL Images and resize to CONDITION_IMAGE_SIZE
        CONDITION_IMAGE_SIZE = 384 * 384

        processed_images = []
        for img in images:
            # Convert to PIL Image
            if isinstance(img, (str, Path)):
                img = open_oriented(img).convert("RGB")
            elif isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            elif not isinstance(img, Image.Image):
                raise ValueError(f"Unsupported image type: {type(img)}")

            # Resize to CONDITION_IMAGE_SIZE (384×384) maintaining aspect ratio
            img_w, img_h = img.size
            ratio = img_w / img_h
            condition_width = math.sqrt(CONDITION_IMAGE_SIZE * ratio)
            condition_height = condition_width / ratio
            condition_width = round(condition_width / 32) * 32
            condition_height = round(condition_height / 32) * 32

            img = img.resize((int(condition_width), int(condition_height)), Image.BICUBIC)
            processed_images.append(img)

        # Use our MLX processor for both text and images
        model_inputs = self.processor(
            text=[formatted_text],
            images=processed_images,
            padding=True,
            return_tensors=None,  # Return numpy/MLX arrays, not PyTorch
        )

        grid_thw = model_inputs["image_grid_thw"][0]
        factor = 14 * 2
        self._vl_image_width = int(grid_thw[2]) * factor
        self._vl_image_height = int(grid_thw[1]) * factor

        # Convert to MLX arrays if needed
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        pixel_values = mx.array(model_inputs["pixel_values"])
        image_grid_thw = mx.array(model_inputs["image_grid_thw"])

        return input_ids, attention_mask, pixel_values, image_grid_thw

    def tokenize_text_only(self, prompt: str) -> tuple[mx.array, mx.array]:
        # Use the regular text-only template
        text_template = (
            "<|im_start|>system\n"
            "Describe the image by detailing the color, shape, size, texture, quantity, text, "
            "spatial relationships of the objects and background:<|im_end|>\n"
            "<|im_start|>user\n{}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        formatted_text = text_template.format(prompt)
        tokens = self.processor.tokenizer(
            formatted_text,
            max_length=self.max_length + 34,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # Convert PyTorch tensors to MLX arrays
        input_ids = mx.array(tokens["input_ids"].numpy())
        attention_mask = mx.array(tokens["attention_mask"].numpy())

        return input_ids, attention_mask
