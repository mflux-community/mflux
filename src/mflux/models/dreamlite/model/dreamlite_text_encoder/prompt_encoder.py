import mlx.core as mx
from PIL import Image

from mflux.models.common_models.qwen3_vl.qwen3_vl_processor import Qwen3VLProcessor
from mflux.models.dreamlite.model.dreamlite_text_encoder.dreamlite_text_encoder import DreamLiteTextEncoder


class DreamLitePromptEncoder:
    # The templates' system parts are dropped from the encoder output: 34 tokens when generating, 64 when editing
    GENERATE_TEMPLATE = (
        "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
        "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    )
    GENERATE_DROP = 34
    EDIT_TEMPLATE = (
        "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, "
        "texture, objects, background), then explain how the user's text instruction should alter "
        "or modify the image. Generate a new image that meets the user's requirements while maintaining "
        "consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
    )
    EDIT_DROP = 64
    GENERATE_PREFIX = "[Generate]: "
    EDIT_PREFIX = (
        "[Edit]: A diptych with two side-by-side images of the same scene. "
        "Compared to the right side, the left one has "
    )

    @staticmethod
    def generate_prompt(prompt: str) -> str:
        return DreamLitePromptEncoder.GENERATE_PREFIX + prompt

    @staticmethod
    def edit_prompt(prompt: str) -> str:
        return DreamLitePromptEncoder.EDIT_PREFIX + prompt

    @staticmethod
    def encode_generate(text: str, processor: Qwen3VLProcessor, text_encoder: DreamLiteTextEncoder) -> mx.array:
        inputs = processor([DreamLitePromptEncoder.GENERATE_TEMPLATE.format(text)], return_tensors="np")
        hidden = text_encoder(mx.array(inputs["input_ids"]))
        return hidden[:, DreamLitePromptEncoder.GENERATE_DROP :]

    @staticmethod
    def encode_edit(
        text: str,
        image: Image.Image,
        vision_size: int,
        processor: Qwen3VLProcessor,
        text_encoder: DreamLiteTextEncoder,
    ) -> mx.array:
        # The encoder sees a small square copy of the source image: 512 px on the base model, 256 px on mobile
        vision_image = image.resize((vision_size, vision_size), Image.Resampling.LANCZOS)
        inputs = processor(
            [DreamLitePromptEncoder.EDIT_TEMPLATE.format(text)], images=[vision_image], return_tensors="np"
        )
        hidden = text_encoder(
            mx.array(inputs["input_ids"]),
            mx.array(inputs["pixel_values"]),
            mx.array(inputs["image_grid_thw"]),
        )
        return hidden[:, DreamLitePromptEncoder.EDIT_DROP :]
