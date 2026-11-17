from typing import List

import mlx.core as mx

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.tokenizer import LanguageTokenizer
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.common.weights.mapping.weight_transforms import WeightTransforms
from mflux.models.common_models.qwen3_vl.qwen3_vl_multimodal import Qwen3VLMultimodal


class DreamLiteWeightDefinition:
    # The modules carry the checkpoint's own names (diffusers' UNet2DConditionModel and AutoencoderTiny layouts), so
    # no mapping is needed: only the convolution weights move from PyTorch's (out, in, h, w) to MLX's (out, h, w, in).
    # The UNet (0.39B) and the VAE (2.4M) run in float32: it is faster than bfloat16 here, and bfloat16 group norms
    # and separable convolutions move the image away from the reference.
    DENOISER_PRECISION = mx.float32

    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                precision=DreamLiteWeightDefinition.DENOISER_PRECISION,
                weight_transform=DreamLiteWeightDefinition._conv_weight,
            ),
            ComponentDefinition(
                name="unet",
                hf_subdir="unet",
                precision=DreamLiteWeightDefinition.DENOISER_PRECISION,
                weight_transform=DreamLiteWeightDefinition._conv_weight,
            ),
            ComponentDefinition(
                name="text_encoder",
                hf_subdir="text_encoder",
                precision=ModelConfig.precision,
                key_transform=DreamLiteWeightDefinition._text_encoder_key,
                weight_transform=Qwen3VLMultimodal.vision_weight,
            ),
        ]

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return [
            TokenizerDefinition(
                name="qwen3_vl",
                hf_subdir="processor",
                tokenizer_class="AutoTokenizer",
                encoder_class=LanguageTokenizer,
                download_patterns=[
                    "processor/tokenizer*",
                    "processor/chat_template.jinja",
                ],  # fmt: skip
            ),
        ]

    @staticmethod
    def get_download_patterns() -> List[str]:
        return [
            "vae/*.safetensors",
            "vae/*.json",
            "unet/*.safetensors",
            "unet/*.json",
            "text_encoder/*.safetensors",
            "text_encoder/*.json",
            "processor/*",
        ]

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        # The UNet's timestep and size embedders are tiny and condition every layer, so they stay dense
        if path.startswith(("time_embedding", "add_embedding")):
            return False
        return hasattr(module, "to_quantized")

    @staticmethod
    def _conv_weight(key: str, weight: mx.array) -> mx.array:
        return WeightTransforms.transpose_conv2d_weight(weight)

    @staticmethod
    def _text_encoder_key(key: str) -> str | None:
        # model.language_model.* and model.visual.* load as language_model.* and visual.*
        return key.removeprefix("model.")
