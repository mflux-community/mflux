from typing import List

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.tokenizer import LanguageTokenizer
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.common_models.qwen3_vl.qwen3_vl_multimodal import Qwen3VLMultimodal
from mflux.models.fibo2.model.fibo2_text_encoder.fibo2_text_encoder import Fibo2TextEncoder
from mflux.models.fibo2.model.fibo2_text_encoder.prompt_encoder import Fibo2PromptEncoder
from mflux.models.fibo2.weights.fibo2_weight_mapping import Fibo2WeightMapping


class Fibo2WeightDefinition:
    FONT_PATTERNS = ["fonts/*.otf", "fonts/*.ttf", "fonts/LICENSE-*.txt"]

    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                precision=ModelConfig.precision,
                mapping_getter=Fibo2WeightMapping.get_vae_mapping,
            ),
            ComponentDefinition(
                name="transformer",
                hf_subdir="transformer",
                precision=ModelConfig.precision,
                mapping_getter=Fibo2WeightMapping.get_transformer_mapping,
            ),
            ComponentDefinition(
                name="text_encoder",
                hf_subdir="text_encoder",
                precision=ModelConfig.precision,
                key_transform=Fibo2WeightDefinition._text_encoder_key,
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
                max_length=Fibo2PromptEncoder.MAX_LENGTH,
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
            "transformer/*.safetensors",
            "transformer/*.json",
            "text_encoder/*.safetensors",
            "text_encoder/*.json",
            "processor/*",
            *Fibo2WeightDefinition.FONT_PATTERNS,
        ]

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        return hasattr(module, "to_quantized")

    @staticmethod
    def _text_encoder_key(key: str) -> str | None:
        # model.language_model.* and model.visual.* load as language_model.* and visual.*; the last layer is unused
        name = key.removeprefix("model.")
        if name.startswith("language_model.layers.") and int(name.split(".")[2]) >= Fibo2TextEncoder.NUM_LAYERS:
            return None
        return name
