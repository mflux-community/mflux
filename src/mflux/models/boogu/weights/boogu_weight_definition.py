from typing import List

from mflux.models.boogu.weights.boogu_weight_mapping import BooguWeightMapping
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.tokenizer import LanguageTokenizer
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition


class BooguWeightDefinition:
    """Component, tokenizer, and download definitions for Boogu-Image-Turbo.

    Loads three components from the Boogu HF repo: the FLUX.1 ``vae``, the
    ``transformer`` (identity-mapped), and the Qwen3-VL ``mllm`` text decoder
    (only ``model.language_model.*`` tensors; the vision tower and ``lm_head``
    are skipped for text-only T2I).
    """

    # Boogu's hidden size (3360) is divisible by 32 but not 64, so MLX's default
    # quantization group size of 64 fails on the transformer's 3360-dim linears.
    quantization_group_size = 32

    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                precision=ModelConfig.precision,
                mapping_getter=BooguWeightMapping.get_vae_mapping,
            ),
            ComponentDefinition(
                name="transformer",
                hf_subdir="transformer",
                precision=ModelConfig.precision,
                mapping_getter=BooguWeightMapping.get_transformer_mapping,
            ),
            ComponentDefinition(
                name="text_encoder",
                hf_subdir="mllm",
                precision=ModelConfig.precision,
                weight_prefix_filters=["model.language_model"],
                mapping_getter=BooguWeightMapping.get_text_encoder_mapping,
            ),
        ]

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return [
            TokenizerDefinition(
                name="qwen3vl",
                hf_subdir="mllm",
                tokenizer_class="Qwen2Tokenizer",
                encoder_class=LanguageTokenizer,
                max_length=1024,
                fallback_subdirs=["tokenizer", "processor"],
                download_patterns=[
                    "mllm/vocab.json",
                    "mllm/merges.txt",
                    "mllm/tokenizer.json",
                    "mllm/tokenizer_config.json",
                    "mllm/chat_template.jinja",
                ],
            ),
        ]

    @staticmethod
    def get_download_patterns() -> List[str]:
        return [
            "vae/*.safetensors",
            "vae/*.json",
            "transformer/*.safetensors",
            "transformer/*.json",
            "mllm/*.safetensors",
            "mllm/*.json",
            "mllm/chat_template.jinja",
        ]

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        return hasattr(module, "to_quantized")
