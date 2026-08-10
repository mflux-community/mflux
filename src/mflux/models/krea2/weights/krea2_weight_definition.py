from typing import List

import mlx.core as mx

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.tokenizer import LanguageTokenizer
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.krea2.model.krea2_text_encoder.text_encoder import KREA2_TEMPLATE
from mflux.models.krea2.weights.krea2_weight_mapping import Krea2WeightMapping


class Krea2WeightDefinition:
    _TE_PREFIXES = ("model.language_model.", "language_model.")

    @staticmethod
    def strip_te_prefix(key: str) -> str | None:
        for prefix in Krea2WeightDefinition._TE_PREFIXES:
            if key.startswith(prefix):
                return key[len(prefix) :]
        return None

    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                loading_mode="single",
                mapping_getter=Krea2WeightMapping.get_vae_mapping,
            ),
            ComponentDefinition(
                # The official repo's transformer/ subdir is diffusers-format (different
                # keys). We load the native single-file turbo.safetensors from the repo
                # root instead (hf_subdir="" -> the model root), which matches our mapping.
                name="transformer",
                hf_subdir="",
                loading_mode="mlx_native",
                weight_files=["turbo.safetensors"],
                precision=ModelConfig.precision,
                mapping_getter=Krea2WeightMapping.get_transformer_mapping,
                num_layers=28,
            ),
            ComponentDefinition(
                name="text_encoder",
                hf_subdir="text_encoder",
                loading_mode="mlx_native",
                precision=mx.bfloat16,
                skip_quantization=True,  # quantizing the TE degrades conditioning
                mapping_getter=None,  # direct load; key_transform strips prefix to match module paths
                key_transform=Krea2WeightDefinition.strip_te_prefix,
            ),
        ]

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return [
            TokenizerDefinition(
                name="qwen3vl",
                hf_subdir="tokenizer",
                tokenizer_class="AutoTokenizer",
                encoder_class=LanguageTokenizer,
                max_length=1024,
                padding="longest",
                template=KREA2_TEMPLATE,
                download_patterns=["tokenizer/**", "added_tokens.json", "chat_template.jinja"],
            ),
        ]

    @staticmethod
    def get_download_patterns() -> List[str]:
        # Fetch the native root transformer (turbo.safetensors) + VAE + single-file
        # text encoder + tokenizer. Deliberately skip the diffusers-format
        # transformer/ subdir (different keys, and ~26 GB of redundant shards).
        return [
            "turbo.safetensors",
            "vae/*.safetensors",
            "vae/*.json",
            "text_encoder/*.safetensors",
            "text_encoder/*.json",
            "tokenizer/**",
        ]

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        # Skip layers whose input dim isn't divisible by the group size (e.g. the
        # txtfusion projector, Linear(12->1)) — mx.quantize requires last-dim % 64 == 0.
        if not hasattr(module, "to_quantized"):
            return False
        weight = getattr(module, "weight", None)
        return weight is not None and weight.shape[-1] % 64 == 0
