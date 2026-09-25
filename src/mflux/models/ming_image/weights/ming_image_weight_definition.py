from typing import List

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.tokenizer import LanguageTokenizer
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.qwen.weights.qwen_weight_mapping import QwenWeightMapping
from mflux.models.z_image.weights.z_image_weight_mapping import ZImageWeightMapping


class MingImageWeightDefinition:
    """inclusionAI/Ming-Image-0.1-Design. The text encoder, connector and heads load in
    passthrough mode (their module paths follow the checkpoint names after the key transforms
    below); the DiT reuses Z-Image's mapping and the VAE Qwen-Image's, whose checkpoints use the
    same tensor names. The mllm/ shards also hold the Qwen2.5 ViT, lm_head and audio router,
    which text-to-image never uses and which are dropped on load."""

    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="text_encoder",
                hf_subdir="mllm",
                precision=ModelConfig.precision,
                mapping_getter=None,
                key_transform=MingImageWeightDefinition._text_encoder_key,
            ),
            ComponentDefinition(
                name="connector",
                hf_subdir="connector",
                precision=ModelConfig.precision,
                mapping_getter=None,
                key_transform=MingImageWeightDefinition._connector_key,
            ),
            ComponentDefinition(
                name="heads",
                hf_subdir="mlp",
                precision=ModelConfig.precision,
                skip_quantization=True,  # ~30M params that feed the DiT directly
                mapping_getter=None,
                key_transform=MingImageWeightDefinition._heads_key,
            ),
            ComponentDefinition(
                name="transformer",
                hf_subdir="transformer",
                num_layers=30,
                precision=ModelConfig.precision,
                mapping_getter=ZImageWeightMapping.get_transformer_mapping,
            ),
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                loading_mode="single",
                precision=ModelConfig.precision,
                mapping_getter=QwenWeightMapping.get_vae_mapping,
            ),
        ]

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return [
            TokenizerDefinition(
                name="ming",
                hf_subdir="mllm",
                tokenizer_class="PreTrainedTokenizerFast",
                encoder_class=LanguageTokenizer,
                max_length=1024,
                add_special_tokens=False,
                download_patterns=["mllm/tokenizer*", "mllm/special_tokens_map.json"],
            ),
        ]

    @staticmethod
    def get_download_patterns() -> List[str]:
        return [
            "mllm/*.safetensors",
            "mllm/*.json",
            "connector/*.safetensors",
            "connector/*.json",
            "mlp/*.safetensors",
            "mlp/*.json",
            "transformer/*.safetensors",
            "transformer/*.json",
            "vae/*.safetensors",
            "vae/*.json",
        ]

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        # Routers stay in full precision: their scores pick the experts, and the gate modules
        # have no to_quantized anyway. Norms and the query tokens are not Linear/Embedding.
        return hasattr(module, "to_quantized")

    @staticmethod
    def _text_encoder_key(key: str) -> str | None:
        prefix = "model.model."
        if not key.startswith(prefix) or ".audio_gate." in key:
            return None
        return key[len(prefix) :]

    @staticmethod
    def _connector_key(key: str) -> str | None:
        prefix = "model."
        if not key.startswith(prefix) or key.startswith("model.embed_tokens"):
            return None
        return key[len(prefix) :]

    @staticmethod
    def _heads_key(key: str) -> str | None:
        if key.startswith("query_tokens_dict."):
            return "query_tokens"
        if key.startswith(("proj_in.", "proj_out.", "proj_directvlm.")):
            return key
        return None  # "mlp.*" is the DiT-side adapter, an identity for this checkpoint
