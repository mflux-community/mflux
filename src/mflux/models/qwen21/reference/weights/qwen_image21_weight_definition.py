import mlx.core as mx

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.tokenizer import LanguageTokenizer
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition


class QwenImage21WeightDefinition:
    @staticmethod
    def get_components() -> list[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                precision=mx.float32,
                skip_quantization=True,
                weight_transform=QwenImage21WeightDefinition.vae_weight,
            ),
            ComponentDefinition(name="transformer", hf_subdir="transformer", precision=ModelConfig.precision),
            ComponentDefinition(
                name="text_encoder",
                hf_subdir="text_encoder",
                precision=ModelConfig.precision,
                key_transform=QwenImage21WeightDefinition.text_key,
                weight_transform=QwenImage21WeightDefinition.text_weight,
            ),
        ]

    @staticmethod
    def get_download_patterns() -> list[str]:
        return [
            "model_index.json",
            "scheduler/*.json",
            "processor/**",
            "vae/*.json",
            "vae/*.safetensors",
            "transformer/*.json",
            "transformer/*.safetensors",
            "text_encoder/*.json",
            "text_encoder/*.safetensors",
        ]

    @staticmethod
    def get_tokenizers() -> list[TokenizerDefinition]:
        return [
            TokenizerDefinition(
                name="qwen21",
                hf_subdir="processor",
                encoder_class=LanguageTokenizer,
                download_patterns=["processor/**"],
                padding="longest",
            )
        ]

    @staticmethod
    def text_key(key: str) -> str | None:
        key = key.removeprefix("model.")
        if key == "language_model.norm.weight" or key.startswith("lm_head."):
            return None
        return key

    @staticmethod
    def text_weight(key: str, weight: mx.array) -> mx.array:
        if key.endswith("patch_embed.proj.weight"):
            return weight.transpose(0, 2, 3, 4, 1)
        return weight

    @staticmethod
    def vae_weight(key: str, weight: mx.array) -> mx.array:
        if key.endswith(".gamma"):
            return weight.reshape(-1)
        if key.endswith(".weight") and weight.ndim == 4:
            return weight.transpose(0, 2, 3, 1)
        return weight

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        return (
            hasattr(module, "to_quantized")
            and module.weight.shape[-1] % 64 == 0
            and not any(name in path for name in ("modulation", "time_text_embed", "norm_out"))
        )
