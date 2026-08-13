from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

from mflux.models.common.config import ModelConfig
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.flux2.weights.flux2_weight_mapping import Flux2WeightMapping
from mflux.models.ideogram4.model.ideogram4_text_encoder import Ideogram4Tokenizer
from mflux.models.ideogram4.weights.ideogram4_weight_mapping import Ideogram4WeightMapping


class Ideogram4WeightDefinition:
    FP8_TEXT_ENCODER_CONFIG_KEY = "ideogram_fp8_weight_only"

    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                num_blocks=4,
                precision=ModelConfig.precision,
                mapping_getter=Flux2WeightMapping.get_vae_mapping,
            ),
            ComponentDefinition(
                name="conditional_transformer",
                hf_subdir="transformer",
                loading_mode="fp8_safetensors",
                weight_transform=Ideogram4WeightMapping.prepare_tensor,
            ),
            ComponentDefinition(
                name="unconditional_transformer",
                hf_subdir="unconditional_transformer",
                loading_mode="fp8_safetensors",
                weight_transform=Ideogram4WeightMapping.prepare_tensor,
            ),
            ComponentDefinition(
                name="text_encoder",
                hf_subdir="text_encoder",
                loading_mode="fp8_safetensors",
                weight_prefix_filters=["language_model."],
                key_transform=Ideogram4WeightMapping.transform_text_encoder_key,
                weight_transform=Ideogram4WeightMapping.prepare_tensor,
            ),
        ]

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return [
            TokenizerDefinition(
                name="ideogram4",
                hf_subdir="tokenizer",
                tokenizer_class="AutoTokenizer",
                encoder_class=Ideogram4Tokenizer,
                max_length=2048,
                add_special_tokens=False,
                download_patterns=["tokenizer/**"],
            ),
        ]

    @staticmethod
    def get_download_patterns() -> List[str]:
        return [
            "model_index.json",
            "scheduler/*.json",
            "vae/*.safetensors",
            "vae/*.json",
            "transformer/*.safetensors",
            "transformer/*.json",
            "unconditional_transformer/*.safetensors",
            "unconditional_transformer/*.json",
            "text_encoder/*.safetensors",
            "text_encoder/*.json",
            "tokenizer/**",
        ]

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        return hasattr(module, "to_quantized")

    @staticmethod
    def validate_fp8_checkpoint(root: str | Path) -> Path:
        checkpoint = Path(root).expanduser()
        Ideogram4WeightDefinition._validate_fp8_layout(checkpoint)
        return checkpoint

    @staticmethod
    def is_builtin_name(name: str | None) -> bool:
        if name is None:
            return False
        try:
            return ModelConfig.from_name(name).model_name == ModelConfig.ideogram4_fp8().model_name
        except ValueError:
            return False

    @staticmethod
    def resolve_inference_config(
        model: str,
        base_model: str | None,
        model_path: str | None,
    ) -> ModelConfig:
        if Ideogram4WeightDefinition.is_builtin_name(model):
            return ModelConfig.from_name(model, base_model=base_model)
        if Ideogram4WeightDefinition.is_builtin_name(base_model):
            return ModelConfig.ideogram4_fp8()
        if model_path is None:
            return ModelConfig.from_name(model, base_model=base_model)
        return ModelConfig.ideogram4_fp8()

    @staticmethod
    def _validate_fp8_layout(root: Path) -> None:
        if Ideogram4WeightDefinition._is_mflux_saved_checkpoint(root):
            return
        text_encoder_config = Ideogram4WeightDefinition._load_json(root / "text_encoder" / "config.json")
        if text_encoder_config.get(Ideogram4WeightDefinition.FP8_TEXT_ENCODER_CONFIG_KEY) is True:
            return
        raise ValueError(
            "Ideogram 4 support currently requires the FP8 checkpoint layout "
            f"(expected text_encoder/config.json to contain "
            f"{Ideogram4WeightDefinition.FP8_TEXT_ENCODER_CONFIG_KEY!r}: true): {root}"
        )

    @staticmethod
    def _is_mflux_saved_checkpoint(root: Path) -> bool:
        for component in Ideogram4WeightDefinition.get_components():
            component_dir = root / component.hf_subdir
            index_path = component_dir / "model.safetensors.index.json"
            if not index_path.exists():
                continue
            index = Ideogram4WeightDefinition._load_json(index_path)
            metadata = index.get("metadata")
            if isinstance(metadata, dict) and metadata.get("mflux_version"):
                return True
        return False

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            value = json.loads(path.read_text())
        except OSError as exc:
            raise ValueError(f"Could not read JSON file: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON file: {path}") from exc
        return value if isinstance(value, dict) else {}
