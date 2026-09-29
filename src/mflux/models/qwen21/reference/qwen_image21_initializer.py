import json
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
from mlx.utils import tree_flatten

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.qwen21.reference.model.qwen_image21_text_encoder.processor import QwenImage21Processor
from mflux.models.qwen21.reference.model.qwen_image21_text_encoder.text_encoder import QwenImage21TextEncoder
from mflux.models.qwen21.reference.model.qwen_image21_transformer.transformer import QwenImage21Transformer
from mflux.models.qwen21.reference.model.qwen_image21_vae.vae import QwenImage21VAE
from mflux.models.qwen21.reference.weights.qwen_image21_weight_definition import QwenImage21WeightDefinition

if TYPE_CHECKING:
    from mflux.models.qwen21.reference import QwenImage21Edit


class QwenImage21Initializer:
    @staticmethod
    def init(model: "QwenImage21Edit", model_config: ModelConfig, quantize: int | None, model_path: str | None) -> None:
        root = PathResolution.resolve(
            model_path or model_config.model_name, QwenImage21WeightDefinition.get_download_patterns()
        )
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = None
        model.prompt_cache = {}
        model.lora_paths = None
        model.lora_scales = None
        model._checkpoint_path = str(root)
        model._component_configs = {
            name: json.loads((root / name / "config.json").read_text())
            for name in ("vae", "transformer", "text_encoder")
        }
        model.tokenizers = TokenizerLoader.load_all(QwenImage21WeightDefinition.get_tokenizers(), str(root))
        model.processor = QwenImage21Processor(root / "processor", model.tokenizers["qwen21"].tokenizer)
        model.vae = QwenImage21VAE(model._component_configs["vae"])
        model.transformer = QwenImage21Transformer(model._component_configs["transformer"])
        model.text_encoder = QwenImage21TextEncoder(model._component_configs["text_encoder"])
        # Load one component at a time so quantization does not retain all dense weights.
        for component in QwenImage21WeightDefinition.get_components():
            module = getattr(model, component.name)
            weights = WeightLoader.load_single_local(component, Path(root))
            supplied = dict(tree_flatten(weights.components[component.name]))
            if weights.meta_data.quantization_level is None:
                QwenImage21Initializer._validate_weights(component.name, module, supplied)
            model.bits = WeightApplier.apply_and_quantize_single(
                weights,
                module,
                component,
                quantize,
                quantization_predicate=QwenImage21WeightDefinition.quantization_predicate,
            )
            if weights.meta_data.quantization_level is not None:
                QwenImage21Initializer._validate_weights(component.name, module, supplied)
            mx.eval(module.parameters())
            del weights, supplied
            mx.clear_cache()

    @staticmethod
    def _validate_weights(name: str, module, supplied: dict[str, mx.array]) -> None:
        expected = dict(tree_flatten(module.parameters()))
        missing = {key for key in set(expected) - set(supplied) if not key.endswith(".inv_freq")}
        unexpected = set(supplied) - set(expected)
        mismatched = [key for key in expected.keys() & supplied.keys() if expected[key].shape != supplied[key].shape]
        if missing or unexpected or mismatched:
            raise ValueError(
                f"{name} checkpoint mismatch: missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}, shapes={mismatched}"
            )
