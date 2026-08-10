import mlx.core as mx

import mflux.models.krea2.model.krea2_scheduler  # noqa: F401 — register er_sde/euler schedulers
from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.krea2.model.krea2_text_encoder.text_encoder import Krea2TextEncoder
from mflux.models.krea2.model.krea2_transformer.transformer import Krea2Transformer
from mflux.models.krea2.weights.krea2_lora_mapping import Krea2LoRAMapping
from mflux.models.krea2.weights.krea2_weight_definition import Krea2WeightDefinition
from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE


class Krea2Initializer:
    @staticmethod
    def init(
        model,
        model_config: ModelConfig,
        quantize: int | None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ) -> None:
        path = model_path if model_path else model_config.model_name
        Krea2Initializer._init_config(model, model_config)
        weights = Krea2Initializer._load_weights(path)
        Krea2Initializer._init_tokenizers(model, path)
        Krea2Initializer._init_models(model, model_config)
        Krea2Initializer._apply_weights(model, weights, quantize)
        Krea2Initializer._apply_lora(model, lora_paths, lora_scales)
        del weights
        mx.eval(model)
        mx.clear_cache()

    @staticmethod
    def _init_tokenizers(model, model_path: str) -> None:
        model.tokenizers = TokenizerLoader.load_all(
            definitions=Krea2WeightDefinition.get_tokenizers(),
            model_path=model_path,
        )

    @staticmethod
    def _init_config(model, model_config: ModelConfig) -> None:
        model.prompt_cache = {}
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = None
        model.lora_paths = None
        model.lora_scales = None

    @staticmethod
    def _load_weights(model_path: str) -> LoadedWeights:
        return WeightLoader.load(weight_definition=Krea2WeightDefinition, model_path=model_path)

    @staticmethod
    def _init_models(model, model_config: ModelConfig) -> None:
        model.vae = QwenVAE()
        model.transformer = Krea2Transformer(**(model_config.transformer_overrides or {}))
        model.text_encoder = Krea2TextEncoder()

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None) -> None:
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=Krea2WeightDefinition,
            models={
                "vae": model.vae,
                "transformer": model.transformer,
                "text_encoder": model.text_encoder,
            },
        )

    @staticmethod
    def _apply_lora(model, lora_paths: list[str] | None, lora_scales: list[float] | None) -> None:
        model.lora_paths, model.lora_scales = LoRALoader.load_and_apply_lora(
            lora_mapping=Krea2LoRAMapping.get_mapping(),
            transformer=model.transformer,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )
