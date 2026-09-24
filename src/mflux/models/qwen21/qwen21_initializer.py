from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_text_encoder import Qwen21TextEncoder
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer
from mflux.models.qwen21.model.qwen21_vae.qwen21_vae import Qwen21VAE
from mflux.models.qwen21.weights.qwen21_lora_mapping import Qwen21LoRAMapping
from mflux.models.qwen21.weights.qwen21_weight_definition import Qwen21WeightDefinition


class Qwen21Initializer:
    @staticmethod
    def init(
        model,
        quantize: int | None,
        model_path: str | None,
        model_config: ModelConfig,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        bake_lora: bool = True,
    ) -> None:
        path = model_path if model_path else model_config.model_name
        Qwen21Initializer._init_config(model, model_config)
        weights = Qwen21Initializer._load_weights(path)
        Qwen21Initializer._init_tokenizers(model, path)
        Qwen21Initializer._init_models(model)
        Qwen21Initializer._apply_weights(model, weights, quantize)
        model.lora_paths, model.lora_scales = LoRALoader.load_and_apply_lora(
            lora_mapping=Qwen21LoRAMapping.get_mapping(),
            transformer=model.transformer,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            bake_lora=bake_lora,
        )

    @staticmethod
    def _init_config(model, model_config: ModelConfig) -> None:
        model.prompt_cache = {}
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = None

    @staticmethod
    def _load_weights(model_path: str) -> LoadedWeights:
        return WeightLoader.load(
            weight_definition=Qwen21WeightDefinition,
            model_path=model_path,
        )

    @staticmethod
    def _init_tokenizers(model, model_path: str) -> None:
        model.tokenizers = TokenizerLoader.load_all(
            definitions=Qwen21WeightDefinition.get_tokenizers(),
            model_path=model_path,
        )

    @staticmethod
    def _init_models(model) -> None:
        model.vae = Qwen21VAE()
        model.transformer = Qwen21Transformer()
        model.text_encoder = Qwen21TextEncoder()

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None) -> None:
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=Qwen21WeightDefinition,
            models={
                "vae": model.vae,
                "transformer": model.transformer,
                "text_encoder": model.text_encoder,
            },
        )
