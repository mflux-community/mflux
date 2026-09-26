from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_text_encoder import Qwen21TextEncoder
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer
from mflux.models.qwen21.model.qwen21_vae.qwen21_vae import Qwen21VAE
from mflux.models.qwen21.weights.qwen21_edit_weight_definition import Qwen21EditWeightDefinition


class Qwen21EditInitializer:
    @staticmethod
    def init(
        model,
        quantize: int | None,
        model_path: str | None,
        model_config: ModelConfig,
    ) -> None:
        path = model_path if model_path else model_config.model_name
        Qwen21EditInitializer._init_config(model, model_config)
        weights = Qwen21EditInitializer._load_weights(path)
        Qwen21EditInitializer._init_tokenizers(model, path)
        Qwen21EditInitializer._init_models(model)
        Qwen21EditInitializer._apply_weights(model, weights, quantize)

    @staticmethod
    def _init_config(model, model_config: ModelConfig) -> None:
        model.prompt_cache = {}
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = None

    @staticmethod
    def _load_weights(model_path: str) -> LoadedWeights:
        return WeightLoader.load(
            weight_definition=Qwen21EditWeightDefinition,
            model_path=model_path,
        )

    @staticmethod
    def _init_tokenizers(model, model_path: str) -> None:
        model.tokenizers = TokenizerLoader.load_all(
            definitions=Qwen21EditWeightDefinition.get_tokenizers(),
            model_path=model_path,
        )

    @staticmethod
    def _init_models(model) -> None:
        model.vae = Qwen21VAE()
        model.transformer = Qwen21Transformer()
        model.text_encoder = Qwen21TextEncoder(with_visual=True)

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None) -> None:
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=Qwen21EditWeightDefinition,
            models={
                "vae": model.vae,
                "transformer": model.transformer,
                "text_encoder": model.text_encoder,
            },
        )
