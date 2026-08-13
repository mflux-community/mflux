from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.boogu.model.boogu_text_encoder import BooguTextEncoder
from mflux.models.boogu.model.boogu_transformer.boogu_transformer import BooguImageTransformer
from mflux.models.boogu.weights.boogu_weight_definition import BooguWeightDefinition
from mflux.models.common.config import ModelConfig
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.flux.model.flux_vae.vae import VAE


class BooguInitializer:
    """Assemble a Boogu-Image-Turbo model: FLUX VAE + transformer + Qwen3-VL encoder."""

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
        BooguInitializer._init_config(model, model_config)
        weights = BooguInitializer._load_weights(path)
        BooguInitializer._init_tokenizers(model, path)
        BooguInitializer._init_models(model)
        BooguInitializer._apply_weights(model, weights, quantize)

    @staticmethod
    def _init_config(model, model_config: ModelConfig) -> None:
        model.prompt_cache = {}
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = None
        model.lora_paths = []
        model.lora_scales = []

    @staticmethod
    def _load_weights(model_path: str) -> LoadedWeights:
        return WeightLoader.load(weight_definition=BooguWeightDefinition, model_path=model_path)

    @staticmethod
    def _init_tokenizers(model, model_path: str) -> None:
        model.tokenizers = TokenizerLoader.load_all(
            definitions=BooguWeightDefinition.get_tokenizers(),
            model_path=model_path,
        )

    @staticmethod
    def _init_models(model) -> None:
        model.vae = VAE()
        model.transformer = BooguImageTransformer()
        model.text_encoder = BooguTextEncoder()

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None) -> None:
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=BooguWeightDefinition,
            models={
                "vae": model.vae,
                "transformer": model.transformer,
                "text_encoder": model.text_encoder,
            },
        )
