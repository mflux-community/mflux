from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.fibo.model.fibo_text_encoder import SmolLM3_3B_TextEncoder
from mflux.models.fibo.model.fibo_transformer import FiboTransformer
from mflux.models.fibo.model.fibo_vae.wan_2_2_vae import Wan2_2_VAE
from mflux.models.fibo.weights.fibo_weight_definition import FIBOWeightDefinition


class FIBOInitializer:
    # No lora_paths/lora_scales parameters: mflux has no LoRA mapping for FIBO, and a
    # signature that accepts an adapter it cannot apply is how `mflux-generate-fibo-edit
    # --lora` came to download a file and then ignore it.
    @staticmethod
    def init(
        model,
        model_config: ModelConfig,
        quantize: int | None = None,
        model_path: str | None = None,
    ) -> None:
        path = model_path if model_path else model_config.model_name
        FIBOInitializer._init_config(model, model_config)
        weights = FIBOInitializer._load_weights(path)
        FIBOInitializer._init_tokenizers(model, path)
        FIBOInitializer._init_models(model)
        FIBOInitializer._apply_weights(model, weights, quantize)

    @staticmethod
    def _init_config(model, model_config: ModelConfig) -> None:
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = None

    @staticmethod
    def _load_weights(model_path: str) -> LoadedWeights:
        return WeightLoader.load(
            weight_definition=FIBOWeightDefinition,
            model_path=model_path,
        )

    @staticmethod
    def _init_tokenizers(model, model_path: str) -> None:
        model.tokenizers = TokenizerLoader.load_all(
            definitions=FIBOWeightDefinition.get_tokenizers(),
            model_path=model_path,
        )

    @staticmethod
    def _init_models(model) -> None:
        model.vae = Wan2_2_VAE()
        model.text_encoder = SmolLM3_3B_TextEncoder()
        model.transformer = FiboTransformer()

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None) -> None:
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=FIBOWeightDefinition,
            models={
                "vae": model.vae,
                "transformer": model.transformer,
                "text_encoder": model.text_encoder,
            },
        )
