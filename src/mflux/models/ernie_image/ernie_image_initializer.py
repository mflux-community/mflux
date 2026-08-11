import mlx.core as mx

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.vae.tiling_config import TilingConfig
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.ernie_image.model.ernie_text_encoder.text_encoder import ErnieMistralTextEncoder
from mflux.models.ernie_image.model.ernie_transformer.transformer import ErnieTransformer
from mflux.models.ernie_image.weights.ernie_lora_mapping import ErnieLoRAMapping
from mflux.models.ernie_image.weights.ernie_weight_definition import ErnieWeightDefinition
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE


class ErnieImageInitializer:
    @staticmethod
    def init(
        model,
        model_config: ModelConfig,
        quantize: int | None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        bake_lora: bool = True,
    ) -> None:
        path = model_path if model_path else model_config.model_name
        ErnieImageInitializer._init_config(model, model_config)
        weights = ErnieImageInitializer._load_weights(path)
        ErnieImageInitializer._init_tokenizers(model, path)
        ErnieImageInitializer._init_models(model, model_config)
        ErnieImageInitializer._apply_weights(model, weights, quantize)
        del weights
        mx.eval(model)
        mx.clear_cache()
        ErnieImageInitializer._apply_lora(model, lora_paths, lora_scales, bake_lora)

    @staticmethod
    def _init_config(model, model_config: ModelConfig) -> None:
        model.prompt_cache = {}
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = TilingConfig(vae_decode_tiles_per_dim=None)

    @staticmethod
    def _load_weights(model_path: str) -> LoadedWeights:
        return WeightLoader.load(
            weight_definition=ErnieWeightDefinition,
            model_path=model_path,
        )

    @staticmethod
    def _init_tokenizers(model, model_path: str) -> None:
        model.tokenizers = TokenizerLoader.load_all(
            definitions=ErnieWeightDefinition.get_tokenizers(),
            model_path=model_path,
        )

    @staticmethod
    def _init_models(model, model_config: ModelConfig) -> None:
        model.vae = Flux2VAE()
        model.transformer = ErnieTransformer(**(model_config.transformer_overrides or {}))
        model.text_encoder = ErnieMistralTextEncoder()

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None) -> None:
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=ErnieWeightDefinition,
            models={
                "vae": model.vae,
                "transformer": model.transformer,
                "text_encoder": model.text_encoder,
            },
        )

    @staticmethod
    def _apply_lora(
        model,
        lora_paths: list[str] | None,
        lora_scales: list[float] | None,
        bake_lora: bool,
    ) -> None:
        model.lora_paths, model.lora_scales = LoRALoader.load_and_apply_lora(
            lora_mapping=ErnieLoRAMapping.get_mapping(),
            transformer=model.transformer,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            bake_lora=bake_lora,
        )
