import mlx.core as mx

import mflux.models.moss_image.model.moss_image_scheduler  # noqa: F401 - registers the moss_image scheduler
from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.common_models.qwen3_5 import Qwen35TextModel
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.models.moss_image.model.moss_image_transformer.moss_image_transformer import MOSSImageTransformer
from mflux.models.moss_image.weights.moss_image_lora_mapping import MOSSImageLoRAMapping
from mflux.models.moss_image.weights.moss_image_weight_definition import MOSSImageWeightDefinition

# MOSS-Image ships the FLUX.2 VAE with the small decoder (vae/config.json decoder_block_out_channels).
MOSS_IMAGE_DECODER_BLOCK_OUT_CHANNELS = (96, 192, 384, 384)


class MOSSImageInitializer:
    @staticmethod
    def init(
        model,
        model_config: ModelConfig,
        quantize: int | None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        bake_lora: bool = False,
    ) -> None:
        path = model_path if model_path else model_config.model_name
        MOSSImageInitializer._init_config(model, model_config)
        weights = MOSSImageInitializer._load_weights(path)
        MOSSImageInitializer._init_tokenizers(model, path)
        MOSSImageInitializer._init_models(model, model_config)
        MOSSImageInitializer._apply_weights(model, weights, quantize)
        MOSSImageInitializer._apply_lora(model, lora_paths, lora_scales, bake_lora, weights)
        # The parameters stay lazy: each one is read (and quantized) when a forward pass first needs
        # it. Evaluating them here held the text encoder, transformer and VAE at once, so --low-ram
        # could only free the text encoder after that peak (#948).
        del weights
        mx.clear_cache()

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
        return WeightLoader.load(weight_definition=MOSSImageWeightDefinition, model_path=model_path)

    @staticmethod
    def _init_tokenizers(model, model_path: str) -> None:
        model.tokenizers = TokenizerLoader.load_all(
            definitions=MOSSImageWeightDefinition.get_tokenizers(),
            model_path=model_path,
        )

    @staticmethod
    def _init_models(model, model_config: ModelConfig) -> None:
        model.vae = Flux2VAE(decoder_block_out_channels=MOSS_IMAGE_DECODER_BLOCK_OUT_CHANNELS)
        model.transformer = MOSSImageTransformer(**model_config.transformer_overrides)
        model.text_encoder = Qwen35TextModel(**model_config.text_encoder_overrides)

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None) -> None:
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=MOSSImageWeightDefinition,
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
        weights: LoadedWeights,
    ) -> None:
        model.lora_paths, model.lora_scales = LoRALoader.load_and_apply_lora(
            lora_mapping=MOSSImageLoRAMapping.get_mapping(),
            transformer=model.transformer,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            bake_lora=bake_lora,
            dense_weights=weights.dense_component("transformer"),
        )
