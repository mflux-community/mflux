import mlx.core as mx

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.compute_precision import ComputePrecision
from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.resolution.prequantized_resolution import PrequantizedResolution
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.vae.block_decoder import VAEBlockDecoder
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
from mflux.models.flux2.model.flux2_text_encoder.small_text_encoder import (
    SUPPORTED_JOINT_ATTENTION_DIM,
    Flux2SmallTextEncoder,
)
from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer
from mflux.models.flux2.model.flux2_vae.small_decoder import Flux2SmallDecoderLoader
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.models.flux2.weights.flux2_lora_mapping import Flux2LoRAMapping
from mflux.models.flux2.weights.flux2_weight_definition import (
    Flux2KleinNoTextEncoderWeightDefinition,
    Flux2KleinWeightDefinition,
)


class Flux2Initializer:
    @staticmethod
    def init(
        model,
        model_config: ModelConfig,
        quantize: int | None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        bake_lora: bool = True,
        compute_precision: mx.Dtype | None = None,
        small_text_encoder: bool = False,
        small_decoder: bool = False,
    ) -> None:
        if small_text_encoder:
            Flux2Initializer.validate_small_text_encoder(model_config)
        precision = ComputePrecision(compute_precision)
        definition = Flux2KleinNoTextEncoderWeightDefinition if small_text_encoder else Flux2KleinWeightDefinition
        path = PrequantizedResolution.weights_root(
            model_config,
            model_path,
            quantize,
            has_loras=bool(lora_paths),
            bake_lora=bake_lora,
            **PrequantizedResolution.layout(definition),
        )
        Flux2Initializer._init_config(model, model_config)
        model.small_text_encoder = small_text_encoder
        if small_text_encoder:
            weights = Flux2Initializer._load_weights_without_text_encoder(path)
        else:
            weights = Flux2Initializer._load_weights(path)
        Flux2Initializer._init_tokenizers(model, path)
        Flux2Initializer._init_models(model)
        Flux2Initializer._apply_weights(model, weights, quantize)
        if small_text_encoder:
            model.text_encoder = Flux2SmallTextEncoder.load(bits=model.bits)
        model.small_decoder = small_decoder
        if small_decoder:
            model.vae = Flux2SmallDecoderLoader.with_small_decoder(model.vae, bits=model.bits)
        Flux2Initializer._apply_lora(model, lora_paths, lora_scales, bake_lora, weights)
        model.compute_precision = precision
        if compute_precision is not None:
            # Last, so it casts the final parameters, whatever quantization and LoRA produced.
            model.transformer.apply_compute_precision(precision)

    @staticmethod
    def supports_small_text_encoder(model_config: ModelConfig) -> bool:
        return model_config.transformer_overrides.get("joint_attention_dim") == SUPPORTED_JOINT_ATTENTION_DIM

    @staticmethod
    def validate_small_text_encoder(model_config: ModelConfig) -> None:
        if not Flux2Initializer.supports_small_text_encoder(model_config):
            raise ValueError("--small-text-encoder only works with FLUX.2-klein-4B / klein-base-4B.")

    @staticmethod
    def _init_config(model, model_config: ModelConfig) -> None:
        model.prompt_cache = {}
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = None

    @staticmethod
    def _load_weights(model_path: str) -> LoadedWeights:
        return WeightLoader.load(
            weight_definition=Flux2KleinWeightDefinition,
            model_path=model_path,
        )

    @staticmethod
    def _load_weights_without_text_encoder(model_path: str) -> LoadedWeights:
        return WeightLoader.load(
            weight_definition=Flux2KleinNoTextEncoderWeightDefinition,
            model_path=model_path,
        )

    @staticmethod
    def _init_tokenizers(model, model_path: str) -> None:
        model.tokenizers = TokenizerLoader.load_all(
            definitions=Flux2KleinWeightDefinition.get_tokenizers(),
            model_path=model_path,
        )

    @staticmethod
    def _init_models(model) -> None:
        model.vae = Flux2VAE()
        model.vae.decode_mode = VAEBlockDecoder.LOW_MEMORY
        model.transformer = Flux2Transformer(**model.model_config.transformer_overrides)
        model.text_encoder = None
        if not model.small_text_encoder:
            model.text_encoder = Qwen3TextEncoder(**model.model_config.text_encoder_overrides)

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None) -> None:
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=Flux2KleinNoTextEncoderWeightDefinition
            if model.small_text_encoder
            else Flux2KleinWeightDefinition,
            models=Flux2Initializer._weighted_models(model),
        )

    @staticmethod
    def _weighted_models(model) -> dict:
        models = {"vae": model.vae, "transformer": model.transformer}
        if not model.small_text_encoder:
            models["text_encoder"] = model.text_encoder
        return models

    @staticmethod
    def _apply_lora(
        model,
        lora_paths: list[str] | None,
        lora_scales: list[float] | None,
        bake_lora: bool,
        weights: LoadedWeights,
    ) -> None:
        model.lora_paths, model.lora_scales = LoRALoader.load_and_apply_lora(
            lora_mapping=Flux2LoRAMapping.get_mapping(),
            transformer=model.transformer,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            bake_lora=bake_lora,
            dense_weights=weights.dense_component("transformer"),
        )
