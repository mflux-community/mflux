import mlx.core as mx

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.compute_precision import ComputePrecision
from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.resolution.prequantized_resolution import PrequantizedResolution
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.models.flux3.model.flux3_transformer.flux3_transformer import Flux3Transformer
from mflux.models.flux3.weights.flux3_lora_mapping import Flux3LoRAMapping
from mflux.models.flux3.weights.flux3_weight_definition import (
    Flux3KleinBaseWeightDefinition,
    Flux3KleinWeightDefinition,
)
from mflux.utils.exceptions import ModelConfigError


class Flux3Initializer:
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
    ) -> None:
        precision = ComputePrecision(compute_precision)
        definition = Flux3Initializer.weight_definition(model_config)
        path = PrequantizedResolution.weights_root(
            model_config,
            model_path,
            quantize,
            has_loras=bool(lora_paths),
            bake_lora=bake_lora,
            **PrequantizedResolution.layout(definition),
        )
        Flux3Initializer._init_config(model, model_config)
        weights = Flux3Initializer._load_weights(path, definition)
        Flux3Initializer._check_checkpoint_kind(weights, model_config, path)
        Flux3Initializer._init_tokenizers(model, path)
        Flux3Initializer._init_models(model)
        Flux3Initializer._apply_weights(model, weights, quantize, definition)
        Flux3Initializer._apply_lora(model, lora_paths, lora_scales, bake_lora, weights)
        model.compute_precision = precision
        if compute_precision is not None:
            # Last, so it casts the final parameters, whatever quantization and LoRA produced.
            model.transformer.apply_compute_precision(precision)

    @staticmethod
    def _init_config(model, model_config: ModelConfig) -> None:
        model.prompt_cache = {}
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = None
        model._noise_override = None  # test hook: golden-parity scripts replace the seeded noise with a fixed tensor

    @staticmethod
    def weight_definition(model_config: ModelConfig):
        flow_map = model_config.transformer_overrides.get("flow_map", True)
        return Flux3KleinWeightDefinition if flow_map else Flux3KleinBaseWeightDefinition

    @staticmethod
    def _load_weights(model_path: str, definition) -> LoadedWeights:
        return WeightLoader.load(
            weight_definition=definition,
            model_path=model_path,
        )

    @staticmethod
    def _check_checkpoint_kind(weights: LoadedWeights, model_config: ModelConfig, model_path: str) -> None:
        # Only the distilled checkpoint has target_timestep_embedder. A mismatch would otherwise load with
        # strict=False and leave that embedder random (Base read as distilled) or drop it (the reverse).
        flow_map = model_config.transformer_overrides.get("flow_map", True)
        has_embedder = "target_timestep_embedder" in (weights.transformer or {})
        if has_embedder == flow_map:
            return
        kind, fix = (
            ("Distilled", ModelConfig.flux3_image_klein())
            if has_embedder
            else ("Base", ModelConfig.flux3_image_klein_base())
        )
        raise ModelConfigError(
            f"The checkpoint at {model_path} {'has' if has_embedder else 'has no'} target_timestep_embedder, "
            f"so it is a FLUX 3 Image Klein {kind} checkpoint, but it was loaded as "
            f"{'Base' if has_embedder else 'Distilled'}. Pass --base-model {fix.aliases[0]}."
        )

    @staticmethod
    def _init_tokenizers(model, model_path: str) -> None:
        model.tokenizers = TokenizerLoader.load_all(
            definitions=Flux3KleinWeightDefinition.get_tokenizers(),
            model_path=model_path,
        )

    @staticmethod
    def _init_models(model) -> None:
        model.vae = Flux2VAE()
        model.vae.decode_mode = Flux2VAE.DECODE_LOW_MEMORY
        model.transformer = Flux3Transformer(**model.model_config.transformer_overrides)
        model.text_encoder = Qwen3TextEncoder(**model.model_config.text_encoder_overrides)

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None, definition) -> None:
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=definition,
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
            lora_mapping=Flux3LoRAMapping.get_mapping(),
            transformer=model.transformer,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            bake_lora=bake_lora,
            dense_weights=weights.dense_component("transformer"),
        )
