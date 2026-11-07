from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.resolution.lora_resolution import LoraResolution
from mflux.models.common.resolution.prequantized_resolution import PrequantizedResolution
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
from mflux.models.iris.model.iris_scheduler.iris_few_step_scheduler import IrisFewStepScheduler
from mflux.models.iris.model.iris_transformer.iris_transformer import IrisTransformer
from mflux.models.iris.weights.iris_lora_mapping import IrisLoRAMapping
from mflux.models.iris.weights.iris_weight_definition import IrisTurboWeightDefinition, IrisWeightDefinition


class IrisInitializer:
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
        definition = IrisInitializer.weight_definition(model_config)
        saved_path = IrisInitializer.saved_path(model_config, model_path, quantize, bool(lora_paths), bake_lora)
        IrisInitializer._init_config(model, model_config)
        weights = IrisInitializer._load_weights(definition, model_config, saved_path)
        IrisInitializer._init_tokenizers(model, definition, saved_path)
        IrisInitializer._init_models(model)
        IrisInitializer._apply_weights(model, weights, quantize, definition)
        IrisInitializer._apply_loras(model, weights, lora_paths, lora_scales, bake_lora)

    @staticmethod
    def saved_path(
        model_config: ModelConfig,
        model_path: str | None,
        quantize: int | None = None,
        has_loras: bool = False,
        bake_lora: bool = False,
    ) -> str | None:
        # The Iris repo id (what a metadata sidecar records as the model) means the upstream checkpoints. Any other
        # path is a saved model, even when it is the model_name of a config that mflux-save built for that path, or
        # when it only arrives as that model_name (Iris(model_config=...) without model_path).
        upstream_repo = model_config.base_model or model_config.model_name
        candidate = PrequantizedResolution.weights_root(model_config, model_path, quantize, has_loras, bake_lora)
        return None if candidate == upstream_repo else candidate

    @staticmethod
    def weight_definition(model_config: ModelConfig):
        return IrisWeightDefinition if model_config.supports_guidance else IrisTurboWeightDefinition

    @staticmethod
    def _init_config(model, model_config: ModelConfig) -> None:
        model.prompt_cache = {}
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = None

    @staticmethod
    def _load_weights(definition, model_config: ModelConfig, model_path: str | None) -> LoadedWeights:
        if model_path is not None:
            # An mflux-saved model (mflux-save) holds the transformer, the text encoder and the tokenizer.
            return WeightLoader.load(weight_definition=definition, model_path=model_path)
        transformer = WeightLoader.load_single(
            component=definition.source_transformer(),
            repo_id=model_config.model_name,
            file_pattern=definition.SOURCE_TRANSFORMER_PATTERN,
        )
        text_encoder = WeightLoader.load_single(
            component=definition.source_text_encoder(),
            repo_id=IrisWeightDefinition.TEXT_ENCODER_REPO,
            file_pattern=IrisWeightDefinition.TEXT_ENCODER_PATTERN,
        )
        return LoadedWeights(
            components={**transformer.components, **text_encoder.components},
            meta_data=MetaData(),
        )

    @staticmethod
    def _init_tokenizers(model, definition, model_path: str | None) -> None:
        if model_path is not None:
            definitions, path = definition.get_tokenizers(), model_path
        else:
            definitions, path = definition.source_tokenizers(), IrisWeightDefinition.TEXT_ENCODER_REPO
        model.tokenizers = TokenizerLoader.load_all(definitions=definitions, model_path=path)

    @staticmethod
    def _init_models(model) -> None:
        model.transformer = IrisTransformer()
        model.text_encoder = Qwen3TextEncoder(**model.model_config.text_encoder_overrides)

    @staticmethod
    def _apply_loras(model, weights: LoadedWeights, lora_paths, lora_scales, bake_lora: bool) -> None:
        IrisFewStepScheduler.refuse_turbo_svd(LoraResolution.resolve_paths(lora_paths))
        model.lora_paths, model.lora_scales = LoRALoader.load_and_apply_lora(
            lora_mapping=IrisLoRAMapping.get_mapping(),
            transformer=model.transformer,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            bake_lora=bake_lora,
            dense_weights=weights.dense_component("transformer"),
        )
        # A few-step LoRA distilled from the base model switches the base model to its sampler, and says so. Turbo
        # always runs its own 4-step sampler, so there the LoRA changes only the weights
        is_base = model.model_config.supports_guidance
        model.few_step = IrisFewStepScheduler.from_lora_files(model.lora_paths, model.lora_scales) if is_base else None
        if model.few_step is not None:
            steps, shift = model.few_step
            print(f"⏩ Few-step LoRA (turbo_steps={steps}, shift={shift} in its metadata): the base model runs {steps} steps without CFG; --steps still applies, --guidance and --negative-prompt do not.")  # fmt: off

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None, definition) -> None:
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=definition,
            models={"transformer": model.transformer, "text_encoder": model.text_encoder},
        )
