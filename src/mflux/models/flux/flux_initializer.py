from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.depth_pro.model.depth_pro import DepthPro
from mflux.models.flux.model.flux_text_encoder.clip_encoder.clip_encoder import CLIPEncoder
from mflux.models.flux.model.flux_text_encoder.t5_encoder.t5_encoder import T5Encoder
from mflux.models.flux.model.flux_transformer.transformer import Transformer
from mflux.models.flux.model.flux_vae.vae import VAE
from mflux.models.flux.model.redux_encoder.redux_encoder import ReduxEncoder
from mflux.models.flux.model.siglip_vision_transformer.siglip_vision_transformer import SiglipVisionTransformer
from mflux.models.flux.variants.controlnet.transformer_controlnet import TransformerControlnet
from mflux.models.flux.weights.flux_lora_mapping import FluxLoRAMapping
from mflux.models.flux.weights.flux_weight_definition import (
    FluxControlnetWeightDefinition,
    FluxReduxWeightDefinition,
    FluxWeightDefinition,
)


class FluxInitializer:
    @staticmethod
    def init(
        model,
        model_config: ModelConfig,
        quantize: int | None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        bake_lora: bool = True,
        custom_transformer=None,
    ) -> None:
        path = model_path if model_path else model_config.model_name
        FluxInitializer._init_config(model, model_config)
        weights = FluxInitializer._load_weights(path)
        FluxInitializer._init_tokenizers(model, path, model_config)
        FluxInitializer._init_models(model, model_config, weights, custom_transformer)
        FluxInitializer._apply_weights(model, weights, quantize)
        FluxInitializer._apply_lora(model, lora_paths, lora_scales, bake_lora)

    @staticmethod
    def init_depth(
        model,
        model_config: ModelConfig,
        quantize: int | None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        bake_lora: bool = True,
    ) -> None:
        FluxInitializer.init(
            model=model,
            model_config=model_config,
            quantize=quantize,
            model_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            bake_lora=bake_lora,
        )
        model.depth_pro = DepthPro()

    @staticmethod
    def init_redux(
        model,
        model_config: ModelConfig,
        quantize: int | None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        bake_lora: bool = True,
    ) -> None:
        FluxInitializer.init(
            model=model,
            quantize=quantize,
            model_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            bake_lora=bake_lora,
            model_config=model_config,
        )

        redux_weights = WeightLoader.load(
            weight_definition=FluxReduxWeightDefinition,
            model_path=ModelConfig.dev_redux().model_name,
        )
        model.image_embedder = ReduxEncoder()
        model.image_encoder = SiglipVisionTransformer()
        WeightApplier.apply_and_quantize(
            weights=redux_weights,
            quantize_arg=quantize,
            weight_definition=FluxReduxWeightDefinition,
            models={
                "siglip": model.image_encoder,
                "redux_encoder": model.image_embedder,
            },
        )

    @staticmethod
    def init_controlnet(
        model,
        model_config: ModelConfig,
        quantize: int | None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        bake_lora: bool = True,
    ) -> None:
        FluxInitializer.init(
            model=model,
            model_config=model_config,
            quantize=quantize,
            model_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            bake_lora=bake_lora,
        )

        controlnet_component = FluxControlnetWeightDefinition.get_controlnet_component()
        controlnet_weights = WeightLoader.load_single(
            component=controlnet_component,
            repo_id=model_config.controlnet_model,
        )
        model.transformer_controlnet = TransformerControlnet(
            model_config=model_config,
            num_transformer_blocks=controlnet_weights.num_transformer_blocks(),
            num_single_transformer_blocks=controlnet_weights.num_single_transformer_blocks(),
        )
        WeightApplier.apply_and_quantize_single(
            weights=controlnet_weights,
            model=model.transformer_controlnet,
            component=controlnet_component,
            quantize_arg=quantize,
            quantization_predicate=FluxWeightDefinition.quantization_predicate,
        )

    @staticmethod
    def init_concept(
        model,
        model_config: ModelConfig,
        quantize: int | None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        bake_lora: bool = True,
    ) -> None:
        from mflux.models.flux.variants.concept_attention.transformer_concept import TransformerConcept

        path = model_path if model_path else model_config.model_name
        FluxInitializer._init_config(model, model_config)
        weights = FluxInitializer._load_weights(path)
        FluxInitializer._init_tokenizers(model, path, model_config)
        custom_transformer = TransformerConcept(
            model_config=model_config,
            num_transformer_blocks=weights.num_transformer_blocks(),
            num_single_transformer_blocks=weights.num_single_transformer_blocks(),
        )
        FluxInitializer._init_models(model, model_config, weights, custom_transformer)
        FluxInitializer._apply_weights(model, weights, quantize)
        FluxInitializer._apply_lora(model, lora_paths, lora_scales, bake_lora)

    @staticmethod
    def _init_config(model, model_config: ModelConfig) -> None:
        model.prompt_cache = {}
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = None

    @staticmethod
    def _load_weights(model_path: str) -> LoadedWeights:
        return WeightLoader.load(
            weight_definition=FluxWeightDefinition,
            model_path=model_path,
        )

    @staticmethod
    def _init_tokenizers(model, model_path: str, model_config: ModelConfig) -> None:
        max_length_overrides = (
            {"t5": model_config.max_sequence_length} if model_config.max_sequence_length is not None else {}
        )
        model.tokenizers = TokenizerLoader.load_all(
            definitions=FluxWeightDefinition.get_tokenizers(),
            model_path=model_path,
            max_length_overrides=max_length_overrides,
        )

    @staticmethod
    def _init_models(
        model,
        model_config: ModelConfig,
        weights: LoadedWeights,
        custom_transformer=None,
    ) -> None:
        model.vae = VAE()
        model.t5_text_encoder = T5Encoder()
        model.clip_text_encoder = CLIPEncoder()
        if custom_transformer is not None:
            model.transformer = custom_transformer
        else:
            model.transformer = Transformer(
                model_config=model_config,
                num_transformer_blocks=weights.num_transformer_blocks(),
                num_single_transformer_blocks=weights.num_single_transformer_blocks(),
            )

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None) -> None:
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=FluxWeightDefinition,
            models={
                "vae": model.vae,
                "transformer": model.transformer,
                "t5_encoder": model.t5_text_encoder,
                "clip_encoder": model.clip_text_encoder,
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
            lora_mapping=FluxLoRAMapping.get_mapping(),
            transformer=model.transformer,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            bake_lora=bake_lora,
        )
