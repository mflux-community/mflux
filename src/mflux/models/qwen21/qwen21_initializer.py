from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.resolution.lora_resolution import LoraResolution
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_text_encoder import Qwen21TextEncoder
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer
from mflux.models.qwen21.model.qwen21_vae.qwen21_vae import Qwen21VAE
from mflux.models.qwen21.pdd_loader import Qwen21PDDLoader
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
        Qwen21Initializer._init_models(model, weights)
        Qwen21Initializer._apply_weights(model, weights, quantize)
        Qwen21Initializer._apply_lora(model, lora_paths, lora_scales, bake_lora)

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
    def _init_models(model, weights: LoadedWeights) -> None:
        model.vae = Qwen21VAE()
        model.transformer = Qwen21Transformer()
        transformer_weights = weights.components.get("transformer", {})
        pdd_weights = transformer_weights.get("pdd_proj_out_weights")
        pdd_sigmas = transformer_weights.get("pdd_sigmas")
        if (pdd_weights is None) != (pdd_sigmas is None):
            raise ValueError("Saved PDD checkpoints must contain both output heads and their sigma grid.")
        if pdd_weights is not None:
            model.transformer.enable_pdd(pdd_weights, pdd_sigmas)
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

    @staticmethod
    def _apply_lora(model, lora_paths, lora_scales, bake_lora: bool) -> None:
        resolved_paths = LoraResolution.resolve_paths(lora_paths)
        if len(resolved_paths) == 1 and Qwen21PDDLoader.is_pdd_adapter(resolved_paths[0]):
            resolved_scales = LoraResolution.resolve_scales(lora_scales, 1)
            Qwen21PDDLoader.load_and_apply(model.transformer, resolved_paths[0], resolved_scales[0], bake_lora)
            model.lora_paths, model.lora_scales = resolved_paths, resolved_scales
            return
        if any(Qwen21PDDLoader.is_pdd_adapter(path) for path in resolved_paths):
            raise ValueError("Qwen Image 2.1 PDD adapters cannot be combined with other LoRA files.")
        model.lora_paths, model.lora_scales = LoRALoader.load_and_apply_lora(
            lora_mapping=Qwen21LoRAMapping.get_mapping(),
            transformer=model.transformer,
            lora_paths=resolved_paths,
            lora_scales=lora_scales,
            bake_lora=bake_lora,
        )
