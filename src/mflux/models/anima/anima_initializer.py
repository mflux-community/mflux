from pathlib import Path

import mlx.core as mx

import mflux.models.anima.model.anima_scheduler  # noqa: F401 - registers the anima scheduler
from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.anima.model.anima_transformer.anima_transformer import AnimaTransformer
from mflux.models.anima.weights.anima_lora_mapping import AnimaLoRAMapping
from mflux.models.anima.weights.anima_weight_definition import AnimaWeightDefinition
from mflux.models.common.compute_precision import ComputePrecision
from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE

# Qwen3-0.6B-Base, the text encoder Anima ships.
QWEN3_06B_CONFIG = {
    "hidden_size": 1024,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "intermediate_size": 3072,
    "max_position_embeddings": 32768,
    "head_dim": 128,
}


class AnimaInitializer:
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
        if quantize == 3:
            raise ValueError("Anima does not work with -q 3, the 2B transformer turns to noise. Use -q 4, 5, 6 or 8.")
        precision = ComputePrecision(compute_precision)
        path = model_path or model_config.model_name
        AnimaInitializer._init_config(model, model_config)
        weights = AnimaInitializer._load_weights_and_tokenizers(model, model_config, path)
        AnimaInitializer._init_models(model)
        AnimaInitializer._apply_weights(model, weights, quantize)
        model.lora_paths, model.lora_scales = LoRALoader.load_and_apply_lora(
            lora_mapping=AnimaLoRAMapping.get_mapping(),
            transformer=model.transformer,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            bake_lora=bake_lora,
            dense_weights=weights.dense_component("transformer"),
        )
        del weights
        mx.eval(model.parameters())
        mx.clear_cache()
        model.compute_precision = precision
        if compute_precision is not None:
            # Last, so it casts the final parameters, whatever quantization and LoRA produced.
            model.transformer.apply_compute_precision(precision)

    @staticmethod
    def is_upstream(path: str) -> bool:
        # The ComfyUI single files: the model repository or a local copy of it. Anything else is an mflux-save.
        return path == AnimaWeightDefinition.REPO or AnimaWeightDefinition.is_repository_copy(Path(path).expanduser())

    @staticmethod
    def _init_config(model, model_config: ModelConfig) -> None:
        model.prompt_cache = {}
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = None

    @staticmethod
    def _load_weights_and_tokenizers(model, model_config: ModelConfig, path: str) -> LoadedWeights:
        if not AnimaInitializer.is_upstream(path):
            weights = WeightLoader.load(weight_definition=AnimaWeightDefinition, model_path=path)
            model.tokenizers = TokenizerLoader.load_all(definitions=AnimaWeightDefinition.get_tokenizers(), model_path=path)  # fmt: off
            return weights
        weights = AnimaInitializer._load_upstream_weights(model_config, path)
        # The repository has no tokenizers; they come from the Qwen3 and T5 repositories.
        model.tokenizers = {}
        for repo, definitions in AnimaWeightDefinition.source_tokenizers().items():
            model.tokenizers.update(TokenizerLoader.load_all(definitions=definitions, model_path=repo))
        return weights

    @staticmethod
    def _load_upstream_weights(model_config: ModelConfig, path: str) -> LoadedWeights:
        # Every variant is a single file in the repository; only the files of this one are read or downloaded.
        components = {}
        for component in (
            AnimaWeightDefinition.source_vae(),
            AnimaWeightDefinition.source_transformer(model_config.aliases[0]),
            AnimaWeightDefinition.source_text_encoder(),
        ):
            if path == AnimaWeightDefinition.REPO:
                pattern = f"{component.hf_subdir}/{component.weight_files[0]}"
                loaded = WeightLoader.load_single(component=component, repo_id=path, file_pattern=pattern)
            else:
                loaded = WeightLoader.load_single_local(component=component, root_path=Path(path).expanduser())
            components.update(loaded.components)
        return LoadedWeights(components=components, meta_data=MetaData())

    @staticmethod
    def _init_models(model) -> None:
        model.vae = QwenVAE()
        model.transformer = AnimaTransformer()
        model.text_encoder = Qwen3TextEncoder(**QWEN3_06B_CONFIG)

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None) -> None:
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=AnimaWeightDefinition,
            models={
                "vae": model.vae,
                "transformer": model.transformer,
                "text_encoder": model.text_encoder,
            },
        )
