from pathlib import Path

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.common.resolution.prequantized_resolution import PrequantizedResolution
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.sensenova.model.sensenova_text_encoder.sensenova_text_encoder import SenseNovaTextEncoder
from mflux.models.sensenova.model.sensenova_transformer.sensenova_transformer import SenseNovaTransformer
from mflux.models.sensenova.weights.sensenova_lora_mapping import SenseNovaLoRAMapping
from mflux.models.sensenova.weights.sensenova_weight_definition import SenseNovaWeightDefinition


class SenseNovaInitializer:
    _SAVED_SUBDIRS = ("transformer", "text_encoder", "tokenizer")
    _SHARD_INDEX = "model.safetensors.index.json"

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
        definition = SenseNovaWeightDefinition
        saved_path = SenseNovaInitializer.saved_path(model_config, model_path, quantize, bool(lora_paths), bake_lora)
        source = model_config.model_name
        if saved_path is not None and SenseNovaInitializer.is_upstream_checkpoint(saved_path):
            # A local copy of the upstream checkpoint loads like the upstream repo, from that directory (#993)
            source, saved_path = saved_path, None
        num_layers = SenseNovaInitializer.num_layers(model_config)
        SenseNovaInitializer._init_config(model, model_config)
        weights = SenseNovaInitializer._load_weights(definition, source, saved_path, num_layers)
        SenseNovaInitializer._init_tokenizers(model, definition, source, saved_path)
        SenseNovaInitializer._init_models(model, num_layers)
        SenseNovaInitializer._apply_weights(model, weights, quantize, definition)
        SenseNovaInitializer._apply_loras(model, weights, lora_paths, lora_scales, bake_lora)

    @staticmethod
    def saved_path(
        model_config: ModelConfig,
        model_path: str | None,
        quantize: int | None = None,
        has_loras: bool = False,
        bake_lora: bool = False,
    ) -> str | None:
        # The upstream repo id means the upstream checkpoint; any other path is an mflux-saved model
        upstream_repo = model_config.base_model or model_config.model_name
        candidate = PrequantizedResolution.weights_root(model_config, model_path, quantize, has_loras, bake_lora)
        return None if candidate == upstream_repo else candidate

    @staticmethod
    def is_upstream_checkpoint(path: str) -> bool:
        # Upstream layout: model-*.safetensors shards at the root and none of the mflux-save subdirs
        # Raises FileNotFoundError for that layout when the index or a shard it lists is missing
        root = Path(path).expanduser()
        if not root.is_dir() or not any(shard.is_file() for shard in root.glob("model-*.safetensors")):
            return False
        if any((root / subdir).exists() for subdir in SenseNovaInitializer._SAVED_SUBDIRS):
            return False
        index = SenseNovaInitializer._SHARD_INDEX
        if not (root / index).is_file():
            raise FileNotFoundError(
                f"{root} has model-*.safetensors shards but no {index}; "
                "the upstream checkpoint looks incomplete (copy the whole HuggingFace folder)."
            )
        if not PathResolution._indexed_shards_present(root):
            raise FileNotFoundError(
                f"{index} in {root} cannot be read, has no weight map, or lists a shard that is missing; "
                "the upstream checkpoint looks incomplete (copy the whole HuggingFace folder)."
            )
        return True

    @staticmethod
    def num_layers(model_config: ModelConfig) -> int:
        return model_config.transformer_overrides.get("num_layers", SenseNovaWeightDefinition.NUM_LAYERS)

    @staticmethod
    def _init_config(model, model_config: ModelConfig) -> None:
        model.prompt_cache = {}
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = None

    @staticmethod
    def _load_weights(definition, source: str, model_path: str | None, num_layers: int) -> LoadedWeights:
        if model_path is not None:
            # An mflux-saved model (mflux-save) holds the transformer, the text encoder and the tokenizer.
            return WeightLoader.load(weight_definition=definition, model_path=model_path)
        components = {}
        for component in (definition.source_transformer(num_layers), definition.source_text_encoder(num_layers)):
            loaded = WeightLoader.load_single(
                component=component,
                repo_id=source,
                file_pattern=definition.SOURCE_PATTERN,
            )
            components.update(loaded.components)
        return LoadedWeights(components=components, meta_data=MetaData())

    @staticmethod
    def _init_tokenizers(model, definition, source: str, model_path: str | None) -> None:
        if model_path is not None:
            definitions, path = definition.get_tokenizers(), model_path
        else:
            definitions, path = definition.source_tokenizers(), source
        model.tokenizers = TokenizerLoader.load_all(definitions=definitions, model_path=path)

    @staticmethod
    def _init_models(model, num_layers: int) -> None:
        model.transformer = SenseNovaTransformer(num_layers=num_layers)
        model.text_encoder = SenseNovaTextEncoder(num_layers=num_layers)

    @staticmethod
    def _apply_loras(model, weights: LoadedWeights, lora_paths, lora_scales, bake_lora: bool) -> None:
        model.lora_paths, model.lora_scales = LoRALoader.load_and_apply_lora(
            lora_mapping=SenseNovaLoRAMapping.get_mapping(),
            transformer=model.transformer,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            bake_lora=bake_lora,
            dense_weights=weights.dense_component("transformer"),
        )

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None, definition) -> None:
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=definition,
            models={"transformer": model.transformer, "text_encoder": model.text_encoder},
        )
