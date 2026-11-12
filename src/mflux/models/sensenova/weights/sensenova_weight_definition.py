import re
from typing import List

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.tokenizer import LanguageTokenizer
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.sensenova.weights.sensenova_weight_mapping import SenseNovaWeightMapping


class SenseNovaWeightDefinition:
    # The mflux layout (what mflux-save writes and model_path reads): transformer/, text_encoder/, tokenizer/.
    # The upstream checkpoint is one set of shards holding both streams; source_* reads the gen stream (transformer)
    # from them and, in the text encoder, the und stream.
    NUM_LAYERS = 42
    SOURCE_PATTERN = "model-*.safetensors"
    _LAYER_KEY = re.compile(r"^language_model\.model\.layers\.(\d+)\.")
    _GEN_PREFIXES = (
        "fm_modules.fm_head.",
        "fm_modules.timestep_embedder.",
        "fm_modules.noise_scale_embedder.",
        "fm_modules.vision_model_mot_gen.",
        "language_model.model.norm_mot_gen.",
    )

    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(name="transformer", hf_subdir="transformer", precision=ModelConfig.precision),
            ComponentDefinition(name="text_encoder", hf_subdir="text_encoder", precision=ModelConfig.precision),
        ]

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return [
            TokenizerDefinition(
                name="sensenova",
                hf_subdir="tokenizer",
                tokenizer_class="Qwen2Tokenizer",
                encoder_class=LanguageTokenizer,
                download_patterns=["tokenizer/*"],
            ),
        ]

    @staticmethod
    def source_transformer(num_layers: int = NUM_LAYERS) -> ComponentDefinition:
        return ComponentDefinition(
            name="transformer",
            hf_subdir="",
            precision=ModelConfig.precision,
            mapping_getter=SenseNovaWeightMapping.get_transformer_mapping,
            num_layers=num_layers,
            key_transform=lambda key: SenseNovaWeightDefinition._gen_key(key, num_layers),
        )

    @staticmethod
    def source_text_encoder(num_layers: int = NUM_LAYERS) -> ComponentDefinition:
        return ComponentDefinition(
            name="text_encoder",
            hf_subdir="",
            precision=ModelConfig.precision,
            mapping_getter=SenseNovaWeightMapping.get_text_encoder_mapping,
            num_layers=num_layers,
            key_transform=lambda key: SenseNovaWeightDefinition._und_key(key, num_layers),
        )

    @staticmethod
    def source_tokenizers() -> List[TokenizerDefinition]:
        return [
            TokenizerDefinition(
                name="sensenova",
                hf_subdir="",
                tokenizer_class="Qwen2Tokenizer",
                encoder_class=LanguageTokenizer,
                download_patterns=[
                    "tokenizer*",
                    "vocab.json",
                    "merges.txt",
                    "added_tokens.json",
                    "special_tokens_map.json",
                ],  # fmt: skip
            ),
        ]

    @staticmethod
    def get_download_patterns() -> List[str]:
        return [
            "transformer/*.safetensors",
            "transformer/*.json",
            "text_encoder/*.safetensors",
            "text_encoder/*.json",
            "tokenizer/*",
        ]

    @staticmethod
    def quantization_predicate(path: str, module, bits: int | None = None) -> bool:
        if not hasattr(module, "to_quantized"):
            return False
        # The conditioning embedders are tiny and set the whole image's level of noise, so they stay dense
        if path.startswith(("timestep_embedder", "noise_scale_embedder")):
            return False
        weight = getattr(module, "weight", None)
        return weight is None or weight.shape[-1] % 64 == 0

    @staticmethod
    def _gen_key(key: str, num_layers: int) -> str | None:
        match = SenseNovaWeightDefinition._LAYER_KEY.match(key)
        if match:
            return key if int(match.group(1)) < num_layers and "_mot_gen" in key else None
        return key if key.startswith(SenseNovaWeightDefinition._GEN_PREFIXES) else None

    @staticmethod
    def _und_key(key: str, num_layers: int) -> str | None:
        if key == "language_model.model.embed_tokens.weight":
            return key
        match = SenseNovaWeightDefinition._LAYER_KEY.match(key)
        return key if match and int(match.group(1)) < num_layers and "_mot_gen" not in key else None


class SenseNovaUpstreamWeightDefinition:
    # What mflux-check validates: the upstream shards at the repository root, read through the stream key filters.
    # An mflux-saved copy still passes, because the checker looks in the save subdirs (transformer/, text_encoder/) first.
    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [SenseNovaWeightDefinition.source_transformer(), SenseNovaWeightDefinition.source_text_encoder()]
