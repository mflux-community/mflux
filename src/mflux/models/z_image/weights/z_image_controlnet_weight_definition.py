from typing import List

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.z_image.weights.z_image_controlnet_weight_mapping import ZImageControlnetWeightMapping
from mflux.models.z_image.weights.z_image_weight_definition import ZImageWeightDefinition


class ZImageControlnetWeightDefinition:
    @staticmethod
    def get_controlnet_component() -> ComponentDefinition:
        # Union checkpoints are a single safetensors file in the repo root.
        return ComponentDefinition(
            name="controlnet",
            hf_subdir=".",
            num_blocks=15,
            precision=ModelConfig.precision,
            mapping_getter=ZImageControlnetWeightMapping.get_controlnet_mapping,
        )

    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return ZImageWeightDefinition.get_components() + [
            ComponentDefinition(
                name="controlnet",
                hf_subdir="controlnet",
                num_blocks=15,
                precision=ModelConfig.precision,
                mapping_getter=ZImageControlnetWeightMapping.get_controlnet_mapping,
            )
        ]

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return ZImageWeightDefinition.get_tokenizers()

    @staticmethod
    def get_download_patterns() -> List[str]:
        return ZImageWeightDefinition.get_download_patterns() + [
            "controlnet/*.safetensors",
            "controlnet/*.json",
        ]

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        if not ZImageWeightDefinition.quantization_predicate(path, module):
            return False
        # MLX quantizes in groups of 64 along the last weight axis, so a layer whose last dimension
        # is not a multiple of 64 cannot be quantized. The control patch-embed
        # (control_all_x_embedder, in_features 132) is such a layer; keep it full precision instead
        # of raising during quantization.
        weight = getattr(module, "weight", None)
        if weight is not None and weight.shape[-1] % 64 != 0:
            return False
        return True
