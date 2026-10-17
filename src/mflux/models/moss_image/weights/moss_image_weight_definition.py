from typing import List

from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
from mflux.models.moss_image.weights.moss_image_weight_mapping import (
    MOSSImageWeightMapping,
)


class MOSSImageWeightDefinition:
    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="transformer",
                hf_subdir="transformer",
                loading_mode="multi_glob",
                mapping_getter=MOSSImageWeightMapping.get_transformer_mapping,
            ),
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                loading_mode="single",
            ),
        ]

    @staticmethod
    def get_download_patterns() -> list[str]:
        return [
            "model_index.json",
            "transformer/*",
            "vae/*",
            "text_encoder/*",
            "tokenizer/*",
        ]
