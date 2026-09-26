from typing import List

import mlx.core as mx

from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
from mflux.models.common.weights.mapping.weight_mapping import WeightTarget
from mflux.models.qwen21.weights.qwen21_weight_definition import Qwen21WeightDefinition
from mflux.models.qwen21.weights.qwen21_weight_mapping import Qwen21WeightMapping


class Qwen21EditWeightDefinition(Qwen21WeightDefinition):
    # Same components as t2i, but the text encoder also maps the Qwen3-VL vision tower
    # (model.visual.*) that lives inside the text_encoder shards and is required to
    # encode condition images for editing.

    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                loading_mode="single",
                mapping_getter=Qwen21WeightMapping.get_vae_mapping,
            ),
            ComponentDefinition(
                name="transformer",
                hf_subdir="transformer",
                loading_mode="multi_glob",
                mapping_getter=Qwen21WeightMapping.get_transformer_mapping,
            ),
            ComponentDefinition(
                name="text_encoder",
                hf_subdir="text_encoder",
                loading_mode="multi_json",
                precision=mx.bfloat16,
                skip_quantization=True,  # Quantization causes significant semantic degradation
                mapping_getter=Qwen21EditWeightDefinition.get_text_encoder_mapping_with_visual,
            ),
        ]

    @staticmethod
    def get_text_encoder_mapping_with_visual() -> List[WeightTarget]:
        return Qwen21WeightMapping.get_text_encoder_mapping() + Qwen21WeightMapping.get_text_encoder_visual_mapping()
