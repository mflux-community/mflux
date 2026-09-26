from dataclasses import replace
from typing import List

from mflux.models.common.weights.mapping.weight_mapping import WeightTarget
from mflux.models.z_image.weights.z_image_weight_mapping import ZImageWeightMapping


class MingImageWeightMapping:
    @staticmethod
    def get_transformer_mapping() -> List[WeightTarget]:
        # Z-Image's layout without pad tokens: Ming drops the alignment padding before the transformer.
        return [
            replace(target, required=False) if target.to_pattern in ("x_pad_token", "cap_pad_token") else target
            for target in ZImageWeightMapping.get_transformer_mapping()
        ]
