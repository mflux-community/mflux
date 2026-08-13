import pytest

from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.ideogram4.weights.ideogram4_lora_mapping import Ideogram4LoRAMapping

pytestmark = pytest.mark.fast


class TestIdeogram4LoRAMapping:
    def test_matches_transformer_layer_attention_keys(self):
        keys = [
            "transformer.layers.0.attention.qkv.lora_A.weight",
            "transformer.layers.0.attention.qkv.lora_B.weight",
            "transformer.layers.3.attention.o.lora_A.weight",
            "transformer.layers.3.attention.o.lora_B.weight",
        ]

        assert self._matched_keys(keys) == set(keys)

    def test_matches_diffusion_model_layer_keys(self):
        keys = [
            "diffusion_model.layers.1.adaln_modulation.lora_A.weight",
            "diffusion_model.layers.1.adaln_modulation.lora_B.weight",
            "diffusion_model.layers.2.feed_forward.w1.lora_A.weight",
            "diffusion_model.layers.2.feed_forward.w1.lora_B.weight",
        ]

        assert self._matched_keys(keys) == set(keys)

    def test_matches_global_projection_keys(self):
        keys = [
            "transformer.input_proj.lora_A.weight",
            "transformer.input_proj.lora_B.weight",
            "transformer.final_layer.linear.lora_A.weight",
            "transformer.final_layer.linear.lora_B.weight",
            "lora_unet_final_linear.lora_down.weight",
            "lora_unet_final_linear.lora_up.weight",
        ]

        assert self._matched_keys(keys) == set(keys)

    def _matched_keys(self, keys: list[str]) -> set[str]:
        matched_keys: set[str] = set()

        for key in keys:
            for target in Ideogram4LoRAMapping.get_mapping():
                pattern_groups = (
                    target.possible_up_patterns,
                    target.possible_down_patterns,
                    target.possible_alpha_patterns,
                )
                for patterns in pattern_groups:
                    if any(LoRALoader._match_pattern(key, pattern) is not None for pattern in patterns):
                        matched_keys.add(key)
                        break
                if key in matched_keys:
                    break

        return matched_keys
