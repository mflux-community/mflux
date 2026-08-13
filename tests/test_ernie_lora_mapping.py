import pytest

from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.ernie_image.weights.ernie_lora_mapping import ErnieLoRAMapping

pytestmark = pytest.mark.fast


class TestErnieLoRAMapping:
    def test_matches_diffusion_model_layer_attention_keys(self):
        keys = [
            "diffusion_model.layers.0.self_attention.to_q.lora_A.weight",
            "diffusion_model.layers.0.self_attention.to_q.lora_B.weight",
            "diffusion_model.layers.3.self_attention.to_k.lora_A.weight",
            "diffusion_model.layers.3.self_attention.to_k.lora_B.weight",
            "diffusion_model.layers.7.self_attention.to_v.lora_A.weight",
            "diffusion_model.layers.7.self_attention.to_v.lora_B.weight",
            "diffusion_model.layers.12.self_attention.to_out.0.lora_A.weight",
            "diffusion_model.layers.12.self_attention.to_out.0.lora_B.weight",
        ]

        assert self._matched_keys(keys) == set(keys)

    def test_matches_transformer_layer_mlp_keys(self):
        keys = [
            "transformer.layers.1.mlp.gate_proj.lora_A.weight",
            "transformer.layers.1.mlp.gate_proj.lora_B.weight",
            "transformer.layers.2.mlp.up_proj.lora_A.weight",
            "transformer.layers.2.mlp.up_proj.lora_B.weight",
            "transformer.layers.4.mlp.linear_fc2.lora_A.weight",
            "transformer.layers.4.mlp.linear_fc2.lora_B.weight",
        ]

        assert self._matched_keys(keys) == set(keys)

    def test_matches_kohya_style_layer_keys(self):
        keys = [
            "lora_unet_layers_0_self_attention_to_q.lora_down.weight",
            "lora_unet_layers_0_self_attention_to_q.lora_up.weight",
            "lora_unet_layers_5_self_attention_to_out_0.lora_down.weight",
            "lora_unet_layers_5_self_attention_to_out_0.lora_up.weight",
        ]

        assert self._matched_keys(keys) == set(keys)

    def test_matches_global_projection_keys(self):
        keys = [
            "transformer.text_proj.lora_A.weight",
            "transformer.text_proj.lora_B.weight",
            "diffusion_model.adaln_modulation.lora_A.weight",
            "diffusion_model.adaln_modulation.lora_B.weight",
            "lora_unet_final_linear.lora_down.weight",
            "lora_unet_final_linear.lora_up.weight",
        ]

        assert self._matched_keys(keys) == set(keys)

    def _matched_keys(self, keys: list[str]) -> set[str]:
        matched_keys: set[str] = set()

        for key in keys:
            for target in ErnieLoRAMapping.get_mapping():
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
