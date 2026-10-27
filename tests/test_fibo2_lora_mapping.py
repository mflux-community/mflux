import pytest
from mlx import nn

from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.fibo2.model.fibo2_transformer.fibo2_transformer import Fibo2Transformer
from mflux.models.fibo2.weights.fibo2_lora_mapping import Fibo2LoRAMapping

pytestmark = pytest.mark.fast

# The MLX names that differ from the diffusers names a LoRA is saved with
DIFFUSERS_NAMES = {
    "t_embedder.linear1": "t_embedder.mlp.0",
    "t_embedder.linear2": "t_embedder.mlp.2",
    "final_layer.adaLN_modulation.0": "final_layer.adaLN_modulation.1",
}


class TestFibo2LoRAMapping:
    def test_every_linear_layer_is_reached_from_its_diffusers_key(self):
        # diffusers' FIBO 2 trainer adapts every linear layer ("all-linear"); each must land on its own MLX module
        for path in TestFibo2LoRAMapping._linear_paths():
            source = DIFFUSERS_NAMES.get(path, path)
            for prefix, suffix in (("transformer.", ".lora_A.weight"), ("", ".lora_B.default.weight")):
                assert TestFibo2LoRAMapping._resolve(f"{prefix}{source}{suffix}") == path

    def test_comfyui_style_keys_are_accepted(self):
        key = "diffusion_model.layers.6.cross_attention.to_q.lora_B.weight"
        assert TestFibo2LoRAMapping._resolve(key) == "layers.6.cross_attention.to_q"

    @staticmethod
    def _linear_paths() -> list[str]:
        # The full-size layout: 44 layers, 5 of them with cross-attention, 2 refiner and 2 Perceiver layers
        transformer = Fibo2Transformer()
        return [name for name, module in transformer.named_modules() if isinstance(module, nn.Linear)]

    @staticmethod
    def _resolve(key: str) -> str | None:
        for target in Fibo2LoRAMapping.get_mapping():
            for pattern in target.possible_up_patterns + target.possible_down_patterns:
                block = LoRALoader._match_pattern(key, pattern)
                if block is not None:
                    return target.model_path.format(block=block)
        return None
