import pytest

from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_text_encoder import Qwen21TextEncoder
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer
from mflux.models.qwen21.weights.qwen21_weight_definition import Qwen21WeightDefinition
from tests.model_saving.tiny_checkpoint_helper import TinyCheckpointRoundtrip, TinyVAEStandIn


class TestTinyQwenImage21ModelSaving:
    @pytest.mark.fast
    def test_tiny_quantized_checkpoint_roundtrips_exactly(self, tmp_path):
        # Qwen-Image-2.1 has no slow save test. This test covers QwenImage21.save_model
        # (Qwen21WeightDefinition), which qwen-image-2.1 and qwen-image-2.1-turbo share. The text
        # encoder has skip_quantization, so it round-trips unquantized beside the quantized
        # transformer.
        TinyCheckpointRoundtrip.save_and_reload_expecting_identical_weights(
            weight_definition=Qwen21WeightDefinition,
            make_components=TestTinyQwenImage21ModelSaving._tiny_components,
            base_path=tmp_path / "qwen_image_21_tiny_q8",
            bits=8,
            # Force shard boundaries so index.json/weight_map multi-shard paths are
            # exercised — the size-based split never shards test-sized tensors.
            tensors_per_shard=8,
        )

    @staticmethod
    def _tiny_components():
        # axes_dims_rope must sum to attention_head_dim (16 + 24 + 24 = 64).
        # mrope_section must sum to half the text head_dim (12 + 10 + 10 = 32).
        return {
            "vae": TinyVAEStandIn(),
            "transformer": Qwen21Transformer(
                num_layers=2,
                attention_head_dim=64,
                num_attention_heads=2,
                context_in_dim=128,
                axes_dims_rope=(16, 24, 24),
            ),
            "text_encoder": Qwen21TextEncoder(
                vocab_size=128,
                hidden_size=128,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                intermediate_size=128,
                head_dim=64,
                mrope_section=[12, 10, 10],
            ),
        }
