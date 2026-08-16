import pytest

from mflux.models.ernie_image.model.ernie_text_encoder.text_encoder import ErnieMistralTextEncoder
from mflux.models.ernie_image.model.ernie_transformer.transformer import ErnieTransformer
from mflux.models.ernie_image.weights.ernie_weight_definition import ErnieWeightDefinition
from tests.model_saving.tiny_checkpoint_helper import TinyCheckpointRoundtrip, TinyVAEStandIn


class TestTinyErnieModelSaving:
    @pytest.mark.fast
    def test_tiny_quantized_checkpoint_roundtrips_exactly(self, tmp_path):
        # Fast twin of tests/model_saving/test_model_saving_ernie_image.py: the same
        # ModelSaver -> WeightLoader -> WeightApplier seam, with real ERNIE component
        # classes at tiny dimensions instead of the downloaded full-size checkpoint.
        TinyCheckpointRoundtrip.save_and_reload_expecting_identical_weights(
            weight_definition=ErnieWeightDefinition,
            make_components=TestTinyErnieModelSaving._tiny_components,
            base_path=tmp_path / "ernie_tiny_q8",
            bits=8,
            # Force shard boundaries so index.json/weight_map multi-shard paths are
            # exercised — the size-based split never shards test-sized tensors.
            tensors_per_shard=8,
        )

    @staticmethod
    def _tiny_components():
        # Every dimension is a multiple of 64 (MLX's quantization group size) so the
        # tiny components quantize exactly like the full-size checkpoint does.
        # rope_axes_dim must sum to head_dim (128 / 2 heads = 64).
        return {
            "vae": TinyVAEStandIn(),
            "transformer": ErnieTransformer(
                hidden_size=128,
                num_attention_heads=2,
                num_layers=2,
                ffn_hidden_size=128,
                in_channels=64,
                out_channels=64,
                text_in_dim=64,
                rope_axes_dim=[16, 24, 24],
            ),
            "text_encoder": ErnieMistralTextEncoder(
                vocab_size=128,
                hidden_size=128,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=64,
                intermediate_size=128,
            ),
        }
