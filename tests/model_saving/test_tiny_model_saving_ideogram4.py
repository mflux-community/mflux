import pytest

from mflux.models.ideogram4.model import Ideogram4Config, Ideogram4Transformer, Qwen3TextEncoder
from mflux.models.ideogram4.weights.ideogram4_weight_definition import Ideogram4WeightDefinition
from tests.model_saving.tiny_checkpoint_helper import TinyCheckpointRoundtrip, TinyVAEStandIn

# Every dimension is a multiple of 64 (MLX's quantization group size) so the
# tiny components quantize exactly like the full-size checkpoint does.
TINY_TRANSFORMER_CONFIG = Ideogram4Config(
    emb_dim=128,
    num_layers=2,
    num_heads=2,
    intermediate_size=128,
    adanln_dim=64,
    in_channels=64,
    llm_features_dim=128,
)


class TestTinyIdeogram4ModelSaving:
    @pytest.mark.fast
    def test_tiny_quantized_checkpoint_roundtrips_exactly(self, tmp_path):
        # Fast twin of tests/model_saving/test_model_saving_ideogram4.py: the same
        # ModelSaver -> WeightLoader -> WeightApplier seam, with real Ideogram 4
        # component classes (incl. Fp8Linear layers) at tiny dimensions instead of
        # the downloaded 26 GB checkpoint.
        TinyCheckpointRoundtrip.save_and_reload_expecting_identical_weights(
            weight_definition=Ideogram4WeightDefinition,
            make_components=TestTinyIdeogram4ModelSaving._tiny_components,
            base_path=tmp_path / "ideogram4_tiny_q8",
            bits=8,
            # Force shard boundaries so index.json/weight_map multi-shard paths are
            # exercised — the size-based split never shards test-sized tensors.
            tensors_per_shard=8,
        )

    @staticmethod
    def _tiny_components():
        return {
            "vae": TinyVAEStandIn(),
            "conditional_transformer": Ideogram4Transformer(TINY_TRANSFORMER_CONFIG),
            "unconditional_transformer": Ideogram4Transformer(TINY_TRANSFORMER_CONFIG),
            "text_encoder": Qwen3TextEncoder(
                vocab_size=128,
                hidden_size=128,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                intermediate_size=128,
                head_dim=64,
            ),
        }
