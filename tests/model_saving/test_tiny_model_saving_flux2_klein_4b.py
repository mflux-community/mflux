import pytest

from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer
from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition
from tests.model_saving.tiny_checkpoint_helper import TinyCheckpointRoundtrip, TinyVAEStandIn


class TestTinyFlux2Klein4BModelSaving:
    @pytest.mark.fast
    def test_tiny_quantized_checkpoint_roundtrips_exactly(self, tmp_path):
        # There is no slow twin under tests/model_saving/ for flux.2 klein 4b (no
        # test_model_saving_flux2_klein.py exists) — this is the first checkpoint
        # coverage for it. Same ModelSaver -> WeightLoader -> WeightApplier seam as
        # the other tiny tests, with real Flux2Transformer/Qwen3TextEncoder component
        # classes at tiny dimensions instead of the downloaded checkpoint.
        TinyCheckpointRoundtrip.save_and_reload_expecting_identical_weights(
            weight_definition=Flux2KleinWeightDefinition,
            make_components=TestTinyFlux2Klein4BModelSaving._tiny_components,
            base_path=tmp_path / "flux2_klein_4b_tiny_q8",
            bits=8,
            # Force shard boundaries so index.json/weight_map multi-shard paths are
            # exercised — the size-based split never shards test-sized tensors.
            tensors_per_shard=8,
        )

    @staticmethod
    def _tiny_components():
        # dim=64 (inner_dim = num_attention_heads * attention_head_dim) is the
        # smallest multiple of 64 that keeps every downstream Linear's last dim a
        # multiple of 64 too (mlp_ratio=3.0 keeps the SwiGLU hidden size at 3*dim).
        # axes_dims_rope must sum to attention_head_dim (4 * 8 = 32).
        return {
            "vae": TinyVAEStandIn(),
            "transformer": Flux2Transformer(
                in_channels=64,
                num_layers=2,
                num_single_layers=2,
                attention_head_dim=32,
                num_attention_heads=2,
                joint_attention_dim=128,
                axes_dims_rope=(8, 8, 8, 8),
            ),
            "text_encoder": Qwen3TextEncoder(
                vocab_size=128,
                hidden_size=128,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=64,
                intermediate_size=128,
            ),
        }
