import pytest

from mflux.models.z_image.model.z_image_text_encoder.text_encoder import TextEncoder
from mflux.models.z_image.model.z_image_transformer.transformer import ZImageTransformer
from mflux.models.z_image.weights.z_image_weight_definition import ZImageWeightDefinition
from tests.model_saving.tiny_checkpoint_helper import TinyCheckpointRoundtrip, TinyVAEStandIn


class TestTinyZImageTurboModelSaving:
    @pytest.mark.fast
    def test_tiny_quantized_checkpoint_roundtrips_exactly(self, tmp_path):
        # Fast twin of tests/model_saving/test_model_saving.py: the same
        # ModelSaver -> WeightLoader -> WeightApplier seam, with real Z-Image Turbo
        # component classes at tiny dimensions instead of the downloaded checkpoint.
        TinyCheckpointRoundtrip.save_and_reload_expecting_identical_weights(
            weight_definition=ZImageWeightDefinition,
            make_components=TestTinyZImageTurboModelSaving._tiny_components,
            base_path=tmp_path / "z_image_turbo_tiny_q8",
            bits=8,
            # Force shard boundaries so index.json/weight_map multi-shard paths are
            # exercised — the size-based split never shards test-sized tensors.
            tensors_per_shard=8,
        )

    @staticmethod
    def _tiny_components():
        # dim=192 is the smallest multiple of 64 for which the feed-forward hidden
        # size (int(dim / 3 * 8)) is also a multiple of 64 — both are quantized last
        # dims. rope_axes_dim must sum to head_dim (192 / 3 heads = 64).
        return {
            "vae": TinyVAEStandIn(),
            "transformer": ZImageTransformer(
                dim=192,
                n_layers=2,
                n_refiner_layers=1,
                n_heads=3,
                cap_feat_dim=128,
                axes_dims=[16, 24, 24],
            ),
            "text_encoder": TextEncoder(
                vocab_size=128,
                hidden_size=128,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=64,
                intermediate_size=128,
            ),
        }
