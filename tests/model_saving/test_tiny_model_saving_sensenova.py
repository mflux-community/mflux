import mlx.core as mx
import pytest

from mflux.models.sensenova.model.sensenova_text_encoder.sensenova_text_encoder import SenseNovaTextEncoder
from mflux.models.sensenova.model.sensenova_transformer.sensenova_transformer import SenseNovaTransformer
from mflux.models.sensenova.weights.sensenova_weight_definition import SenseNovaWeightDefinition
from tests.model_saving.tiny_checkpoint_helper import TinyCheckpointRoundtrip


class TestTinySenseNovaModelSaving:
    # There is no slow save/load twin for SenseNova-U1.5; this is its checkpoint coverage. The same
    # ModelSaver -> WeightLoader -> WeightApplier seam as the other tiny tests, on the real transformer and
    # text encoder classes at tiny widths.
    @pytest.mark.fast
    @pytest.mark.parametrize("bits", [4, 8])
    def test_tiny_quantized_checkpoint_roundtrips_exactly(self, tmp_path, bits):
        TinyCheckpointRoundtrip.save_and_reload_expecting_identical_weights(
            weight_definition=SenseNovaWeightDefinition,
            make_components=TestTinySenseNovaModelSaving._tiny_components,
            base_path=tmp_path / f"sensenova_tiny_q{bits}",
            bits=bits,
            # Force shard boundaries so the index.json / weight_map multi-shard paths are exercised.
            tensors_per_shard=16,
        )

    @pytest.mark.fast
    def test_the_saved_layout_is_transformer_and_text_encoder_directories(self, tmp_path):
        from mflux.models.common.weights.saving.model_saver import ModelSaver

        ModelSaver.save_model(
            model=TinyCheckpointRoundtrip._as_model(
                TestTinySenseNovaModelSaving._tiny_components(), SenseNovaWeightDefinition
            ),
            bits=None,
            base_path=str(tmp_path),
            weight_definition=SenseNovaWeightDefinition,
        )

        assert sorted(path.name for path in tmp_path.iterdir()) == ["text_encoder", "transformer"]

    @pytest.mark.fast
    def test_the_conditioning_embedders_stay_dense_in_a_quantized_save(self, tmp_path):
        components = TestTinySenseNovaModelSaving._tiny_components()
        TinyCheckpointRoundtrip._quantize(components, SenseNovaWeightDefinition, 8)

        transformer = components["transformer"]
        assert not hasattr(transformer.timestep_embedder.mlp[0], "scales")
        assert not hasattr(transformer.noise_scale_embedder.mlp[2], "scales")
        assert hasattr(transformer.layers[0].self_attn.q_proj, "scales")

    @pytest.mark.fast
    def test_the_tiny_transformer_runs_a_forward_pass_on_a_prefix(self):
        transformer = TestTinySenseNovaModelSaving._tiny_components()["transformer"]
        prefix = [(mx.zeros((1, 8, 3, 128)), mx.zeros((1, 8, 3, 128))) for _ in transformer.layers]

        out = transformer(mx.zeros((1, 3, 64, 96)), mx.array([0.5]), 1.0, prefix, 3)

        assert out.shape == (1, 3, 64, 96)

    @staticmethod
    def _tiny_components():
        # The attention keeps its 32 / 8 / 128 head layout; every width is a multiple of the quantization group size (64).
        return {
            "transformer": SenseNovaTransformer(num_layers=2, hidden_size=128, intermediate_size=128),
            "text_encoder": SenseNovaTextEncoder(num_layers=2, vocab_size=128, hidden_size=128, intermediate_size=128),
        }
