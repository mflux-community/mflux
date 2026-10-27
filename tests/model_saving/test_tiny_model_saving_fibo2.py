import pytest

from mflux.models.fibo2.model.fibo2_text_encoder.fibo2_text_encoder import Fibo2TextEncoder
from mflux.models.fibo2.model.fibo2_transformer.fibo2_transformer import Fibo2Transformer
from mflux.models.fibo2.weights.fibo2_weight_definition import Fibo2WeightDefinition
from tests.model_saving.tiny_checkpoint_helper import TinyCheckpointRoundtrip, TinyVAEStandIn


class TestTinyFibo2ModelSaving:
    # There is no slow save/load twin for FIBO 2; this is its checkpoint coverage. The same
    # ModelSaver -> WeightLoader -> WeightApplier seam as the other tiny tests, on real Fibo2Transformer and
    # Fibo2TextEncoder (Qwen3-VL language model and vision tower) classes at tiny dimensions.
    @pytest.mark.fast
    def test_tiny_quantized_checkpoint_roundtrips_exactly(self, tmp_path):
        TinyCheckpointRoundtrip.save_and_reload_expecting_identical_weights(
            weight_definition=Fibo2WeightDefinition,
            make_components=TestTinyFibo2ModelSaving._tiny_components,
            base_path=tmp_path / "fibo2_tiny_q8",
            bits=8,
            # Force shard boundaries so the index.json / weight_map multi-shard paths are exercised.
            tensors_per_shard=8,
        )

    @staticmethod
    def _tiny_components():
        # dim 128 over 2 heads gives head_dim 64, which the transformer's axes_dims and twice the text model's
        # mrope_section must sum to. The FFN hidden size (128 * 8 // 3) // 64 * 64 = 320 is a multiple of the
        # quantization group size. Layer 1 is an injection layer.
        text_config = {
            **Fibo2TextEncoder.TEXT_CONFIG,
            "vocab_size": 128,
            "hidden_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 64,
            "intermediate_size": 128,
            "rope_scaling": {"mrope_section": [12, 10, 10]},
        }
        vision_config = {
            **Fibo2TextEncoder.VISION_CONFIG,
            "depth": 2,
            "hidden_size": 128,
            "intermediate_size": 128,
            "num_heads": 2,
            "out_hidden_size": 128,
            "num_position_embeddings": 64,
            "deepstack_visual_indexes": [0, 1],
        }
        return {
            "vae": TinyVAEStandIn(),
            "transformer": Fibo2Transformer(
                dim=128,
                n_heads=2,
                n_layers=2,
                n_refiner_layers=1,
                cap_feat_dim=128,
                injection_layer_ids=(1,),
                perceiver_num_layers=1,
                axes_dims=(16, 24, 24),
            ),
            "text_encoder": Fibo2TextEncoder(text_config, vision_config),
        }
