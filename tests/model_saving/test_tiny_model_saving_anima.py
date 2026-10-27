import pytest

from mflux.models.anima.model.anima_transformer.anima_transformer import AnimaTransformer
from mflux.models.anima.weights.anima_weight_definition import AnimaWeightDefinition
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
from tests.model_saving.tiny_checkpoint_helper import TinyCheckpointRoundtrip, TinyVAEStandIn


class TestTinyAnimaModelSaving:
    @pytest.mark.fast
    def test_tiny_quantized_checkpoint_roundtrips_exactly(self, tmp_path):
        # Anima has no slow save test. This runs the ModelSaver -> WeightLoader -> WeightApplier seam with the
        # real Anima transformer (its LLM adapter included) and Qwen3 text encoder at tiny dimensions. The text
        # encoder, the adapter and the modulations are saved dense, the rest of the transformer quantized.
        TinyCheckpointRoundtrip.save_and_reload_expecting_identical_weights(
            weight_definition=AnimaWeightDefinition,
            make_components=TestTinyAnimaModelSaving._tiny_components,
            base_path=tmp_path / "anima_tiny_q8",
            bits=8,
            # Force shard boundaries so the index.json / weight_map multi-shard paths are exercised.
            tensors_per_shard=16,
        )

    @staticmethod
    def _tiny_components():
        # Every quantized input width is a multiple of 64 (the group size). The patch embedding (68 inputs) and the
        # text path stay dense through the definition's predicate, so their widths are free.
        return {
            "vae": TinyVAEStandIn(),
            "transformer": AnimaTransformer(dim=128, num_blocks=2, num_heads=2, context_dim=64, adaln_lora_dim=64),
            "text_encoder": Qwen3TextEncoder(
                vocab_size=128,
                hidden_size=64,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=32,
                intermediate_size=128,
            ),
        }
