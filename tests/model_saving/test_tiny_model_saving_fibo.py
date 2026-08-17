import pytest
from mlx import nn

from mflux.models.fibo.model.fibo_text_encoder.smol_lm3_3b_text_encoder import SmolLM3_3B_TextEncoder
from mflux.models.fibo.model.fibo_transformer.joint_transformer_block import FiboJointTransformerBlock
from mflux.models.fibo.model.fibo_transformer.single_transformer_block import FiboSingleTransformerBlock
from mflux.models.fibo.model.fibo_transformer.text_projection import BriaFiboTextProjection
from mflux.models.fibo.model.fibo_transformer.time_embed import BriaFiboTimestepProjEmbeddings
from mflux.models.fibo.weights.fibo_weight_definition import FIBOWeightDefinition
from mflux.models.flux.model.flux_transformer.ada_layer_norm_continuous import AdaLayerNormContinuous
from tests.model_saving.tiny_checkpoint_helper import TinyCheckpointRoundtrip, TinyVAEStandIn


class _TinyFiboTransformer(nn.Module):
    # FIBOWeightDefinition.get_components() names one "transformer" component, but
    # unlike ERNIE/Ideogram 4 the real FiboTransformer.__init__ only exposes
    # in_channels/num_layers/num_single_layers — dim=3072, num_attention_heads=24,
    # attention_head_dim=128, and the 4096->3072 / 2048->1536 projections are
    # hardcoded in its submodule defaults, so the top-level class cannot be built
    # small. This composes the same real leaf classes (same attribute names
    # FiboTransformer uses) at toy dimensions instead. Every leaf shrinks to 128
    # since #617 parameterized the two shared FLUX.1 classes on the path,
    # AdaLayerNormZeroSingle and TimestepEmbedder; this test was held back off
    # tiny-tests-2 while those were hardcoded at 3072.
    def __init__(self) -> None:
        super().__init__()
        self.x_embedder = nn.Linear(64, 128)
        self.time_embed = BriaFiboTimestepProjEmbeddings(embedding_dim=128)
        self.context_embedder = nn.Linear(128, 128)
        self.transformer_blocks = [
            FiboJointTransformerBlock(i, dim=128, num_attention_heads=2, attention_head_dim=64) for i in range(2)
        ]
        self.single_transformer_blocks = [
            FiboSingleTransformerBlock(i, dim=128, num_attention_heads=2, attention_head_dim=64) for i in range(2)
        ]
        self.norm_out = AdaLayerNormContinuous(128, 128)
        self.proj_out = nn.Linear(128, 64)
        self.caption_projection = [BriaFiboTextProjection(in_features=128, hidden_size=64) for _ in range(4)]


class TestTinyFiboModelSaving:
    @pytest.mark.fast
    def test_tiny_quantized_checkpoint_roundtrips_exactly(self, tmp_path):
        # No slow twin exists for FIBO yet (no tests/model_saving/test_model_saving_fibo.py).
        # This exercises the same ModelSaver -> WeightLoader -> WeightApplier seam with
        # real FIBO component classes at tiny dimensions, with no downloads.
        TinyCheckpointRoundtrip.save_and_reload_expecting_identical_weights(
            weight_definition=FIBOWeightDefinition,
            make_components=TestTinyFiboModelSaving._tiny_components,
            base_path=tmp_path / "fibo_tiny_q8",
            bits=8,
            # Force shard boundaries so index.json/weight_map multi-shard paths are
            # exercised — the size-based split never shards test-sized tensors.
            tensors_per_shard=8,
        )

    @staticmethod
    def _tiny_components():
        # Every dimension is a multiple of 64 (MLX's quantization group size,
        # FIBOWeightDefinition has no quantization_group_size override so it
        # defaults to 64) so the tiny components quantize exactly like the
        # full-size checkpoint does. head_dim (64) = hidden_size (128) / heads (2)
        # for both the transformer blocks and the text encoder; num_key_value_heads
        # (1) divides num_attention_heads (2).
        return {
            "vae": TinyVAEStandIn(),
            "transformer": _TinyFiboTransformer(),
            "text_encoder": SmolLM3_3B_TextEncoder(
                vocab_size=128,
                hidden_size=128,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
            ),
        }
