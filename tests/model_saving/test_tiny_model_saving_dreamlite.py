import mlx.core as mx
import pytest

from mflux.models.dreamlite.model.dreamlite_text_encoder.dreamlite_text_encoder import DreamLiteTextEncoder
from mflux.models.dreamlite.model.dreamlite_unet.unet import DreamLiteUNet
from mflux.models.dreamlite.weights.dreamlite_weight_definition import DreamLiteWeightDefinition
from tests.model_saving.tiny_checkpoint_helper import TinyCheckpointRoundtrip, TinyVAEStandIn


class TestTinyDreamLiteModelSaving:
    # There is no slow save/load twin for DreamLite; this is its checkpoint coverage. The same
    # ModelSaver -> WeightLoader -> WeightApplier seam as the other tiny tests, on the real UNet and
    # text encoder classes at tiny widths.
    @pytest.mark.fast
    @pytest.mark.parametrize("bits", [4, 8])
    def test_tiny_quantized_checkpoint_roundtrips_exactly(self, tmp_path, bits):
        TinyCheckpointRoundtrip.save_and_reload_expecting_identical_weights(
            weight_definition=DreamLiteWeightDefinition,
            make_components=TestTinyDreamLiteModelSaving._tiny_components,
            base_path=tmp_path / f"dreamlite_tiny_q{bits}",
            bits=bits,
            # Force shard boundaries so the index.json / weight_map multi-shard paths are exercised.
            tensors_per_shard=16,
        )

    @pytest.mark.fast
    def test_the_saved_layout_is_the_upstream_component_directories(self, tmp_path):
        from mflux.models.common.weights.saving.model_saver import ModelSaver

        ModelSaver.save_model(
            model=TinyCheckpointRoundtrip._as_model(
                TestTinyDreamLiteModelSaving._tiny_components(), DreamLiteWeightDefinition
            ),
            bits=None,
            base_path=str(tmp_path),
            weight_definition=DreamLiteWeightDefinition,
        )

        assert sorted(path.name for path in tmp_path.iterdir()) == ["text_encoder", "unet", "vae"]

    @pytest.mark.fast
    def test_a_quantized_unet_keeps_its_convolutions_norms_and_conditioning_embedders_dense(self):
        components = TestTinyDreamLiteModelSaving._tiny_components()
        TinyCheckpointRoundtrip._quantize(components, DreamLiteWeightDefinition, 8)

        unet = components["unet"]
        assert hasattr(unet.down_blocks[0].attentions[0].transformer_blocks[0].attn2.to_q, "scales")
        assert not hasattr(unet.conv_in, "scales")
        assert not hasattr(unet.down_blocks[0].resnets[0].conv1.depthwise, "scales")
        assert not hasattr(unet.encoder_hid_proj[1], "scales")
        assert not hasattr(unet.time_embedding.linear_1, "scales")
        assert not hasattr(unet.add_embedding.linear_2, "scales")

    @pytest.mark.fast
    def test_the_tiny_unet_predicts_at_the_size_of_its_side_by_side_input(self):
        unet = TestTinyDreamLiteModelSaving._tiny_components()["unet"]

        out = unet(mx.zeros((1, 4, 16, 32)), mx.array(500.0), mx.zeros((1, 5, 128)), (128, 128))

        assert out.shape == (1, 4, 16, 32)

    @staticmethod
    def _tiny_components():
        # Every stage keeps the 64-wide attention head, so its width is 64 times its head count; all widths are
        # multiples of the group norms' 32 groups and of the quantization group size (64).
        text_config = {
            **DreamLiteTextEncoder.TEXT_CONFIG,
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
            **DreamLiteTextEncoder.VISION_CONFIG,
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
            "unet": DreamLiteUNet(
                block_channels=(64, 128, 128),
                num_heads=(1, 2, 2),
                transformer_layers=(1, 1, 1),
                time_proj_dim=64,
                time_embed_dim=128,
                text_dim=128,
                context_dim=128,
            ),
            "text_encoder": DreamLiteTextEncoder(text_config, vision_config),
        }
