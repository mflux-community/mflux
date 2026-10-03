import dataclasses

import mlx.core as mx
import pytest

from mflux.models.depth_pro.model.depth_pro import DepthPro
from mflux.models.depth_pro.model.depth_pro_model import DepthProModel
from mflux.models.depth_pro.weights.depth_pro_weight_definition import DepthProWeightDefinition
from tests.model_saving.tiny_checkpoint_helper import TinyCheckpointRoundtrip


class _LocalDepthProWeightDefinition(DepthProWeightDefinition):
    # The real definition sets download_url, and WeightLoader always loads such a component
    # from the Apple CDN, never from a saved directory. Remove the URL so the loader reads
    # the checkpoint that ModelSaver wrote. Everything else, including the quantization
    # predicate, stays the production definition.
    @staticmethod
    def get_components():
        return [dataclasses.replace(c, download_url=None) for c in DepthProWeightDefinition.get_components()]


class TestTinyDepthProModelSaving:
    @pytest.mark.fast
    def test_tiny_quantized_checkpoint_roundtrips_exactly(self, tmp_path):
        # Depth Pro has no slow save/load twin: mflux-save-depth writes a depth image, not a
        # checkpoint. This puts the real DepthProModel tree, at tiny dimensions, through the
        # ModelSaver -> WeightLoader -> WeightApplier seam.
        TinyCheckpointRoundtrip.save_and_reload_expecting_identical_weights(
            weight_definition=_LocalDepthProWeightDefinition,
            make_components=TestTinyDepthProModelSaving._tiny_components,
            base_path=tmp_path / "depth_pro_tiny_q8",
            bits=8,
            tensors_per_shard=8,
        )

    @pytest.mark.fast
    def test_tiny_model_runs_a_forward_pass(self):
        # The save test reads only the weights. This test also runs the toy model on the real
        # patch pyramid, so a toy dimension that breaks the forward pass fails here.
        model = TestTinyDepthProModelSaving._tiny_components()["depth_pro"]
        x0, x1, x2 = DepthPro._create_patches(DepthPro._resize(mx.zeros((3, 64, 64))))
        depth = model(x0, x1, x2)
        assert depth.shape == (1, 1, 1536, 1536)

    @staticmethod
    def _tiny_components():
        # Every Linear input is a multiple of 64 (the quantization group size). The predicate
        # skips Conv2d layers. encoder_feature_dims[0] differs from decoder_features, so the forward
        # pass checks that the encoder and decoder agree on each level's width. Two blocks, so the
        # hooks are 0 and 1.
        return {
            "depth_pro": DepthProModel(
                embed_dim=64,
                num_heads=1,
                mlp_hidden_dim=128,
                num_blocks=2,
                hook_block_ids=(0, 1),
                encoder_feature_dims=(128, 64, 64, 64),
                decoder_features=64,
            ),
        }
