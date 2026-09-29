import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mflux.models.ming_image.model.ming_text_encoder.ling_moe_encoder import (
    NUM_EXPERTS,
    TOP_K,
    LingGate,
    LingMoeEncoder,
    LingRope,
)
from mflux.models.ming_image.model.ming_text_encoder.ming_condition_encoder import (
    IMAGE_END_ID,
    IMAGE_PATCH_ID,
    IMAGE_START_ID,
    MingConditionEncoder,
)
from mflux.models.ming_image.model.ming_transformer.ming_transformer import MingTransformer
from mflux.models.ming_image.model.ming_vae.ming_vae import MingVAE
from mflux.models.ming_image.variants.ming_image import MingImage
from mflux.models.ming_image.weights.ming_image_weight_definition import MingImageWeightDefinition


@pytest.mark.fast
class TestMingConditionInputs:
    def test_query_block_and_video_rope_positions(self):
        # Matches the positions get_t_scale_rope_index produced in the official pipeline's dump:
        # text counts up on all axes, the 1x1x256 query grid shares t = h and centres w on it,
        # and the closing token resumes one past the grid's t.
        input_ids, pos, image_mask = MingConditionEncoder.build_inputs([11, 12, 13])
        ids = np.array(input_ids)
        assert ids.tolist()[:4] == [11, 12, 13, IMAGE_START_ID]
        assert (ids[4:260] == IMAGE_PATCH_ID).all() and ids[-1] == IMAGE_END_ID
        pos = np.array(pos)
        assert pos.shape == (3, 3 + 1 + 256 + 1)
        assert (pos[:, :4] == np.arange(4)).all()
        assert (pos[0, 4:260] == 4).all() and (pos[1, 4:260] == 4).all()
        assert pos[2, 4] == 4 - 127 and pos[2, 259] == 4 + 128
        assert (pos[:, -1] == 5).all()
        assert int(np.array(image_mask).sum()) == 256

    def test_video_rope_axis_split(self):
        pos = mx.array([[0], [1], [2]])  # t=0, h=1, w=2
        cos, _ = LingRope.cos_sin(pos)
        inv = 1.0 / (600000.0 ** (np.arange(0, 64, 2) / 64))
        expected_pos = np.where(np.arange(32) >= 24, 0, np.where(np.arange(32) % 2 == 0, 1, 2))
        np.testing.assert_allclose(np.array(cos[0, :32]), np.cos(expected_pos * inv), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(np.array(cos[0, 32:]), np.array(cos[0, :32]))


@pytest.mark.fast
class TestLingRouter:
    def test_group_limited_topk(self):
        mx.random.seed(0)
        gate = LingGate()
        gate.weight = mx.random.normal((NUM_EXPERTS, 2048)) * 0.02
        # Bias the last group hard: at most TOP_K experts may come from outside the best 4 groups.
        gate.expert_bias = mx.concatenate([mx.zeros(NUM_EXPERTS - 32), mx.full((32,), 5.0)])
        idx, weights = gate(mx.random.normal((5, 2048)))
        idx = np.array(idx)
        assert idx.shape == (5, TOP_K)
        groups = idx // 32
        assert all(len(set(row)) <= 4 for row in groups)
        np.testing.assert_allclose(np.array(weights.sum(-1).astype(mx.float32)), 2.5, rtol=1e-5)

    def test_stack_experts_is_idempotent(self):
        per_expert = [{p: {"weight": mx.full((2, 3), float(e))} for p in ("gate_proj", "up_proj", "down_proj")} for e in range(4)]  # fmt: off
        weights = {"layers": [{"mlp": {"gate_proj": {"weight": mx.zeros((1,))}}}, {"mlp": {"experts": per_expert}}]}
        stacked = LingMoeEncoder.stack_experts(weights)
        w = stacked["layers"][1]["mlp"]["switch_mlp"]["up_proj"]["weight"]
        assert w.shape == (4, 2, 3) and float(w[3, 0, 0]) == 3.0
        assert "switch_mlp" not in stacked["layers"][0]["mlp"]
        again = LingMoeEncoder.stack_experts(stacked)
        assert again["layers"][1]["mlp"]["switch_mlp"]["up_proj"]["weight"].shape == (4, 2, 3)


@pytest.mark.fast
class TestMingWeightKeys:
    def test_text_encoder_keeps_only_the_language_model(self):
        key = MingImageWeightDefinition._text_encoder_key
        assert key("model.model.layers.3.mlp.experts.7.up_proj.weight") == "layers.3.mlp.experts.7.up_proj.weight"
        assert key("model.model.layers.3.mlp.image_gate.expert_bias") == "layers.3.mlp.image_gate.expert_bias"
        assert key("model.model.layers.3.mlp.audio_gate.weight") is None
        assert key("vision.blocks.0.attn.qkv.weight") is None
        assert key("model.lm_head.weight") is None
        assert key("linear_proj.0.weight") is None

    def test_connector_and_heads(self):
        assert MingImageWeightDefinition._connector_key("model.layers.0.self_attn.q_proj.bias") == "layers.0.self_attn.q_proj.bias"  # fmt: off
        assert MingImageWeightDefinition._connector_key("model.embed_tokens.weight") is None
        assert MingImageWeightDefinition._connector_key("lm_head.weight") is None
        assert MingImageWeightDefinition._heads_key("query_tokens_dict.16x16") == "query_tokens"
        assert MingImageWeightDefinition._heads_key("proj_directvlm.1.bias") == "proj_directvlm.1.bias"
        assert MingImageWeightDefinition._heads_key("mlp.anything") is None


@pytest.mark.fast
class TestMingTransformer:
    def test_positions_keep_the_padded_caption_offset(self):
        x_pos, cap_pos = MingTransformer._position_ids(291, (1, 2, 3))
        assert np.array(cap_pos)[:, 0].tolist() == list(range(1, 292))
        # 291 caption tokens pad to 320 upstream, so the image grid starts at t = 321.
        assert (np.array(x_pos)[:, 0] == 321).all()
        assert np.array(x_pos)[:, 1:].tolist() == [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]

    def test_tiny_forward(self):
        model = MingTransformer(dim=64, n_layers=1, n_refiner_layers=1, n_heads=2, cap_feat_dim=16, axes_dims=(8, 12, 12))  # fmt: off
        out = model(mx.random.normal((16, 1, 8, 8)), mx.array([0.3]), mx.random.normal((5, 16)), mx.random.normal((3, 64)))  # fmt: off
        assert out.shape == (16, 1, 8, 8)
        assert bool(mx.isfinite(out).all())

    def test_weight_paths_match_z_image_convention(self):
        model = MingTransformer(dim=64, n_layers=1, n_refiner_layers=1, n_heads=2, cap_feat_dim=16, axes_dims=(8, 12, 12))  # fmt: off
        keys = {k for k, _ in tree_flatten(model.parameters())}
        assert {
            "layers.0.attention.to_out.0.weight",
            "layers.0.adaLN_modulation.0.weight",
            "context_refiner.0.feed_forward.w3.weight",
            "noise_refiner.0.attention.norm_k.weight",
            "cap_embedder.1.bias",
            "all_final_layer.2-1.adaLN_modulation.0.weight",
            "t_embedder.linear1.weight",
        } <= keys
        assert not any(k.startswith("context_refiner.0.adaLN") for k in keys)
        assert "x_pad_token" not in keys


@pytest.mark.fast
class TestMingSchedule:
    def test_sigmas_match_official_pipeline(self):
        # sched.sigmas captured from the official pipeline at 12 steps.
        official = [1.0, 0.98361, 0.96429, 0.94118, 0.91304, 0.87805, 0.83333, 0.77419, 0.69231, 0.57143, 0.375, 0.0, 0.0]  # fmt: off
        np.testing.assert_allclose(np.array(MingImage.sigmas(12)), official, atol=1e-5)


@pytest.mark.fast
def test_vae_is_rgba():
    vae = MingVAE()
    assert vae.decoder.conv_out.conv3d.weight.shape[0] == 4
    assert vae.encoder.conv_in.conv3d.weight.shape[-1] == 4
