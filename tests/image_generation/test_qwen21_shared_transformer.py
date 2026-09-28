import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mflux.models.qwen21.model.qwen21_transformer.qwen21_layout import QwenImage21Layout
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer_block import Qwen21TransformerBlock
from mflux.models.qwen21.reference.model.qwen_image21_transformer.transformer import QwenImage21Transformer

pytestmark = pytest.mark.fast


class TestQwen21SharedTransformer:
    CONFIG = dict(
        in_channels=4,
        out_channels=4,
        context_in_dim=16,
        num_layers=2,
        num_attention_heads=2,
        attention_head_dim=8,
        mlp_ratio=2,
        axes_dims_rope=(2, 2, 4),
    )

    @staticmethod
    def model(dtype=mx.float32):
        model = QwenImage21Transformer(TestQwen21SharedTransformer.CONFIG)
        rng = np.random.default_rng(741)
        weights = []
        for name, value in sorted(tree_flatten(model.parameters())):
            data = rng.standard_normal(value.shape).astype(np.float32) * 0.1
            if ".norm_q." in name or ".norm_k." in name:
                data += 1
            weights.append((name, mx.array(data).astype(dtype)))
        model.load_weights(weights)
        return model

    def test_edit_uses_canonical_blocks_and_parameter_names(self):
        model = self.model()
        assert isinstance(model, Qwen21Transformer)
        assert all(type(block) is Qwen21TransformerBlock for block in model.transformer_blocks)
        keys = dict(tree_flatten(model.parameters()))
        assert "modulation.layers.1.weight" in keys
        assert "modulation.1.weight" not in keys
        assert "time_text_embed.time_proj.freqs" not in keys
        assert not any(key.startswith("pos_embed.") for key in keys)
        canonical = Qwen21Transformer(**self.CONFIG)
        assert keys.keys() == dict(tree_flatten(canonical.parameters())).keys()
        canonical.load_weights(list(keys.items()))

    @pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
    @pytest.mark.parametrize(
        "slots,shapes",
        [
            ([False] * 5, [(1, 2, 2)]),
            ([False, True, False], [(1, 2, 2)] * 2),
            ([False, True, False, True, False], [(1, 2, 2)] * 3),
            ([False, True, True, False], [(1, 2, 2)] * 3),
        ],
    )
    def test_cache_matches_full_sequence_across_requests(self, slots, shapes, dtype):
        model = self.model(dtype)
        rng = np.random.default_rng(21)
        layout = QwenImage21Layout.create(mx.array(slots), shapes, self.CONFIG["axes_dims_rope"])
        latents = mx.array(rng.standard_normal((1, 4 * len(shapes), 4)).astype(np.float32)).astype(dtype)
        text = mx.array(rng.standard_normal((1, len(slots), 16)).astype(np.float32)).astype(dtype)
        mask = mx.array([[False, *([True] * (len(slots) - 1))]])
        caches = [[], []]
        for request, cache in enumerate(caches):
            condition = text + request * 0.5
            for timestep in [0.8, 0.5, 0.1]:
                # The denoising target changes; reference latents and prompt stay fixed per cache.
                target = latents.at[:, -4:].add(timestep)
                expected = model(target, condition, mx.array([timestep]), layout, None, mask)
                actual = model(target, condition, mx.array([timestep]), layout, cache, mask)
                tolerance = 1e-4 if dtype == mx.float32 else 1e-2
                np.testing.assert_allclose(
                    np.asarray(actual.astype(mx.float32)),
                    np.asarray(expected.astype(mx.float32)),
                    atol=tolerance,
                    rtol=tolerance,
                )
            assert len(cache) == self.CONFIG["num_layers"]
        assert caches[0] is not caches[1]
        assert not np.array_equal(
            np.asarray(caches[0][0][0].astype(mx.float32)), np.asarray(caches[1][0][0].astype(mx.float32))
        )

    def test_incomplete_prefix_cache_is_rejected(self):
        model = self.model()
        layout = QwenImage21Layout.create(mx.array([False] * 3), [(1, 2, 2)], self.CONFIG["axes_dims_rope"])
        with pytest.raises(ValueError, match="Incomplete"):
            model(mx.zeros((1, 4, 4)), mx.zeros((1, 3, 16)), mx.array([0.5]), layout, [(None, None)])
