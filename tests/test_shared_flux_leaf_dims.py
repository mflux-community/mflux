import mlx.core as mx
import pytest
from mlx import nn
from mlx.utils import tree_flatten

from mflux.models.flux.model.flux_transformer.ada_layer_norm_zero_single import AdaLayerNormZeroSingle
from mflux.models.flux.model.flux_transformer.timestep_embedder import TimestepEmbedder


class TestSharedFluxLeafDims:
    @pytest.mark.fast
    def test_the_default_width_keeps_the_production_parameter_keys_and_shapes(self):
        # FLUX.1 constructs both leaves with no arguments, so the default must stay the
        # production 3072 or every existing checkpoint stops matching.
        ada = dict(tree_flatten(AdaLayerNormZeroSingle().parameters()))
        assert set(ada) == {"linear.weight", "linear.bias"}
        assert ada["linear.weight"].shape == (9216, 3072)

        ts = dict(tree_flatten(TimestepEmbedder().parameters()))
        assert set(ts) == {"linear_1.weight", "linear_1.bias", "linear_2.weight", "linear_2.bias"}
        assert ts["linear_1.weight"].shape == (3072, 256)
        assert ts["linear_2.weight"].shape == (3072, 3072)

    @pytest.mark.fast
    def test_the_modulation_chunks_follow_dim_not_the_production_literal(self):
        # The forward pass used to slice at chunk_size = 9216 // 3. Parameterizing the
        # constructor without this slice makes every non-default width read shift/scale/
        # gate from the wrong offsets, so pin the chunks against explicit slicing.
        dim = 128
        norm = AdaLayerNormZeroSingle(dim=dim)
        hidden_states = mx.random.normal((1, 8, dim))
        text_embeddings = mx.random.normal((1, dim))

        out, gate = norm(hidden_states, text_embeddings)

        emb = norm.linear(nn.silu(text_embeddings))
        expected = norm.norm(hidden_states) * (1 + emb[:, dim : 2 * dim][:, None]) + emb[:, 0:dim][:, None]
        assert mx.array_equal(out, expected)
        assert mx.array_equal(gate, emb[:, 2 * dim : 3 * dim])

    @pytest.mark.fast
    def test_the_sinusoidal_input_width_stays_fixed_while_dim_shrinks(self):
        # Both FLUX.1 and FIBO project timesteps to 256 sinusoidal features regardless of
        # model width, so only the output side follows dim.
        ts = TimestepEmbedder(dim=128)

        assert ts(mx.random.normal((2, 256))).shape == (2, 128)
