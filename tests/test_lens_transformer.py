import mlx.core as mx
import pytest

from mflux.models.lens.model.transformer.lens_transformer import LensTransformer


@pytest.mark.fast
class TestLensTransformerShapes:
    def test_tiny_forward_shapes_and_finiteness(self):
        model = LensTransformer(
            num_layers=2,
            attention_head_dim=64,
            num_attention_heads=4,
            enc_hidden_dim=16,
            num_selected_layers=4,
            in_channels=128,
            out_channels=32,
        )
        h = w = 4
        latents = mx.random.normal((1, h * w, 128))
        features = mx.random.normal((1, 7, 4, 16))
        out = model(
            hidden_states=latents,
            encoder_layers=features,
            timestep=mx.array([0.9]),
            latent_height=h,
            latent_width=w,
        )
        assert out.shape == (1, h * w, 2 * 2 * 32)
        assert bool(mx.isfinite(out).all())

    def test_weight_paths_match_checkpoint_convention(self):
        from mlx.utils import tree_flatten

        model = LensTransformer(num_layers=1, attention_head_dim=64, num_attention_heads=4, enc_hidden_dim=16)
        keys = {k for k, _ in tree_flatten(model.parameters())}
        expected = {
            "img_in.weight",
            "txt_in.weight",
            "txt_norm.0.weight",
            "txt_norm.3.weight",
            "time_text_embed.timestep_embedder.linear_1.weight",
            "transformer_blocks.0.img_mod.1.weight",
            "transformer_blocks.0.attn.img_qkv.weight",
            "transformer_blocks.0.attn.to_out.0.weight",
            "transformer_blocks.0.attn.norm_added_q.weight",
            "transformer_blocks.0.img_mlp.w1.weight",
            "norm_out.linear.weight",
            "proj_out.weight",
        }
        missing = expected - keys
        assert not missing, missing
