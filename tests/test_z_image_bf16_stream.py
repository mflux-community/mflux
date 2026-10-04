import mlx.core as mx
import pytest

from mflux.models.z_image.model.z_image_transformer.attention import ZImageAttention
from mflux.models.z_image.model.z_image_transformer.transformer import ZImageTransformer

# A toy transformer (head_dim 32 = 8 + 12 + 12) that runs the real forward in milliseconds.
DIM = 64


def _tiny_transformer(float32: bool = False) -> ZImageTransformer:
    mx.random.seed(0)
    transformer = ZImageTransformer(
        dim=DIM,
        n_layers=2,
        n_refiner_layers=1,
        n_heads=2,
        cap_feat_dim=32,
        axes_dims=[8, 12, 12],
        axes_lens=[64, 32, 32],
    )
    # Loaded weights are bfloat16. The RoPE tables are not parameters and stay float32, as in a real load.
    transformer.set_dtype(mx.bfloat16)
    transformer.set_float32(float32)
    return transformer


def _run(transformer: ZImageTransformer) -> mx.array:
    mx.random.seed(1)
    latents = mx.random.normal((16, 1, 8, 8)).astype(mx.bfloat16)
    cap_feats = mx.random.normal((7, 32)).astype(mx.bfloat16)
    return transformer(x=latents, timestep=mx.array([0.5]), sigmas=mx.array([0.5]), cap_feats=cap_feats)


@pytest.mark.fast
def test_hidden_stream_stays_in_model_precision():
    assert _run(_tiny_transformer()).dtype == mx.bfloat16


@pytest.mark.fast
def test_float32_option_keeps_the_float32_stream():
    assert _run(_tiny_transformer(float32=True)).dtype == mx.float32


@pytest.mark.fast
def test_set_float32_reaches_every_attention():
    transformer = _tiny_transformer(float32=True)
    attentions = [m for m in transformer.modules() if isinstance(m, ZImageAttention)]
    assert len(attentions) == 4  # noise refiner, context refiner, 2 main layers
    assert all(a.float32 for a in attentions)

    transformer.set_float32(False)
    assert not any(a.float32 for a in attentions)


@pytest.mark.fast
def test_rotary_output_is_the_float32_rotation_cast_to_the_input_dtype():
    mx.random.seed(2)
    x = mx.random.normal((1, 5, 2, 32)).astype(mx.bfloat16)
    angles = mx.random.uniform(shape=(5, 16))
    freqs_cis = mx.stack([mx.cos(angles), mx.sin(angles)], axis=-1)

    full = ZImageAttention._apply_rotary_emb(x, freqs_cis, keep_float32=True)
    cast = ZImageAttention._apply_rotary_emb(x, freqs_cis)

    assert full.dtype == mx.float32
    assert cast.dtype == mx.bfloat16
    assert mx.array_equal(cast, full.astype(mx.bfloat16))
