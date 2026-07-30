import mlx.core as mx

from mflux.models.common.pid_decoder.pixdit.pixdit_mmdit_block import MMDiTBlockT2I
from mflux.models.common.pid_decoder.pixdit.pixdit_rope import precompute_freqs_cis_2d_ntk


def test_mmdit_block_output_shapes():
    hidden_size, num_heads = 64, 4
    block = MMDiTBlockT2I(hidden_size, num_heads)
    x = mx.random.normal((1, 16, hidden_size))  # 4x4 image tokens
    y = mx.random.normal((1, 8, hidden_size))  # 8 text tokens
    c = mx.random.normal((1, 1, hidden_size))
    pos_img = precompute_freqs_cis_2d_ntk(hidden_size // num_heads, 4, 4, 4, 4)
    x_out, y_out = block(x, y, c, pos_img, pos_txt=None)
    assert x_out.shape == x.shape
    assert y_out.shape == y.shape


def test_mmdit_block_outputs_differ_from_inputs():
    # Sanity: both streams should actually be transformed, not pass through unchanged.
    hidden_size, num_heads = 64, 4
    block = MMDiTBlockT2I(hidden_size, num_heads)
    x = mx.random.normal((1, 16, hidden_size))
    y = mx.random.normal((1, 8, hidden_size))
    c = mx.random.normal((1, 1, hidden_size))
    pos_img = precompute_freqs_cis_2d_ntk(hidden_size // num_heads, 4, 4, 4, 4)
    x_out, y_out = block(x, y, c, pos_img, pos_txt=None)
    assert not mx.allclose(x_out, x).item()
    assert not mx.allclose(y_out, y).item()
