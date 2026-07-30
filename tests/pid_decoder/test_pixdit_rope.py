import mlx.core as mx

from mflux.models.common.pid_decoder.pixdit.pixdit_rope import apply_rotary_emb, precompute_freqs_cis_2d_ntk


def test_rope_freqs_shape():
    cos, sin = precompute_freqs_cis_2d_ntk(dim=64, height=8, width=8, ref_grid_h=8, ref_grid_w=8)
    # last dim is dim // 2 (one cos/sin entry per real/imaginary pair across both axes),
    # not dim // 4 -- verified against the actual NVIDIA source and a live MLX run.
    assert cos.shape == (64, 32)
    assert sin.shape == (64, 32)


def test_apply_rotary_emb_preserves_shape():
    freqs = precompute_freqs_cis_2d_ntk(dim=64, height=4, width=4, ref_grid_h=4, ref_grid_w=4)
    q = mx.random.normal((1, 16, 2, 64))
    k = mx.random.normal((1, 16, 2, 64))
    q_out, k_out = apply_rotary_emb(q, k, freqs)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


def test_apply_rotary_emb_identity_at_origin():
    # Position 0 of the flattened grid is (y=0, x=0), where both axes' linspace
    # start at 0 -> all freqs are 0 -> cos=1, sin=0 -> rotation is the identity.
    freqs = precompute_freqs_cis_2d_ntk(dim=64, height=4, width=4, ref_grid_h=4, ref_grid_w=4)
    q = mx.random.normal((1, 16, 2, 64))
    k = mx.random.normal((1, 16, 2, 64))
    q_out, k_out = apply_rotary_emb(q, k, freqs)
    assert mx.allclose(q_out[:, 0], q[:, 0], atol=1e-5).item()
    assert mx.allclose(k_out[:, 0], k[:, 0], atol=1e-5).item()


def test_apply_rotary_emb_varies_across_positions():
    # Two distinct grid positions should rotate the same query differently.
    freqs = precompute_freqs_cis_2d_ntk(dim=64, height=4, width=4, ref_grid_h=4, ref_grid_w=4)
    q = mx.random.normal((1, 16, 2, 64))
    q_out, _ = apply_rotary_emb(q, q, freqs)
    assert not mx.allclose(q_out[:, 1], q_out[:, 5], atol=1e-4).item()


def test_rope_tables_are_cached_across_calls():
    """Every patch block and every sampling step asks for the same grid, so the table must be
    built once: at 4096x4096 it is a ~38MB cos/sin pair and there were 12 rebuilds per decode."""
    first = precompute_freqs_cis_2d_ntk(64, 8, 8, 4, 4)
    second = precompute_freqs_cis_2d_ntk(64, 8, 8, 4, 4)
    assert first[0] is second[0] and first[1] is second[1]
    assert precompute_freqs_cis_2d_ntk(64, 16, 8, 4, 4)[0] is not first[0]  # different grid, rebuilt
