import mlx.core as mx
import numpy as np

from mflux.models.common.pid_decoder.pixdit.pixdit_attention import flash_sdpa

# `mx.fast.scaled_dot_product_attention` is a fused Metal kernel, and how far it lands from a
# dense softmax-matmul depends on the GPU rather than on mflux or the mlx version. Measured with
# no mflux code in the loop, same shapes as the tests below:
#
#   Apple M2 Ultra  head_dim 64: 3.3e-07   head_dim 72: 6.6e-07
#   Apple M5 Pro    head_dim 64: 7.5e-04   head_dim 72: 2.0e-03   (reported on #490)
#   CPU backend     both: bit-exact
#
# A 1e-5 tolerance therefore asserts bit-exactness of something this PR does not control, and
# fails on at least some Apple Silicon GPUs at any mlx version. What these tests are for is that
# the padded path returns the same *attention* -- that padding and masking do not leak into the
# result -- and a leak is O(1), orders of magnitude above this bound.
FUSED_KERNEL_ATOL = 5e-3


def _reference_attention(q, k, v, scale, mask=None):
    scores = (q * scale) @ k.swapaxes(-1, -2)
    if mask is not None:
        scores = scores + mask
    return mx.softmax(scores, axis=-1, precise=True) @ v


def test_flash_sdpa_head_dim_72_matches_dense_attention():
    # head_dim 72 is PidNet's pixel stream (attn 1152 / 16 heads) and is NOT a fused-kernel
    # size, so this is the padded path. It must stay numerically exact, not approximate.
    mx.random.seed(0)
    q, k, v = (mx.random.normal((1, 4, 37, 72)) for _ in range(3))
    got = flash_sdpa(q, k, v, scale=72**-0.5)
    want = _reference_attention(q, k, v, scale=72**-0.5)
    assert got.shape == (1, 4, 37, 72)
    np.testing.assert_allclose(np.array(got), np.array(want), atol=FUSED_KERNEL_ATOL)


def test_flash_sdpa_head_dim_64_matches_dense_attention():
    # head_dim 64 is the patch stream: already fused, takes the direct path unchanged.
    mx.random.seed(0)
    q, k, v = (mx.random.normal((1, 4, 37, 64)) for _ in range(3))
    got = flash_sdpa(q, k, v, scale=64**-0.5)
    want = _reference_attention(q, k, v, scale=64**-0.5)
    np.testing.assert_allclose(np.array(got), np.array(want), atol=FUSED_KERNEL_ATOL)


def test_flash_sdpa_padded_path_honors_mask():
    mx.random.seed(0)
    q, k, v = (mx.random.normal((1, 2, 8, 72)) for _ in range(3))
    mask = mx.where(mx.arange(8)[:, None] >= mx.arange(8)[None, :], 0.0, -mx.inf)
    got = flash_sdpa(q, k, v, scale=72**-0.5, mask=mask)
    want = _reference_attention(q, k, v, scale=72**-0.5, mask=mask)
    np.testing.assert_allclose(np.array(got), np.array(want), atol=FUSED_KERNEL_ATOL)


def test_flash_sdpa_avoids_materializing_dense_scores():
    # The regression guard: at this size the dense [1, 16, 4096, 4096] scores are 1GB, so a
    # fallback to the dense kernel is unmistakable in peak memory. Fused stays far below.
    S, H, head_dim = 4096, 16, 72
    dense_scores_gb = S * S * H * 4 / 2**30
    q, k, v = (mx.random.normal((1, H, S, head_dim)) for _ in range(3))
    mx.eval(q, k, v)
    mx.reset_peak_memory()
    mx.eval(flash_sdpa(q, k, v, scale=head_dim**-0.5))
    peak_gb = mx.get_peak_memory() / 2**30
    assert peak_gb < dense_scores_gb / 2, (
        f"peak {peak_gb:.2f}GB suggests dense fallback (scores are {dense_scores_gb:.2f}GB)"
    )
