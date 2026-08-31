import numpy as np
import pytest

pytest.importorskip("mlx.core", exc_type=ImportError)

import mlx.core as mx

from mflux.models.seedvr2.model.seedvr2_transformer.attention import MMAttention
from mflux.models.seedvr2.model.seedvr2_transformer.window import WindowPartitioner


@pytest.mark.fast
def test_window_to_batch_expands_uneven_counts_like_numpy_repeat():
    # window_counts holds one window count per batch element, so the expansion is
    # np.repeat(arr, counts, axis=0). mx.repeat cannot express it: it takes a scalar
    # count, and mlx <= 0.32.1 only accepted mx.array(counts) because pybind11 coerced
    # a one-element array to an int (which is why a single batch element worked and a
    # second one never did); mlx 0.32.2 rejects the array outright.
    counts = [2, 3]
    win_to_batch = MMAttention._window_to_batch(counts)

    assert win_to_batch.tolist() == [0, 0, 1, 1, 1]

    values = mx.array([[10, 11], [20, 21]], dtype=mx.int32)
    expected = np.repeat(np.array(values), counts, axis=0)
    np.testing.assert_array_equal(np.array(values[win_to_batch]), expected)


@pytest.mark.fast
def test_mm_attention_runs_with_more_than_one_batch_element():
    # Every window-repeat call site in MMAttention is reached here: a second batch element
    # makes window_counts longer than one, which used to raise
    # "repeat(): incompatible function arguments" on every mlx version.
    t, h, w, txt_len, vid_dim, txt_dim = 2, 6, 6, 4, 8, 8
    vid_shape = mx.array([[t, h, w], [t, h, w]], dtype=mx.int32)
    txt_shape = mx.full((2, 1), txt_len, dtype=mx.int32)

    assert len(WindowPartitioner(vid_shape, (2, 2, 2), False).window_counts) == 2

    mx.random.seed(0)
    attention = MMAttention(vid_dim=vid_dim, txt_dim=txt_dim, heads=2, head_dim=16, rope_dim=16, window=(2, 2, 2))
    mx.eval(attention.parameters())

    vid = mx.random.normal((2, t * h * w, vid_dim))
    txt = mx.random.normal((2, txt_len, txt_dim))

    vid_out, txt_out = attention(vid, txt, vid_shape, txt_shape)

    assert vid_out.shape == (2, t * h * w, vid_dim)
    assert txt_out.shape == (2, txt_len, txt_dim)

    # Windows never span batch elements, so a batched call must equal the single-element
    # calls it is made of — proof the gather lines each window on its own text, rather
    # than merely producing an array of the right shape.
    for index in range(2):
        vid_alone, txt_alone = attention(
            vid[index : index + 1],
            txt[index : index + 1],
            vid_shape[index : index + 1],
            txt_shape[index : index + 1],
        )
        np.testing.assert_allclose(np.array(vid_alone[0]), np.array(vid_out[index]), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.array(txt_alone[0]), np.array(txt_out[index]), rtol=1e-5, atol=1e-5)
