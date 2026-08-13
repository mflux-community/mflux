import numpy as np
import pytest

pytest.importorskip("mlx.core", exc_type=ImportError)

from mflux.models.seedvr2.variants.upscale.seedvr2_util import SeedVR2Util


@pytest.mark.fast
def test_hist_match_matches_argsort_inverse_with_repeated_values():
    source = np.array(
        [
            [[0.5, 0.1, 0.5], [0.3, 0.1, 0.9]],
            [[0.2, 0.2, 0.8], [0.4, 0.6, 0.4]],
        ],
        dtype=np.float32,
    )
    reference = np.array(
        [
            [[0.9, 0.7, 0.6], [0.4, 0.3, 0.1]],
            [[0.8, 0.7, 0.5], [0.3, 0.2, 0.1]],
        ],
        dtype=np.float32,
    )

    expected = np.empty_like(source, dtype=np.float32)
    for index in range(source.shape[0]):
        src = source[index].reshape(-1)
        src_idx = np.argsort(src, kind="stable")
        inverse = np.argsort(src_idx, kind="stable")
        reference_sorted = np.sort(reference[index].reshape(-1), kind="stable")
        expected[index] = reference_sorted[inverse].reshape(source.shape[1:])

    actual = SeedVR2Util._hist_match(source, reference)

    np.testing.assert_array_equal(actual, expected)
