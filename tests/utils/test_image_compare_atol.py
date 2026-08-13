from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mflux.utils.image_compare import ImageCompare


def _save(path: Path, array: np.ndarray) -> Path:
    Image.fromarray(array.astype(np.uint8)).save(path)
    return path


@pytest.fixture(autouse=True)
def _clean_tolerance_env(monkeypatch):
    # The comparator reads these at call time; an external value would change what
    # "default" means in the assertions below.
    monkeypatch.delenv("MFLUX_IMAGE_ALLCLOSE_ATOL", raising=False)
    monkeypatch.delenv("MFLUX_IMAGE_ALLCLOSE_RTOL", raising=False)
    # The mismatch threshold is captured at import time, so delenv cannot isolate it.
    monkeypatch.setattr(ImageCompare, "ENV_MISMATCH_THRESHOLD", ImageCompare.DEFAULT_MISMATCH_THRESHOLD)


@pytest.fixture
def near_black_pair(tmp_path):
    # Dark reference plus a copy shifted by 1-2 counts: visually identical,
    # but rtol-only comparison flags nearly every pixel near zero.
    rng = np.random.default_rng(42)
    base = rng.integers(0, 6, size=(32, 32, 3)).astype(np.int16)
    shift = rng.integers(1, 3, size=base.shape)
    shifted = np.clip(base + shift, 0, 255)
    path1 = _save(tmp_path / "ref.png", base)
    path2 = _save(tmp_path / "shifted.png", shifted)
    return path1, path2


@pytest.fixture
def different_pair(tmp_path):
    black = np.zeros((32, 32, 3), dtype=np.uint8)
    white = np.full((32, 32, 3), 200, dtype=np.uint8)
    path1 = _save(tmp_path / "black.png", black)
    path2 = _save(tmp_path / "white.png", white)
    return path1, path2


@pytest.mark.fast
def test_near_black_pair_passes_with_default_atol(near_black_pair):
    path1, path2 = near_black_pair
    mismatch_ratio = ImageCompare.check_images_close_enough(path1, path2, "Near-black pair should pass.")
    assert mismatch_ratio == 0.0

    result = ImageCompare.compare_images(path1, path2, mismatch_threshold=0.15)
    assert result["mismatched_pixels"] == 0
    assert bool(result["passes_threshold"]) is True


@pytest.mark.fast
def test_near_black_pair_fails_with_atol_forced_to_zero(near_black_pair, monkeypatch):
    monkeypatch.setenv("MFLUX_IMAGE_ALLCLOSE_ATOL", "0")
    path1, path2 = near_black_pair
    with pytest.raises(AssertionError, match="Near-black pair"):
        ImageCompare.check_images_close_enough(path1, path2, "Near-black pair with atol=0.")

    result = ImageCompare.compare_images(path1, path2, mismatch_threshold=0.15)
    assert result["mismatch_ratio"] > 0.15
    assert bool(result["passes_threshold"]) is False


@pytest.mark.fast
def test_genuinely_different_pair_still_fails_at_default_atol(different_pair):
    path1, path2 = different_pair
    with pytest.raises(AssertionError, match="Different pair"):
        ImageCompare.check_images_close_enough(path1, path2, "Different pair.")

    result = ImageCompare.compare_images(path1, path2, mismatch_threshold=0.15)
    assert result["mismatch_ratio"] == 1.0
    assert bool(result["passes_threshold"]) is False


@pytest.mark.fast
def test_atol_env_var_overrides_default(tmp_path, monkeypatch):
    # A pair differing by exactly 3 counts: fails at the default atol=2 and passes at
    # atol=3, so the configured value is distinguishable from the default.
    base = np.full((16, 16, 3), 4, dtype=np.uint8)
    path1 = _save(tmp_path / "a.png", base)
    path2 = _save(tmp_path / "b.png", base + 3)
    with pytest.raises(AssertionError):
        ImageCompare.check_images_close_enough(path1, path2, "Diff-of-3 at the default atol.")
    monkeypatch.setenv("MFLUX_IMAGE_ALLCLOSE_ATOL", "3")
    mismatch_ratio = ImageCompare.check_images_close_enough(path1, path2, "Diff-of-3 at atol=3.")
    assert mismatch_ratio == 0.0
