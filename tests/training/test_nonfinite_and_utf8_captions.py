import math

import mlx.core as mx
import pytest

from mflux.models.common.training.state.training_spec import DataSpec
from mflux.models.common.training.trainer import TrainingTrainer


@pytest.mark.fast
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_loss_is_flagged_for_skip(bad):
    assert TrainingTrainer._step_is_finite(mx.array(bad)) is False


@pytest.mark.fast
def test_finite_loss_is_not_skipped():
    assert TrainingTrainer._step_is_finite(mx.array(1.23)) is True
    assert TrainingTrainer._step_is_finite(mx.array(0.0)) is True


@pytest.mark.fast
def test_caption_file_with_invalid_utf8_reads_with_replacement(tmp_path):
    # A latin-1 caption ("café" saved by a non-utf8 editor) used to raise
    # UnicodeDecodeError and kill the whole training run before the first step.
    caption = tmp_path / "01.txt"
    caption.write_bytes(b"caf\xe9 con leche")
    (tmp_path / "01.jpeg").write_bytes(b"")

    spec = DataSpec.create(
        {"image": "01.jpeg", "prompt_file": "01.txt"},
        absolute_or_relative_path=str(tmp_path),
        base_path=None,
    )
    assert spec.prompt.startswith("caf")
    assert spec.prompt.endswith("con leche")
    assert "�" in spec.prompt  # the invalid byte is replaced, not fatal


@pytest.mark.fast
def test_valid_multibyte_utf8_caption_reads_intact(tmp_path):
    caption = tmp_path / "02.txt"
    caption.write_text("café über 日本語", encoding="utf-8")
    (tmp_path / "02.jpeg").write_bytes(b"")

    spec = DataSpec.create(
        {"image": "02.jpeg", "prompt_file": "02.txt"},
        absolute_or_relative_path=str(tmp_path),
        base_path=None,
    )
    assert spec.prompt == "café über 日本語"


@pytest.mark.fast
def test_math_isfinite_agreement():
    # Guard against a future rewrite comparing against a python float path.
    for v in (0.5, float("nan"), float("inf")):
        assert TrainingTrainer._step_is_finite(mx.array(v)) is math.isfinite(v)
