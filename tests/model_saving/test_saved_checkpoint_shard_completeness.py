import json

import mlx.core as mx
import mlx.nn as nn
import pytest

from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.common.weights.saving.model_saver import ModelSaver


class _TinyModule(nn.Module):
    def __init__(self, keys: list[str]):
        super().__init__()
        for key in keys:
            setattr(self, key, mx.zeros((2, 2)))


@pytest.fixture
def save(monkeypatch):
    # Writes checkpoints the way mflux-save writes them, one tensor per shard. _split_weights
    # packs by size and test-sized tensors all land in 0.safetensors, which hides every
    # multi-shard case. Splitting per key keeps the real _save_weights, so the shard naming,
    # the per-shard metadata and the index all come from the saver itself.
    monkeypatch.setattr(
        ModelSaver,
        "_split_weights",
        staticmethod(lambda weights, max_file_size_gb=2: [{key: value} for key, value in weights.items()]),
    )

    def _save(tmp_path, keys: list[str], bits: int = 8, subdir: str = "transformer"):
        ModelSaver._save_weights(str(tmp_path), bits, _TinyModule(keys), subdir)
        return tmp_path / subdir

    return _save


@pytest.mark.fast
def test_a_complete_saved_checkpoint_still_loads(tmp_path, save):
    path = save(tmp_path, ["a", "b"])

    weights, quantization_level, mflux_version = WeightLoader._try_load_mflux_format(path)

    assert set(weights) == {"a", "b"}
    assert quantization_level == 8
    assert mflux_version


@pytest.mark.fast
def test_a_half_copied_checkpoint_raises_and_names_every_missing_shard(tmp_path, save):
    # The failure this guards: the loader read the directory, so a checkpoint whose copy
    # was interrupted came up with whatever shards arrived, and the update downstream runs
    # with strict=False and generates noise.
    path = save(tmp_path, ["a", "b", "c"])
    index = json.loads((path / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    for shard in shards[1:]:
        (path / shard).unlink()

    with pytest.raises(FileNotFoundError) as excinfo:
        WeightLoader._try_load_mflux_format(path)

    for shard in shards[1:]:
        assert shard in str(excinfo.value)


@pytest.mark.fast
def test_shards_left_by_an_earlier_save_do_not_come_back(tmp_path, save):
    # _save_weights does mkdir(exist_ok=True) and never clears, so saving a smaller
    # checkpoint over a larger one leaves the old tail in place. The index names only the
    # new shards; reading the directory instead let the stale tensors overwrite them.
    path = save(tmp_path, ["a", "b", "c", "d"])
    stale = sorted(p.name for p in path.glob("*.safetensors"))
    save(tmp_path, ["a"], bits=4)
    assert len(stale) >= 1

    weights, quantization_level, _ = WeightLoader._try_load_mflux_format(path)

    assert set(weights) == {"a"}
    assert quantization_level == 4


@pytest.mark.fast
def test_a_checkpoint_saved_before_the_index_existed_still_loads(tmp_path, save):
    path = save(tmp_path, ["a"])
    (path / "model.safetensors.index.json").unlink()

    weights, quantization_level, _ = WeightLoader._try_load_mflux_format(path)

    assert set(weights) == {"a"}
    assert quantization_level == 8


@pytest.mark.fast
def test_complete_weights_beside_a_damaged_index_still_load(tmp_path, save):
    # The saver writes the shards and then the index, so an interrupted save can leave
    # correct weights next to a truncated json. A damaged manifest is not evidence that
    # the weights are gone, and telling someone to re-save 26 GB over it would be wrong.
    path = save(tmp_path, ["a", "b"])
    (path / "model.safetensors.index.json").write_text('{"weight_map": {"a": ')

    weights, _, _ = WeightLoader._try_load_mflux_format(path)

    assert set(weights) == {"a", "b"}


@pytest.mark.fast
def test_an_index_pointing_outside_the_directory_is_ignored(tmp_path, save):
    path = save(tmp_path, ["a"])
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "../elsewhere.safetensors"}}))

    weights, _, _ = WeightLoader._try_load_mflux_format(path)

    assert set(weights) == {"a"}


@pytest.mark.fast
def test_a_directory_that_is_not_an_mflux_checkpoint_is_still_declined(tmp_path):
    # Returning None is how the caller falls through to the other loading modes, so a
    # foreign directory must not turn into an exception on the way past.
    mx.save_safetensors(str(tmp_path / "model.safetensors"), {"a": mx.zeros((2, 2))})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "missing.safetensors"}}))

    assert WeightLoader._try_load_mflux_format(tmp_path) == (None, None, None)
