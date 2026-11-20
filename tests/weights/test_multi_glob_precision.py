"""Z-Image's transformer is F32 on disk across several shards, so the precision cast has to happen
per shard. Casting only after every shard is resident holds both copies at once -- 22.9 GB of F32
plus the ~12 GB bf16 result -- which does not fit a 36 GB Mac beside the text encoder (#n/a, found
while timing the Stage 3d ANE arms).

These tests fake mx.load, so no checkpoint is downloaded and no real tensors are allocated.
"""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mflux.models.common.weights.loading import weight_loader as wl

SHARDS = 3


@pytest.fixture
def fake_shards(tmp_path, monkeypatch):
    """Three shards of F32 tensors, plus counters for how many loads and evals have happened."""
    calls = {"loads": 0, "evals": [], "files": []}
    for i in range(SHARDS):
        p = tmp_path / f"{i:02d}.safetensors"
        p.touch()

    def fake_load(path, return_metadata=False):
        calls["loads"] += 1
        calls["files"].append(str(path).split("/")[-1])
        w = {f"layer.{calls['loads'] - 1}.{j}": mx.array(np.ones((4, 8), np.float32)) for j in range(3)}
        return (w, {}) if return_metadata else w

    def fake_eval(*arrays):
        calls["evals"].append(calls["loads"])

    monkeypatch.setattr(wl.mx, "load", fake_load)
    monkeypatch.setattr(wl.mx, "eval", fake_eval)
    return tmp_path, calls


@pytest.mark.fast
def test_precision_is_applied_shard_by_shard_not_at_the_end(fake_shards):
    path, calls = fake_shards

    weights = wl.WeightLoader._load_multi_glob(path, mx.bfloat16)

    assert calls["loads"] == SHARDS
    # one cast-and-eval per shard, and crucially the first evals happen before the last shard is
    # read: evals recorded while loads==1 and 2 prove the previous shard's F32 source can be freed
    assert calls["evals"] == [1, 2, 3], calls["evals"]
    assert all(t.dtype == mx.bfloat16 for t in weights.values())
    assert len(weights) == SHARDS * 3


@pytest.mark.fast
def test_no_precision_means_no_cast_and_no_extra_eval(fake_shards):
    path, calls = fake_shards

    weights = wl.WeightLoader._load_multi_glob(path)

    assert calls["evals"] == []
    assert all(t.dtype == mx.float32 for t in weights.values())


@pytest.mark.fast
def test_tensors_already_in_the_target_dtype_are_left_alone(tmp_path, monkeypatch):
    calls = {"loads": 0}

    def fake_load(p, return_metadata=False):
        calls["loads"] += 1
        return {"already.bf16": mx.array(np.ones((2, 2), np.float32)).astype(mx.bfloat16)}, {}

    monkeypatch.setattr(wl.mx, "load", fake_load)
    monkeypatch.setattr(wl.mx, "eval", lambda *a: None)
    (tmp_path / "00.safetensors").touch()

    weights = wl.WeightLoader._load_multi_glob(tmp_path, mx.bfloat16)

    assert weights["already.bf16"].dtype == mx.bfloat16


@pytest.mark.fast
def test_safetensors_dispatch_passes_precision_to_multi_glob(monkeypatch, tmp_path):
    seen = {}

    def spy(p, precision=None):
        seen["precision"] = precision
        return {"x": mx.array(np.ones((2,), np.float32)).astype(precision or mx.float32)}

    monkeypatch.setattr(wl.WeightLoader, "_load_multi_glob", staticmethod(spy))

    out = wl.WeightLoader._load_safetensors(tmp_path, "multi_glob", None, mx.bfloat16)

    assert seen["precision"] == mx.bfloat16
    assert out["x"].dtype == mx.bfloat16


@pytest.mark.fast
def test_mlx_native_casts_shard_by_shard_too(tmp_path, monkeypatch):
    """Z-Image leaves loading_mode at its default, which is mlx_native -- the path that actually
    hits the 36 GB wall, so it has to be covered, not just multi_glob."""
    calls = {"loads": 0, "evals": []}

    def fake_load(p):
        calls["loads"] += 1
        return {f"l{calls['loads'] - 1}.{j}": mx.array(np.ones((4, 8), np.float32)) for j in range(3)}

    monkeypatch.setattr(wl.mx, "load", fake_load)
    monkeypatch.setattr(wl.mx, "eval", lambda *a: calls["evals"].append(calls["loads"]))
    for i in range(SHARDS):
        (tmp_path / f"{i:02d}.safetensors").touch()

    weights = wl.WeightLoader._load_mlx_native(tmp_path, None, mx.bfloat16)

    assert len(weights) == SHARDS * 3
    assert all(t.dtype == mx.bfloat16 for t in weights.values())
    # one materialisation per shard, the first two before the following shard is read
    assert calls["evals"] == [1, 2, 3], calls["evals"]


@pytest.mark.fast
def test_dispatch_passes_precision_to_mlx_native(monkeypatch):
    seen = {}

    def spy(p, weight_files=None, precision=None):
        seen["precision"] = precision
        return {}

    monkeypatch.setattr(wl.WeightLoader, "_load_mlx_native", staticmethod(spy))
    wl.WeightLoader._load_safetensors(Path("."), "mlx_native", None, mx.bfloat16)
    assert seen["precision"] == mx.bfloat16, "Z-Image loads through mlx_native; without this the fix is inert"


@pytest.mark.fast
def test_z_image_components_take_this_route():
    """The fix only helps Z-Image if its components really use mlx_native with a precision set."""
    from mflux.models.z_image.weights.z_image_weight_definition import ZImageWeightDefinition

    modes = {c.name: (c.loading_mode, c.precision) for c in ZImageWeightDefinition.get_components()}
    for name, (mode, precision) in modes.items():
        assert mode == "mlx_native", (name, mode)
        assert precision is not None, name


@pytest.mark.fast
def test_empty_directory_still_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(wl.mx, "load", lambda *a, **k: ({}, {}))
    with pytest.raises(FileNotFoundError):
        wl.WeightLoader._load_multi_glob(tmp_path, mx.bfloat16)
