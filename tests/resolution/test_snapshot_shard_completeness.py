import json

import pytest

from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition
from mflux.models.krea2.weights.krea2_weight_definition import Krea2WeightDefinition


def _snapshot(tmp_path, shards_present: list[str], referenced: list[str]):
    """A cache snapshot with a sharded transformer whose index names `referenced`."""
    transformer = tmp_path / "transformer"
    transformer.mkdir(parents=True)
    for name in shards_present:
        (transformer / name).write_bytes(b"")
    index = {"weight_map": {f"layer.{i}.weight": name for i, name in enumerate(referenced)}}
    (transformer / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps(index))
    return tmp_path


@pytest.mark.fast
def test_a_complete_shard_set_is_accepted(tmp_path):
    shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    snapshot = _snapshot(tmp_path, shards_present=shards, referenced=shards)

    assert PathResolution._is_snapshot_complete(snapshot, {"transformer"}) is True


@pytest.mark.fast
def test_an_interrupted_download_is_not_mistaken_for_a_complete_one(tmp_path):
    # The failure this guards: one shard on disk satisfies "has a safetensors", the loader
    # then updates with strict=False, and the transformer comes up missing weights and
    # generating noise instead of failing.
    shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    snapshot = _snapshot(tmp_path, shards_present=shards[:1], referenced=shards)

    assert PathResolution._is_snapshot_complete(snapshot, {"transformer"}) is False


@pytest.mark.fast
def test_an_unreadable_index_is_not_treated_as_complete(tmp_path):
    transformer = tmp_path / "transformer"
    transformer.mkdir(parents=True)
    (transformer / "model-00001-of-00002.safetensors").write_bytes(b"")
    (transformer / "diffusion_pytorch_model.safetensors.index.json").write_text("{not json")

    assert PathResolution._is_snapshot_complete(tmp_path, {"transformer"}) is False


@pytest.mark.fast
@pytest.mark.parametrize("contents", ["[]", "null", '"weight_map"'])
def test_an_index_that_is_not_an_object_is_not_treated_as_complete(tmp_path, contents):
    # Valid JSON that is not a mapping: reading weight_map off it must fail the
    # snapshot rather than raise out of path resolution.
    transformer = tmp_path / "transformer"
    transformer.mkdir(parents=True)
    (transformer / "model-00001-of-00002.safetensors").write_bytes(b"")
    (transformer / "diffusion_pytorch_model.safetensors.index.json").write_text(contents)

    assert PathResolution._is_snapshot_complete(tmp_path, {"transformer"}) is False


def _root_snapshot(tmp_path, shards_present: list[str], referenced: list[str]):
    """A cache snapshot whose shards and index sit at the root, as briaai/FIBO-vlm ships."""
    for name in shards_present:
        (tmp_path / name).write_bytes(b"")
    index = {"weight_map": {f"layer.{i}.weight": name for i, name in enumerate(referenced)}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    return tmp_path


@pytest.mark.fast
def test_a_complete_root_level_shard_set_is_accepted(tmp_path):
    shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    snapshot = _root_snapshot(tmp_path, shards_present=shards, referenced=shards)

    assert PathResolution._is_snapshot_complete(snapshot, set(), ["*.safetensors", "*.json"]) is True


@pytest.mark.fast
def test_an_interrupted_root_level_download_is_rejected(tmp_path):
    # Root-level patterns take the no-subdirs branch, which used to accept the
    # snapshot as soon as one shard matched the pattern.
    shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    snapshot = _root_snapshot(tmp_path, shards_present=shards[:1], referenced=shards)

    assert PathResolution._is_snapshot_complete(snapshot, set(), ["*.safetensors", "*.json"]) is False


@pytest.mark.fast
def test_an_unsharded_root_level_snapshot_stays_complete(tmp_path):
    # No index means nothing to verify: a single-file checkpoint is still complete.
    (tmp_path / "model.safetensors").write_bytes(b"")

    assert PathResolution._is_snapshot_complete(tmp_path, set(), ["*.safetensors"]) is True


# Krea 2 Turbo keeps its transformer at the snapshot root, so only vae/ and
# text_encoder/ end up in required_subdirs. Built from the shipped definition rather
# than hand-written: the first version of this fix passed a suite full of hand-written
# pattern lists while breaking every FLUX.2 model.
KREA2_TURBO_PATTERNS = Krea2WeightDefinition.get_download_patterns("krea/Krea-2-Turbo")


def _krea2_turbo_snapshot(tmp_path, with_transformer: bool):
    for subdir in ("vae", "text_encoder"):
        (tmp_path / subdir).mkdir(parents=True)
        (tmp_path / subdir / "model.safetensors").write_bytes(b"")
        (tmp_path / subdir / "config.json").write_text("{}")
    (tmp_path / "tokenizer").mkdir()
    (tmp_path / "tokenizer" / "tokenizer.json").write_text("{}")
    if with_transformer:
        (tmp_path / "turbo.safetensors").write_bytes(b"")
    return tmp_path


@pytest.mark.fast
def test_a_snapshot_missing_its_root_level_transformer_is_incomplete(tmp_path):
    # The failure this guards: required_subdirs is {"vae", "text_encoder"}, so the
    # pattern check was skipped entirely and an interrupted download that had fetched
    # the small components but not the transformer reported itself complete.
    snapshot = _krea2_turbo_snapshot(tmp_path, with_transformer=False)

    assert PathResolution._is_snapshot_complete(snapshot, {"vae", "text_encoder"}, KREA2_TURBO_PATTERNS) is False


@pytest.mark.fast
def test_a_snapshot_with_subdirs_and_its_root_transformer_is_complete(tmp_path):
    snapshot = _krea2_turbo_snapshot(tmp_path, with_transformer=True)

    assert PathResolution._is_snapshot_complete(snapshot, {"vae", "text_encoder"}, KREA2_TURBO_PATTERNS) is True


@pytest.mark.fast
def test_every_root_weight_pattern_is_required_not_just_one(tmp_path):
    # Guards the shortcut of stopping at the first satisfied root pattern.
    (tmp_path / "vae").mkdir()
    (tmp_path / "vae" / "model.safetensors").write_bytes(b"")
    (tmp_path / "first.safetensors").write_bytes(b"")
    patterns = ["first.safetensors", "second.safetensors", "vae/*.safetensors"]

    assert PathResolution._is_snapshot_complete(tmp_path, {"vae"}, patterns) is False

    (tmp_path / "second.safetensors").write_bytes(b"")

    assert PathResolution._is_snapshot_complete(tmp_path, {"vae"}, patterns) is True


@pytest.mark.fast
def test_a_root_pattern_the_repo_keeps_in_a_subdir_does_not_fail_a_complete_snapshot(tmp_path):
    # FLUX.2 declares added_tokens.json and chat_template.jinja at the root and ships
    # both under tokenizer/, so no complete klein snapshot has them at the root. Download
    # patterns are a permissive filter, not a manifest, which is why only safetensors
    # count as a root requirement.
    for subdir in ("vae", "transformer", "text_encoder"):
        (tmp_path / subdir).mkdir(parents=True)
        (tmp_path / subdir / "model.safetensors").write_bytes(b"")
        (tmp_path / subdir / "config.json").write_text("{}")
    (tmp_path / "tokenizer").mkdir()
    (tmp_path / "tokenizer" / "added_tokens.json").write_text("{}")
    (tmp_path / "tokenizer" / "chat_template.jinja").write_text("")
    patterns = Flux2KleinWeightDefinition.get_download_patterns()
    subdirs = PathResolution._get_required_subdirs_with_safetensors(patterns)

    assert PathResolution._is_snapshot_complete(tmp_path, subdirs, patterns) is True


@pytest.mark.fast
def test_subdir_patterns_alone_still_need_no_root_files(tmp_path):
    # Krea 2 Raw ships transformer/ as shards, so every safetensors pattern has a "/"
    # and there is nothing at the root to require. This must keep passing.
    for subdir in ("transformer", "vae"):
        (tmp_path / subdir).mkdir(parents=True)
        (tmp_path / subdir / "model.safetensors").write_bytes(b"")
    (tmp_path / "model_index.json").write_text("{}")
    patterns = ["transformer/*.safetensors", "vae/*.safetensors", "model_index.json"]

    assert PathResolution._is_snapshot_complete(tmp_path, {"transformer", "vae"}, patterns) is True
