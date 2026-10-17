import hashlib
import json
import os
import struct

import pytest

from mflux.models.common.checking.model_checker import ModelChecker
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition


def write_safetensors(path, metadata=None, names=None, data_offsets=None):
    header = {"__metadata__": metadata or {}}
    for name in names or ["weight"]:
        header[name] = {"dtype": "F32", "shape": [1], "data_offsets": data_offsets or [0, 4]}
    encoded = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\0\0\0\0")


def write_index(path, shard_names, metadata=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    index = {"weight_map": {f"weight_{index}": name for index, name in enumerate(shard_names)}}
    if metadata is not None:
        index["metadata"] = metadata
    path.write_text(json.dumps(index))


def link_to_missing_blob(link, content):
    # Make a Hugging Face snapshot entry that points to a blob that is not in the cache.
    blob = link.parents[3] / "blobs" / hashlib.sha256(content).hexdigest()
    blob.parent.mkdir(parents=True, exist_ok=True)
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(os.path.relpath(blob, link.parent))
    return blob


@pytest.fixture
def qwen_definition(monkeypatch):
    class Definition:
        @staticmethod
        def get_components():
            return [ComponentDefinition(name="transformer", hf_subdir="transformer")]

    monkeypatch.setattr(ModelChecker, "_definition_for", staticmethod(lambda key: Definition))
    monkeypatch.setattr(ModelChecker, "_resolve_model_key", staticmethod(lambda reference: "qwen-image"))


@pytest.mark.fast
def test_mflux_package_passes_when_all_parts_are_present(tmp_path, qwen_definition):
    root = tmp_path / "qwen-image-mflux-q6"
    write_safetensors(
        root / "transformer" / "0.safetensors",
        {"mflux_version": "0.21.0", "quantization_level": "6"},
        names=["weight_0"],
    )
    write_index(root / "transformer" / "model.safetensors.index.json", ["0.safetensors"])

    result = ModelChecker.check(str(root))

    assert result.check == "pass"
    assert result.format == "mflux"
    assert result.size_bytes > 0


@pytest.mark.fast
def test_supported_hugging_face_safetensors_package_passes(tmp_path, qwen_definition):
    root = tmp_path / "qwen-image"
    write_safetensors(root / "transformer" / "model.safetensors")

    result = ModelChecker.check(str(root))

    assert result.check == "pass"
    assert result.format == "safetensors"


@pytest.mark.fast
def test_incomplete_mflux_package_warns_and_prints_the_fix_command(tmp_path, qwen_definition):
    root = tmp_path / "qwen-image-mflux-q6"
    write_safetensors(root / "transformer" / "0.safetensors", {"mflux_version": "0.21.0"}, names=["weight_0"])
    write_index(root / "transformer" / "model.safetensors.index.json", ["0.safetensors", "1.safetensors"])

    result = ModelChecker.check(str(root))

    assert result.check == "warn"
    assert "1.safetensors" in " ".join(result.problems)
    assert f"mflux-check model --fix {root}" in ModelChecker.format_result(result, str(root))


@pytest.mark.fast
def test_cached_hugging_face_reference_never_downloads(tmp_path, qwen_definition, monkeypatch):
    snapshot = tmp_path / "models--mflux-community--qwen-image-mflux-q6" / "snapshots" / "abc123"
    write_safetensors(snapshot / "transformer" / "0.safetensors", {"mflux_version": "0.21.0"})
    monkeypatch.setattr("mflux.models.common.checking.model_checker.HF_HUB_CACHE", str(tmp_path))

    result = ModelChecker.check("mflux-community/qwen-image-mflux-q6")

    assert result.check == "pass"
    assert result.format == "mflux"


@pytest.fixture
def cached_snapshots(tmp_path, monkeypatch):
    cache = tmp_path / "models--mflux-community--qwen-image-mflux-q6" / "snapshots"
    old_snapshot = cache / "abc123"
    current_snapshot = cache / "def456"
    write_safetensors(
        current_snapshot / "transformer" / "0.safetensors", {"mflux_version": "0.21.0"}, names=["weight_0"]
    )
    write_index(current_snapshot / "transformer" / "model.safetensors.index.json", ["0.safetensors", "1.safetensors"])
    monkeypatch.setattr("mflux.models.common.checking.model_checker.HF_HUB_CACHE", str(tmp_path))
    return old_snapshot, current_snapshot


@pytest.mark.fast
def test_repair_restores_a_missing_blob_with_the_expected_hash(cached_snapshots, qwen_definition):
    old_snapshot, current_snapshot = cached_snapshots
    source = old_snapshot / "transformer" / "1.safetensors"
    write_safetensors(source, {"mflux_version": "0.21.0"}, names=["weight_1"])
    blob = link_to_missing_blob(current_snapshot / "transformer" / "1.safetensors", source.read_bytes())
    os.utime(old_snapshot, (0, 0))  # The checker resolves the newest snapshot.

    assert ModelChecker.check("mflux-community/qwen-image-mflux-q6").check == "warn"
    assert "Restored 1 shard" in ModelChecker.repair("mflux-community/qwen-image-mflux-q6")
    assert blob.read_bytes() == source.read_bytes()
    assert ModelChecker.check("mflux-community/qwen-image-mflux-q6").check == "pass"


@pytest.mark.fast
def test_repair_rejects_a_shard_from_another_revision(cached_snapshots, qwen_definition):
    old_snapshot, current_snapshot = cached_snapshots
    write_safetensors(old_snapshot / "transformer" / "1.safetensors", {"mflux_version": "0.20.0"}, names=["weight_1"])
    blob = link_to_missing_blob(current_snapshot / "transformer" / "1.safetensors", b"other revision")

    assert ModelChecker.repair("mflux-community/qwen-image-mflux-q6") == (
        "No local replacement is available. The check did not download files."
    )
    assert not blob.exists()


@pytest.mark.fast
def test_repair_skips_a_missing_shard_without_a_blob_hash(cached_snapshots, qwen_definition):
    old_snapshot, current_snapshot = cached_snapshots
    write_safetensors(old_snapshot / "transformer" / "1.safetensors", {"mflux_version": "0.21.0"}, names=["weight_1"])

    assert ModelChecker.repair("mflux-community/qwen-image-mflux-q6") == (
        "No local replacement is available. The check did not download files."
    )
    assert not (current_snapshot / "transformer" / "1.safetensors").exists()


@pytest.mark.fast
def test_repair_never_writes_through_a_link_outside_the_blobs_directory(tmp_path, cached_snapshots, qwen_definition):
    old_snapshot, current_snapshot = cached_snapshots
    source = old_snapshot / "transformer" / "1.safetensors"
    write_safetensors(source, {"mflux_version": "0.21.0"}, names=["weight_1"])
    outside = tmp_path / "outside" / hashlib.sha256(source.read_bytes()).hexdigest()
    outside.parent.mkdir()
    (current_snapshot / "transformer" / "1.safetensors").symlink_to(outside)

    assert ModelChecker.repair("mflux-community/qwen-image-mflux-q6") == (
        "No local replacement is available. The check did not download files."
    )
    assert not outside.exists()


@pytest.mark.fast
def test_index_metadata_identifies_an_mflux_package_with_a_missing_shard(tmp_path, qwen_definition):
    root = tmp_path / "qwen-image-mflux-q6"
    write_safetensors(root / "transformer" / "0.safetensors", names=["weight_0"])
    write_index(
        root / "transformer" / "model.safetensors.index.json",
        ["0.safetensors", "1.safetensors"],
        metadata={"quantization_level": "6", "mflux_version": "0.21.0"},
    )

    result = ModelChecker.check(str(root))

    assert result.check == "warn"
    assert result.format == "mflux"


@pytest.mark.fast
def test_invalid_quantization_level_fails(tmp_path, qwen_definition):
    root = tmp_path / "qwen-image-mflux-q6"
    write_safetensors(root / "transformer" / "0.safetensors", {"quantization_level": "q6"}, names=["weight_0"])

    result = ModelChecker.check(str(root))

    assert result.check == "fail"
    assert "Invalid quantization_level 'q6'" in " ".join(result.problems)


@pytest.mark.fast
def test_variant_selector_chooses_the_component_directory(tmp_path, monkeypatch):
    class Definition:
        @staticmethod
        def get_components():
            return [
                ComponentDefinition(
                    name="transformer",
                    hf_subdir="",
                    variant_selector=lambda root: ComponentDefinition(name="transformer", hf_subdir="transformer"),
                )
            ]

    monkeypatch.setattr(ModelChecker, "_definition_for", staticmethod(lambda key: Definition))
    monkeypatch.setattr(ModelChecker, "_resolve_model_key", staticmethod(lambda reference: "krea-2-raw"))
    root = tmp_path / "Krea-2-Raw"
    write_safetensors(root / "transformer" / "model.safetensors")

    result = ModelChecker.check(str(root))

    assert result.check == "pass"
    assert result.problems == ()


@pytest.mark.fast
def test_empty_package_with_missing_index_shards_warns(tmp_path, qwen_definition):
    root = tmp_path / "qwen-image-mflux-q6"
    write_index(root / "transformer" / "model.safetensors.index.json", ["0.safetensors"])

    result = ModelChecker.check(str(root))

    assert result.check == "warn"
    assert result.format == "mflux"
    assert "misses shards: 0.safetensors" in " ".join(result.problems)


@pytest.mark.fast
def test_index_rejects_tensor_absent_from_its_shard(tmp_path, qwen_definition):
    root = tmp_path / "qwen-image-mflux-q6"
    write_safetensors(root / "transformer" / "0.safetensors", {"mflux_version": "0.21.0"}, names=["other"])
    write_index(root / "transformer" / "model.safetensors.index.json", ["0.safetensors"])

    result = ModelChecker.check(str(root))

    assert result.check == "fail"
    assert "assigns tensors absent from shards: weight_0 in 0.safetensors" in " ".join(result.problems)


@pytest.mark.fast
def test_repair_skips_shard_with_unexpected_tensors(tmp_path, qwen_definition, monkeypatch):
    cache = tmp_path / "models--mflux-community--qwen-image-mflux-q6" / "snapshots"
    old_snapshot = cache / "abc123"
    current_snapshot = cache / "def456"
    write_safetensors(old_snapshot / "transformer" / "1.safetensors", {"mflux_version": "0.21.0"}, names=["other"])
    write_safetensors(
        current_snapshot / "transformer" / "0.safetensors", {"mflux_version": "0.21.0"}, names=["weight_0"]
    )
    write_index(current_snapshot / "transformer" / "model.safetensors.index.json", ["0.safetensors", "1.safetensors"])
    monkeypatch.setattr("mflux.models.common.checking.model_checker.HF_HUB_CACHE", str(tmp_path))

    assert ModelChecker.repair("mflux-community/qwen-image-mflux-q6") == (
        "No local replacement is available. The check did not download files."
    )
    assert not (current_snapshot / "transformer" / "1.safetensors").exists()


@pytest.mark.fast
def test_repair_rejects_non_local_shard_names(tmp_path, qwen_definition, monkeypatch):
    cache = tmp_path / "models--mflux-community--qwen-image-mflux-q6" / "snapshots"
    snapshot = cache / "abc123"
    write_safetensors(snapshot / "transformer" / "0.safetensors", {"mflux_version": "0.21.0"}, names=["weight_0"])
    (snapshot / "transformer" / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight_1": "../1.safetensors"}})
    )
    monkeypatch.setattr("mflux.models.common.checking.model_checker.HF_HUB_CACHE", str(tmp_path))

    assert ModelChecker.repair("mflux-community/qwen-image-mflux-q6") == (
        "No local replacement is available. The check did not download files."
    )
    assert not (snapshot / "1.safetensors").exists()


@pytest.mark.fast
def test_safetensors_rejects_offsets_beyond_the_payload(tmp_path, qwen_definition):
    root = tmp_path / "qwen-image"
    write_safetensors(root / "transformer" / "model.safetensors", data_offsets=[0, 5])

    result = ModelChecker.check(str(root))

    assert result.check == "fail"
    assert "has invalid data_offsets" in " ".join(result.problems)


@pytest.mark.fast
def test_safetensors_rejects_a_shape_that_does_not_fit_its_bytes(tmp_path):
    path = tmp_path / "model.safetensors"
    header = json.dumps({"weight": {"dtype": "F32", "shape": [10], "data_offsets": [0, 4]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0\0\0")

    with pytest.raises(ValueError, match="does not match its data_offsets"):
        ModelChecker._read_header(path)


@pytest.mark.fast
def test_safetensors_rejects_an_unknown_dtype(tmp_path):
    path = tmp_path / "model.safetensors"
    header = json.dumps({"weight": {"dtype": "F33", "shape": [1], "data_offsets": [0, 4]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0\0\0")

    with pytest.raises(ValueError, match="unknown dtype"):
        ModelChecker._read_header(path)


@pytest.mark.fast
def test_unrecognized_hugging_face_safetensors_package_passes_structural_check(tmp_path, monkeypatch):
    root = tmp_path / "flux-3-image-klein"
    write_safetensors(root / "transformer" / "model.safetensors")
    monkeypatch.setattr(ModelChecker, "_resolve_model_key", staticmethod(lambda reference: None))

    result = ModelChecker.check(str(root))

    assert result.check == "pass"
    assert result.format == "safetensors"
    assert result.problems == ()


# Set these variables to run the known-model cases on a host that has the packages:
#   MFLUX_CHECK_KNOWN_HF_MODEL     a Hugging Face reference in the local cache (safetensors format)
#   MFLUX_CHECK_KNOWN_MFLUX_MODEL  a local path to an MFlux-saved package
@pytest.mark.known_model
@pytest.mark.parametrize(
    ("variable", "format_name"),
    [("MFLUX_CHECK_KNOWN_HF_MODEL", "safetensors"), ("MFLUX_CHECK_KNOWN_MFLUX_MODEL", "mflux")],
)
def test_known_good_model_passes(variable, format_name):
    reference = os.environ.get(variable)
    if not reference:
        pytest.skip(f"{variable} is not set.")
    if ModelChecker._resolve(reference)[0] is None:
        pytest.skip(f"{reference} is not available on this host.")

    result = ModelChecker.check(reference)

    assert result.check == "pass"
    assert result.format == format_name
