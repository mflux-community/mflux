# WeightLoader.load_single called snapshot_download directly instead of going through
# PathResolution, so the three components loaded that way — the FLUX and Z-Image controlnets
# and the Lens VAE — were the only weights in mflux that still needed the hub when everything
# was already cached, and the only ones a local directory could not stand in for.

from pathlib import Path
from unittest.mock import patch

import pytest

from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
from mflux.models.common.weights.loading.weight_loader import WeightLoader


@pytest.fixture
def component():
    return ComponentDefinition(name="controlnet", hf_subdir="", loading_mode="mlx_native")


@pytest.fixture
def loaded(monkeypatch):
    seen = {}

    def record(component, root_path):
        seen["component"] = component
        seen["root_path"] = root_path
        return "weights"

    monkeypatch.setattr(WeightLoader, "load_single_local", record)
    return seen


@pytest.mark.fast
def test_a_complete_cached_snapshot_is_used_without_the_hub(component, loaded, tmp_path):
    snapshot = tmp_path / "models--org--model" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "diffusion_pytorch_model.safetensors").touch()

    with patch("mflux.models.common.resolution.path_resolution.HF_HUB_CACHE", str(tmp_path)):
        with patch("mflux.models.common.resolution.path_resolution.snapshot_download") as download:
            WeightLoader.load_single(component=component, repo_id="org/model")

    download.assert_not_called()
    assert loaded["root_path"] == snapshot


@pytest.mark.fast
def test_a_local_directory_stands_in_for_the_repo(component, loaded, tmp_path):
    (tmp_path / "diffusion_pytorch_model.safetensors").touch()

    with patch("mflux.models.common.resolution.path_resolution.snapshot_download") as download:
        WeightLoader.load_single(component=component, repo_id=str(tmp_path))

    download.assert_not_called()
    assert loaded["root_path"] == tmp_path


@pytest.mark.fast
def test_an_uncached_repo_still_downloads(component, loaded, tmp_path):
    with patch("mflux.models.common.resolution.path_resolution.HF_HUB_CACHE", str(tmp_path / "empty")):
        with patch("mflux.models.common.resolution.path_resolution.snapshot_download") as download:
            download.return_value = str(tmp_path / "fetched")
            WeightLoader.load_single(component=component, repo_id="org/model", file_pattern="vae/*")

    assert download.call_args[1]["repo_id"] == "org/model"
    assert loaded["root_path"] == tmp_path / "fetched"


@pytest.mark.fast
def test_only_the_weight_pattern_decides_completeness(component, loaded, tmp_path):
    # The repos behind two of the three components have no config.json at the root — the
    # Lens VAE's FLUX.2-klein-4B keeps its under vae/ — so carrying the old allow_patterns
    # over would have marked every cached copy of them incomplete, which is the one case
    # this change exists to fix.
    snapshot = tmp_path / "models--org--model" / "snapshots" / "abc123"
    (snapshot / "vae").mkdir(parents=True)
    (snapshot / "vae" / "diffusion_pytorch_model.safetensors").touch()
    (snapshot / "vae" / "config.json").write_text("{}")

    with patch("mflux.models.common.resolution.path_resolution.HF_HUB_CACHE", str(tmp_path)):
        with patch("mflux.models.common.resolution.path_resolution.snapshot_download") as download:
            WeightLoader.load_single(component=component, repo_id="org/model", file_pattern="vae/*")

    download.assert_not_called()
    assert loaded["root_path"] == snapshot


@pytest.mark.fast
def test_the_patterns_reach_path_resolution(component, loaded, tmp_path, monkeypatch):
    seen = {}

    def record(path, patterns=None):
        seen["path"], seen["patterns"] = path, patterns
        return tmp_path

    monkeypatch.setattr(PathResolution, "resolve", record)
    WeightLoader.load_single(component=component, repo_id="org/model", file_pattern="vae/*")
    assert seen == {"path": "org/model", "patterns": ["vae/*"]}


@pytest.mark.fast
def test_resolving_nothing_names_the_component(component, monkeypatch):
    monkeypatch.setattr(PathResolution, "resolve", lambda path, patterns=None: None)
    with pytest.raises(ValueError, match="controlnet"):
        WeightLoader.load_single(component=component, repo_id="org/model")


@pytest.mark.fast
def test_weight_loader_no_longer_reaches_for_the_hub_itself():
    # The direct import is what let this path skip resolution; keep it gone.
    import mflux.models.common.weights.loading.weight_loader as module

    text = Path(module.__file__).read_text()
    assert "from huggingface_hub import snapshot_download" not in text
