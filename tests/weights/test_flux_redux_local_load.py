# A redux checkpoint written by mflux-save keeps its encoders under image_encoder/ and
# image_embedder/. The initializer used to fetch them from the remote repo
# unconditionally, so reloading a saved checkpoint silently swapped the saved (e.g. -q
# quantized) encoder weights for the hub's and broke offline use. These tests pin the
# routing decision, mirroring test_flux_controlnet_local_load.py: a checkpoint carrying
# the redux subdirs loads them locally, everything else keeps the hub path.
#
# They also pin the base-weights location: the dev-redux config names the adapter repo
# (encoder + embedder only), so without a model_path the base must come from FLUX.1-dev,
# not from a repo that has no transformer/vae at all.

import pytest

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.flux.flux_initializer import FluxInitializer


@pytest.fixture
def loader_calls(monkeypatch):
    seen = {}

    def record(weight_definition, model_path=None, download_patterns=None):
        seen["load"] = {"weight_definition": weight_definition, "model_path": model_path}
        return "redux-weights"

    monkeypatch.setattr(WeightLoader, "load", record)
    return seen


@pytest.mark.fast
def test_a_saved_checkpoint_loads_its_own_redux_encoders(loader_calls, tmp_path):
    (tmp_path / "image_encoder").mkdir()
    (tmp_path / "image_encoder" / "0.safetensors").touch()
    (tmp_path / "image_embedder").mkdir()
    (tmp_path / "image_embedder" / "0.safetensors").touch()

    weights = FluxInitializer._load_redux_weights(model_path=str(tmp_path))

    assert loader_calls["load"]["model_path"] == str(tmp_path)
    assert weights == "redux-weights"


@pytest.mark.fast
def test_a_half_saved_redux_checkpoint_still_loads_locally(loader_calls, tmp_path):
    # A checkpoint with one redux subdir was written by mflux-save (or is half-copied);
    # either way it must not be silently completed from the hub. Routing local lets
    # WeightLoader name the missing half from disk.
    (tmp_path / "image_encoder").mkdir()
    (tmp_path / "image_encoder" / "0.safetensors").touch()

    FluxInitializer._load_redux_weights(model_path=str(tmp_path))

    assert loader_calls["load"]["model_path"] == str(tmp_path)


@pytest.mark.fast
def test_a_builtin_name_downloads_the_redux_encoders(loader_calls):
    FluxInitializer._load_redux_weights(model_path=None)

    assert loader_calls["load"]["model_path"] == ModelConfig.dev_redux().model_name


@pytest.mark.fast
def test_a_local_path_without_redux_encoders_falls_back_to_the_hub(loader_calls, tmp_path):
    # A base-only checkpoint (vae/transformer/text encoders, e.g. from `--model dev`) has
    # no redux subdirs; the encoders still have to come from somewhere, so the hub path stays.
    (tmp_path / "transformer").mkdir()

    FluxInitializer._load_redux_weights(model_path=str(tmp_path))

    assert loader_calls["load"]["model_path"] == ModelConfig.dev_redux().model_name


@pytest.mark.fast
@pytest.mark.parametrize(
    ("model_config", "model_path", "expected_base_path"),
    [
        # The adapter repo ships no base weights, so dev-redux falls back to its base.
        (ModelConfig.dev_redux(), None, ModelConfig.dev().model_name),
        # Other configs keep their own model_name.
        (ModelConfig.dev(), None, ModelConfig.dev().model_name),
        # An explicit model path always wins, even for the dev-redux config.
        (ModelConfig.dev_redux(), "/models/dev-redux-8bit", "/models/dev-redux-8bit"),
    ],
)
def test_redux_base_weights_come_from_a_repo_that_has_them(monkeypatch, model_config, model_path, expected_base_path):
    seen = {}

    def record_init(model, model_config, quantize, model_path=None, **kwargs):
        seen["model_path"] = model_path

    monkeypatch.setattr(FluxInitializer, "init", record_init)
    monkeypatch.setattr(FluxInitializer, "_load_redux_weights", lambda model_path: None)
    monkeypatch.setattr(WeightApplier, "apply_and_quantize", lambda **kwargs: None)

    class ModelStub:
        pass

    FluxInitializer.init_redux(model=ModelStub(), model_config=model_config, quantize=None, model_path=model_path)

    assert seen["model_path"] == expected_base_path
