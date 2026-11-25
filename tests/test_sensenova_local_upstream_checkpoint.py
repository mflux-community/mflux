import json
from types import SimpleNamespace

import pytest
import transformers

from mflux.models.common.config.model_config import AVAILABLE_MODELS
from mflux.models.common.tokenizer.tokenizer import LanguageTokenizer
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
from mflux.models.sensenova import sensenova_initializer as module
from mflux.models.sensenova.sensenova_initializer import SenseNovaInitializer
from mflux.models.sensenova.weights.sensenova_weight_definition import SenseNovaWeightDefinition


# #993: a local copy of the upstream HuggingFace checkpoint (shards at the root, no mflux-save subdirs)
# loads like the upstream repo, from the local directory; an mflux-saved directory still loads as saved.
class _Recorder:
    def __init__(self, monkeypatch, real_tokenizer_loader: bool = False):
        self.load_single_repo_ids = []
        self.load_model_paths = []
        self.tokenizer_paths = []
        self.tokenizer_definitions = []
        self.raw_tokenizer_paths = []
        empty = LoadedWeights(components={}, meta_data=MetaData())

        def load_single(component, repo_id, file_pattern="*.safetensors"):
            self.load_single_repo_ids.append(repo_id)
            return empty

        def load(weight_definition, model_path=None, download_patterns=None):
            self.load_model_paths.append(model_path)
            return empty

        def load_all(definitions, model_path, **kwargs):
            self.tokenizer_definitions.append(definitions)
            self.tokenizer_paths.append(model_path)
            return {}

        monkeypatch.setattr(module.WeightLoader, "load_single", staticmethod(load_single))
        monkeypatch.setattr(module.WeightLoader, "load", staticmethod(load))

        def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
            self.raw_tokenizer_paths.append(pretrained_model_name_or_path)
            return SimpleNamespace(chat_template=None)

        if real_tokenizer_loader:
            # Only the third-party tokenizer load is replaced; TokenizerLoader resolves the path itself.
            monkeypatch.setattr(transformers.Qwen2Tokenizer, "from_pretrained", classmethod(from_pretrained))
        else:
            monkeypatch.setattr(module.TokenizerLoader, "load_all", staticmethod(load_all))
        # The real models are 8B; only where the weights and tokenizers come from is under test.
        monkeypatch.setattr(SenseNovaInitializer, "_init_models", staticmethod(lambda model, num_layers: None))
        monkeypatch.setattr(SenseNovaInitializer, "_apply_weights", staticmethod(lambda *a, **k: None))
        monkeypatch.setattr(SenseNovaInitializer, "_apply_loras", staticmethod(lambda *a, **k: None))


_SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")


class _Checkpoints:
    @staticmethod
    def upstream(root):
        root.mkdir()
        for name in _SHARDS:
            (root / name).write_bytes(b"")
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            (root / name).write_text("{}")
        weight_map = {"a.weight": _SHARDS[0], "b.weight": _SHARDS[1]}
        (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
        return root

    @staticmethod
    def saved(root):
        for subdir in ("transformer", "text_encoder"):
            (root / subdir).mkdir(parents=True)
            (root / subdir / "0.safetensors").write_bytes(b"")
        (root / "tokenizer").mkdir()
        return root

    @staticmethod
    def init(model_path, model=None):
        SenseNovaInitializer.init(
            model=model if model is not None else SimpleNamespace(),
            model_config=AVAILABLE_MODELS["sensenova-u1.5"],
            quantize=8,
            model_path=str(model_path),
        )


@pytest.mark.fast
def test_local_upstream_checkpoint_loads_its_weights_from_the_local_directory(tmp_path, monkeypatch):
    recorder = _Recorder(monkeypatch)
    local = _Checkpoints.upstream(tmp_path / "SenseNova-U1.5-8B-MoT")

    _Checkpoints.init(local)

    assert recorder.load_model_paths == []
    assert len(recorder.load_single_repo_ids) == 2
    assert all(str(repo_id) == str(local) for repo_id in recorder.load_single_repo_ids)


@pytest.mark.fast
def test_local_upstream_checkpoint_loads_its_tokenizer_from_the_local_directory(tmp_path, monkeypatch):
    recorder = _Recorder(monkeypatch)
    local = _Checkpoints.upstream(tmp_path / "SenseNova-U1.5-8B-MoT")

    _Checkpoints.init(local)

    assert [str(path) for path in recorder.tokenizer_paths] == [str(local)]
    assert recorder.tokenizer_definitions == [SenseNovaWeightDefinition.source_tokenizers()]


@pytest.mark.fast
def test_mflux_saved_directory_still_loads_as_a_saved_model(tmp_path, monkeypatch):
    recorder = _Recorder(monkeypatch)
    saved = _Checkpoints.saved(tmp_path / "saved")

    _Checkpoints.init(saved)

    assert recorder.load_single_repo_ids == []
    assert recorder.load_model_paths == [str(saved)]
    assert recorder.tokenizer_definitions == [SenseNovaWeightDefinition.get_tokenizers()]


@pytest.mark.fast
def test_local_upstream_checkpoint_without_the_shard_index_fails_with_a_clear_message(tmp_path, monkeypatch):
    recorder = _Recorder(monkeypatch)
    local = _Checkpoints.upstream(tmp_path / "SenseNova-U1.5-8B-MoT")
    (local / "model.safetensors.index.json").unlink()

    with pytest.raises(FileNotFoundError, match="model.safetensors.index.json"):
        _Checkpoints.init(local)

    assert recorder.load_single_repo_ids == []
    assert recorder.load_model_paths == []


@pytest.mark.fast
def test_local_upstream_checkpoint_missing_a_shard_named_in_the_index_fails_with_a_clear_message(tmp_path, monkeypatch):
    recorder = _Recorder(monkeypatch)
    local = _Checkpoints.upstream(tmp_path / "SenseNova-U1.5-8B-MoT")
    (local / _SHARDS[1]).unlink()

    with pytest.raises(FileNotFoundError, match="incomplete"):
        _Checkpoints.init(local)

    assert recorder.load_single_repo_ids == []
    assert recorder.load_model_paths == []


@pytest.mark.fast
@pytest.mark.parametrize("subdir", ["transformer", "text_encoder", "tokenizer"])
def test_a_directory_with_root_shards_and_a_save_subdir_loads_as_a_saved_model(tmp_path, monkeypatch, subdir):
    recorder = _Recorder(monkeypatch)
    local = _Checkpoints.upstream(tmp_path / "mixed")
    (local / subdir).mkdir()

    _Checkpoints.init(local)

    assert recorder.load_single_repo_ids == []
    assert recorder.load_model_paths == [str(local)]


@pytest.mark.fast
def test_a_directory_named_like_a_shard_does_not_make_an_upstream_checkpoint(tmp_path, monkeypatch):
    recorder = _Recorder(monkeypatch)
    local = _Checkpoints.upstream(tmp_path / "not-a-checkpoint")
    for name in _SHARDS:
        (local / name).unlink()
        (local / name).mkdir()

    _Checkpoints.init(local)

    assert recorder.load_single_repo_ids == []
    assert recorder.load_model_paths == [str(local)]


@pytest.mark.fast
def test_a_local_upstream_checkpoint_given_with_a_tilde_loads_from_that_directory(tmp_path, monkeypatch):
    recorder = _Recorder(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))
    _Checkpoints.upstream(tmp_path / "SenseNova-U1.5-8B-MoT")

    _Checkpoints.init("~/SenseNova-U1.5-8B-MoT")

    assert recorder.load_model_paths == []
    assert len(recorder.load_single_repo_ids) == 2


@pytest.mark.fast
@pytest.mark.parametrize("given", ["absolute", "tilde"])
def test_the_tokenizer_loader_resolves_a_local_upstream_checkpoint_root(tmp_path, monkeypatch, given):
    recorder = _Recorder(monkeypatch, real_tokenizer_loader=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    local = _Checkpoints.upstream(tmp_path / "SenseNova-U1.5-8B-MoT")
    model = SimpleNamespace()

    _Checkpoints.init(local if given == "absolute" else "~/SenseNova-U1.5-8B-MoT", model=model)

    assert isinstance(model.tokenizers["sensenova"], LanguageTokenizer)
    assert recorder.raw_tokenizer_paths
    assert set(recorder.raw_tokenizer_paths) == {str(local)}
