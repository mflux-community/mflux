# With -q N a registry entry loads mflux-community/<stem>-mflux-q<N> (#938). These tests pin the rules that keep
# everything else on the weights it loads today: a path, a third-party name, no -q, a model without a repo, a baked
# LoRA, a missing level, and an upstream download already in the cache.

import json
import sys
from pathlib import Path

import pytest

from mflux.models.common.config.model_config import AVAILABLE_MODELS, ModelConfig
from mflux.models.common.resolution.model_revisions import ModelRevisions
from mflux.models.common.resolution.prequantized_resolution import PrequantizedResolution
from mflux.utils.generated_image import GeneratedImage

REPO = "mflux-community/z-image-turbo-mflux-q4"
PATTERNS = ["transformer/*.safetensors", "vae/*.safetensors"]
SAVED = ["transformer", "vae"]

# The mflux-community model repos on 2026-11-06, one stem per line with the levels it has.
LISTING = Path(__file__).parent / "fixtures" / "mflux_community_repos.txt"


def _org_listing() -> dict[str, set[str]]:
    stem_levels = (line.split() for line in LISTING.read_text().splitlines() if line.strip())
    return {stem: set(levels) for stem, *levels in stem_levels}


@pytest.fixture(autouse=True)
def _nothing_cached(monkeypatch):
    monkeypatch.setattr(PrequantizedResolution, "_upstream_cached", staticmethod(lambda repo_id, patterns: False))
    monkeypatch.setattr(PrequantizedResolution, "_repo_cached", staticmethod(lambda repo_id, subdirs: False))
    yield
    GeneratedImage.weights_repo = None


def _cache(monkeypatch, *repo_ids):
    monkeypatch.setattr(PrequantizedResolution, "_upstream_cached", staticmethod(lambda repo_id, patterns: repo_id in repo_ids))  # fmt: off
    monkeypatch.setattr(PrequantizedResolution, "_repo_cached", staticmethod(lambda repo_id, subdirs: repo_id in repo_ids))  # fmt: off


def _real_cache(monkeypatch, tmp_path):
    # The real cache checks, against an HF cache under tmp_path
    import mflux.models.common.resolution.path_resolution as path_resolution

    monkeypatch.undo()
    monkeypatch.setattr(path_resolution, "HF_HUB_CACHE", str(tmp_path))


def _snapshot(tmp_path, repo_id: str):
    # A pinned repo is read only at its pinned commit
    revision = ModelRevisions.get(repo_id) or "abc"
    snapshot = tmp_path / f"models--{repo_id.replace('/', '--')}" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    return snapshot


def _files(root, *names):
    for name in names:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).touch()


def _saved_component(root, subdir, shards=2):
    index = {"weight_map": {f"w{i}": f"{i}.safetensors" for i in range(shards)}}
    (root / subdir).mkdir(parents=True, exist_ok=True)
    (root / subdir / "model.safetensors.index.json").write_text(json.dumps(index))
    _files(root, *(f"{subdir}/{i}.safetensors" for i in range(shards)))


@pytest.mark.fast
def test_rule_0_a_config_inferred_from_a_third_party_name_is_not_redirected():
    config = ModelConfig.from_name("someone/z-image-turbo-finetune", base_model="z-image-turbo")
    assert config.base_model is not None
    assert config.mflux_repo_stem == "z-image-turbo"  # carried from the base
    assert PrequantizedResolution.resolve(config, None, 4)[0] == "someone/z-image-turbo-finetune"


@pytest.mark.fast
def test_rule_0_a_substring_inferred_config_is_not_redirected():
    config = ModelConfig.from_name("someone/my-z-image-turbo-merge")
    assert config.base_model is not None
    assert PrequantizedResolution.resolve(config, None, 4)[0] == "someone/my-z-image-turbo-merge"


@pytest.mark.fast
@pytest.mark.parametrize("model_path", ["/models/z-image-turbo-q4", "~/saved", "someone/z-image-turbo-mflux-q4", REPO])
def test_rule_1_a_given_path_or_repo_is_untouched(model_path):
    assert PrequantizedResolution.resolve(ModelConfig.z_image_turbo(), model_path, 4)[0] == model_path


@pytest.mark.fast
def test_rule_2_without_q_the_upstream_repo_loads():
    config = ModelConfig.z_image_turbo()
    assert PrequantizedResolution.resolve(config, None, None)[0] == config.model_name


@pytest.mark.fast
def test_rule_3_a_model_without_a_stem_loads_upstream():
    for key in ("iris-3b", "dev-controlnet-canny", "z-image-turbo-controlnet-union-2.1", "dev-redux", "qwen-image-2.1"):
        config = AVAILABLE_MODELS[key]
        assert config.mflux_repo_stem is None
        assert PrequantizedResolution.resolve(config, None, 4)[0] == config.model_name


@pytest.mark.fast
def test_rule_4_a_baked_lora_keeps_the_upstream_weights():
    config = ModelConfig.z_image_turbo()
    assert PrequantizedResolution.resolve(config, None, 4, has_loras=True, bake_lora=True)[0] == config.model_name
    # Unbaked, the adapter runs next to the stored quantization, so the repo still serves
    assert PrequantizedResolution.resolve(config, None, 4, has_loras=True, bake_lora=False)[0] == REPO


@pytest.mark.fast
def test_rule_5_a_missing_level_loads_upstream():
    depth = AVAILABLE_MODELS["dev-depth"]
    assert PrequantizedResolution.resolve(depth, None, 4)[0] == depth.model_name
    assert PrequantizedResolution.resolve(depth, None, 8)[0] == "mflux-community/flux-1-dev-depth-mflux-q8"


@pytest.mark.fast
def test_rule_6_a_cached_upstream_is_used_with_a_hint(monkeypatch, capsys):
    config = ModelConfig.z_image_turbo()
    _cache(monkeypatch, config.model_name)
    assert PrequantizedResolution.resolve(config, None, 4, upstream_patterns=PATTERNS)[0] == config.model_name
    assert REPO in capsys.readouterr().err


@pytest.mark.fast
def test_rule_6_does_not_apply_when_the_repo_is_cached_too(monkeypatch):
    _cache(monkeypatch, ModelConfig.z_image_turbo().model_name, REPO)
    assert (
        PrequantizedResolution.resolve(
            ModelConfig.z_image_turbo(), None, 4, upstream_patterns=PATTERNS, saved_subdirs=SAVED
        )[0]
        == REPO
    )


@pytest.mark.fast
def test_rule_6_does_not_apply_without_the_loader_patterns(monkeypatch, tmp_path):
    # Without the patterns the loader reads, a cached snapshot says nothing about this model
    _real_cache(monkeypatch, tmp_path)
    config = ModelConfig.z_image_turbo()
    _files(_snapshot(tmp_path, config.model_name), "transformer/a.safetensors", "vae/a.safetensors")
    assert PrequantizedResolution.resolve(config, None, 4)[0] == REPO
    assert PrequantizedResolution.resolve(config, None, 4, upstream_patterns=PATTERNS)[0] == config.model_name


@pytest.mark.fast
def test_rule_6_an_upstream_snapshot_with_only_the_vae_is_not_cached(monkeypatch, tmp_path):
    # An interrupted upstream download
    from mflux.models.z_image.weights.z_image_weight_definition import ZImageWeightDefinition

    _real_cache(monkeypatch, tmp_path)
    config = ModelConfig.z_image_turbo()
    _files(_snapshot(tmp_path, config.model_name), "vae/diffusion_pytorch_model.safetensors")
    layout = PrequantizedResolution.layout(ZImageWeightDefinition)
    assert PrequantizedResolution.resolve(config, None, 4, **layout)[0] == REPO


@pytest.mark.fast
def test_rule_6_another_anima_variant_in_the_cache_is_not_this_one(monkeypatch, tmp_path):
    from mflux.models.anima.anima_initializer import AnimaInitializer
    from mflux.models.anima.weights.anima_weight_definition import AnimaWeightDefinition as Anima

    _real_cache(monkeypatch, tmp_path)
    turbo, base = AVAILABLE_MODELS["anima-turbo"], AVAILABLE_MODELS["anima-base"]
    assert turbo.model_name == base.model_name == Anima.REPO
    _files(
        _snapshot(tmp_path, Anima.REPO),
        f"{Anima.TRANSFORMER_DIR}/{Anima.TRANSFORMER_FILES['anima-base']}",
        f"{Anima.TEXT_ENCODER_DIR}/{Anima.TEXT_ENCODER_FILE}",
        f"{Anima.VAE_DIR}/{Anima.VAE_FILE}",
    )
    turbo_layout = PrequantizedResolution.layout(Anima, AnimaInitializer.upstream_patterns(turbo))
    base_layout = PrequantizedResolution.layout(Anima, AnimaInitializer.upstream_patterns(base))
    assert PrequantizedResolution.resolve(turbo, None, 4, **turbo_layout)[0] == "mflux-community/anima-turbo-mflux-q4"
    assert PrequantizedResolution.resolve(base, None, 4, **base_layout)[0] == Anima.REPO


@pytest.mark.fast
def test_the_upstream_cache_check_needs_every_indexed_shard(monkeypatch, tmp_path):
    _real_cache(monkeypatch, tmp_path)
    snapshot = _snapshot(tmp_path, "org/model")
    _files(snapshot, "vae/diffusion_pytorch_model.safetensors")
    _saved_component(snapshot, "transformer")
    (snapshot / "transformer" / "1.safetensors").unlink()
    assert not PrequantizedResolution._upstream_cached("org/model", PATTERNS)
    _files(snapshot, "transformer/1.safetensors")
    assert PrequantizedResolution._upstream_cached("org/model", PATTERNS)
    assert not PrequantizedResolution._upstream_cached("org/absent", PATTERNS)


@pytest.mark.fast
def test_the_repo_cache_check_needs_every_saved_component(monkeypatch, tmp_path):
    _real_cache(monkeypatch, tmp_path)
    snapshot = _snapshot(tmp_path, REPO)
    _saved_component(snapshot, "vae")
    assert not PrequantizedResolution._repo_cached(REPO, SAVED)  # transformer/ missing
    _saved_component(snapshot, "transformer")
    (snapshot / "transformer" / "1.safetensors").unlink()
    assert not PrequantizedResolution._repo_cached(REPO, SAVED)  # one shard missing
    _files(snapshot, "transformer/1.safetensors")
    assert PrequantizedResolution._repo_cached(REPO, SAVED)
    assert not PrequantizedResolution._repo_cached(REPO, None)


@pytest.mark.fast
def test_a_dangling_snapshot_entry_does_not_break_the_cache_checks(monkeypatch, tmp_path):
    # A broken symlink in snapshots/ has no mtime; it is skipped instead of raising before -q can redirect
    _real_cache(monkeypatch, tmp_path)
    assert ModelRevisions.get(REPO) is None  # unpinned: every snapshot is sorted by mtime
    snapshot = _snapshot(tmp_path, REPO)
    _saved_component(snapshot, "transformer")
    _saved_component(snapshot, "vae")
    (snapshot.parent / "dangling").symlink_to(tmp_path / "gone")
    assert PrequantizedResolution._repo_cached(REPO, SAVED)

    config = ModelConfig.z_image_turbo()
    assert ModelRevisions.get(config.model_name) is not None  # pinned: only the pinned snapshot is read
    upstream = _snapshot(tmp_path, config.model_name)
    (upstream.parent / "dangling").symlink_to(tmp_path / "gone")
    assert not PrequantizedResolution._upstream_cached(config.model_name, PATTERNS)
    assert PrequantizedResolution.resolve(config, None, 4, upstream_patterns=PATTERNS)[0] == REPO
    _files(upstream, "transformer/a.safetensors", "vae/a.safetensors")
    assert PrequantizedResolution._upstream_cached(config.model_name, PATTERNS)


@pytest.mark.fast
def test_rule_7_q_loads_the_pre_quantized_repo():
    assert PrequantizedResolution.resolve(ModelConfig.z_image_turbo(), None, 4)[0] == REPO
    assert PrequantizedResolution.resolve(AVAILABLE_MODELS["qwen-image"], None, 6)[0] == (
        "mflux-community/qwen-image-2512-mflux-q6"
    )


@pytest.mark.fast
def test_disabled_keeps_upstream_and_restores():
    config = ModelConfig.z_image_turbo()
    with PrequantizedResolution.disabled():
        assert PrequantizedResolution.resolve(config, None, 4)[0] == config.model_name
    assert PrequantizedResolution.resolve(config, None, 4)[0] == REPO


@pytest.mark.fast
def test_weights_root_records_the_repo_for_the_sidecar_only_when_redirected():
    config = ModelConfig.z_image_turbo()
    assert PrequantizedResolution.weights_root(config, None, 4) == REPO
    assert GeneratedImage.weights_repo == REPO
    assert PrequantizedResolution.weights_root(config, None, None) == config.model_name
    assert GeneratedImage.weights_repo is None
    assert PrequantizedResolution.weights_root(config, REPO, 4) == REPO
    assert GeneratedImage.weights_repo is None


@pytest.mark.fast
def test_every_stem_and_level_resolves_to_a_repo_in_the_org():
    from mflux.cli.defaults.defaults import QUANTIZE_CHOICES

    listing = _org_listing()
    assert len(listing) == 50
    for key, config in AVAILABLE_MODELS.items():
        stem = config.mflux_repo_stem
        if stem is None:
            continue
        assert stem in listing, f"{key}: no {stem} repos in the org"
        for level in config.mflux_repo_missing:
            assert level not in listing[stem], f"{key}: {stem} has {level}"
        for quantize in QUANTIZE_CHOICES:
            root = PrequantizedResolution.resolve(config, None, quantize)[0]
            if f"q{quantize}" in config.mflux_repo_missing:
                assert root == config.model_name
            else:
                assert f"q{quantize}" in listing[stem], f"{key}: {stem} has no q{quantize}"
                assert root == PrequantizedResolution.repo_id(stem, quantize)


@pytest.mark.fast
def test_mflux_save_runs_with_the_pre_quantized_repos_off(monkeypatch, tmp_path):
    from mflux.models.common.cli import save

    seen = {}
    model_class = save.MODEL_CLASSES["z-image-turbo"]

    def fake_init(self, *args, **kwargs):
        seen["enabled"] = PrequantizedResolution._enabled
        seen["quantize"] = kwargs.get("quantize")

    monkeypatch.setattr(model_class, "__init__", fake_init)
    monkeypatch.setattr(model_class, "save_model", lambda self, base_path: None)
    monkeypatch.setattr(save.SavedModelInfo, "write", lambda **kwargs: None)
    argv = ["mflux-save", "--model", "z-image-turbo", "-q", "4", "--path", str(tmp_path / "saved")]
    monkeypatch.setattr(sys, "argv", argv)
    save.main()
    assert seen == {"enabled": False, "quantize": 4}
    assert PrequantizedResolution._enabled


@pytest.mark.fast
def test_mflux_train_runs_with_the_pre_quantized_repos_off(monkeypatch, tmp_path):
    from mflux.models.common.cli import train

    seen = {}
    config = tmp_path / "train.json"
    config.write_text("{}")
    monkeypatch.setattr(train.TrainingRunner, "train", staticmethod(lambda **kwargs: seen.update(enabled=PrequantizedResolution._enabled)))  # fmt: off
    monkeypatch.setattr(sys, "argv", ["mflux-train", "--config", str(config), "-q", "4"])
    train.main()
    assert seen == {"enabled": False}
    assert PrequantizedResolution._enabled


def test_a_level_the_org_does_not_publish_stays_upstream(monkeypatch):
    # The Python API accepts bit widths the CLI does not offer, e.g. 2: no repo exists for them
    monkeypatch.setattr(PrequantizedResolution, "_upstream_cached", staticmethod(lambda *a, **k: False))
    config = ModelConfig.from_name("dev")
    root, _ = PrequantizedResolution.resolve(config, None, 2)
    assert root == config.model_name
