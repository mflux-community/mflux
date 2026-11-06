# With -q N a registry entry loads mflux-community/<stem>-mflux-q<N> (#938). These tests pin the rules that keep
# everything else on the weights it loads today: a path, a third-party name, no -q, a model without a repo, a baked
# LoRA, a missing level, and an upstream download already in the cache.

import json

import pytest

from mflux.models.common.config.model_config import AVAILABLE_MODELS, ModelConfig
from mflux.models.common.resolution.prequantized_resolution import PrequantizedResolution
from mflux.utils.generated_image import GeneratedImage

REPO = "mflux-community/z-image-turbo-mflux-q4"

# Part of the mflux-community listing on 2026-11-06: stems and the levels each one has.
ORG_LISTING = {
    "anima-turbo": {"bf16", "q4", "q5", "q6", "q8"},
    "anima-base": {"bf16", "q4", "q5", "q6", "q8"},
    "flux-1-dev": {"bf16", "q3", "q4", "q5", "q6", "q8"},
    "flux-1-dev-depth": {"bf16", "q3", "q6", "q8"},
    "flux-2-klein-base-4b": {"bf16", "q3", "q4", "q5", "q6", "q8"},
    "flux2-klein-4b": {"bf16", "q3", "q4", "q5", "q6", "q8"},
    "krea-2-turbo": {"bf16", "q3", "q4", "q5", "q6", "q8"},
    "qwen-image-2512": {"bf16", "q3", "q4", "q5", "q6", "q8"},
    "qwen-image-edit-2509": {"bf16", "q3", "q4", "q5", "q6", "q8"},
    "z-image-turbo": {"bf16", "q3", "q4", "q5", "q6", "q8"},
    "fibo-2-0-dev": {"bf16", "q3", "q4", "q5", "q6", "q8"},
}


@pytest.fixture(autouse=True)
def _nothing_cached(monkeypatch):
    monkeypatch.setattr(PrequantizedResolution, "_cached", staticmethod(lambda repo_id: False))
    yield
    GeneratedImage.weights_repo = None


def _cache(monkeypatch, *repo_ids):
    monkeypatch.setattr(PrequantizedResolution, "_cached", staticmethod(lambda repo_id: repo_id in repo_ids))


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
    assert PrequantizedResolution.resolve(config, None, 4)[0] == config.model_name
    assert REPO in capsys.readouterr().err


@pytest.mark.fast
def test_rule_6_does_not_apply_when_the_repo_is_cached_too(monkeypatch):
    _cache(monkeypatch, ModelConfig.z_image_turbo().model_name, REPO)
    assert PrequantizedResolution.resolve(ModelConfig.z_image_turbo(), None, 4)[0] == REPO


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
def test_every_listed_stem_and_level_resolves_to_a_repo_in_the_org():
    from mflux.cli.defaults.defaults import QUANTIZE_CHOICES

    checked = 0
    for config in AVAILABLE_MODELS.values():
        stem = config.mflux_repo_stem
        if stem not in ORG_LISTING:
            continue
        for level in config.mflux_repo_missing:
            assert level not in ORG_LISTING[stem], f"{stem} has {level}"
        for quantize in QUANTIZE_CHOICES:
            root = PrequantizedResolution.resolve(config, None, quantize)[0]
            if f"q{quantize}" in config.mflux_repo_missing:
                assert root == config.model_name
            else:
                assert f"q{quantize}" in ORG_LISTING[stem], f"{stem} q{quantize}"
                assert root == PrequantizedResolution.repo_id(stem, quantize)
            checked += 1
    assert checked >= 40


@pytest.mark.fast
def test_the_cache_check_finds_weights_in_component_folders(monkeypatch, tmp_path):
    import mflux.models.common.resolution.path_resolution as path_resolution

    monkeypatch.undo()  # the real _cached
    monkeypatch.setattr(path_resolution, "HF_HUB_CACHE", str(tmp_path))
    snapshot = tmp_path / "models--org--model" / "snapshots" / "abc"
    (snapshot / "transformer").mkdir(parents=True)
    (snapshot / "vae").mkdir()
    (snapshot / "vae" / "diffusion_pytorch_model.safetensors").touch()
    index = {"weight_map": {"a": "0.safetensors", "b": "1.safetensors"}}
    (snapshot / "transformer" / "model.safetensors.index.json").write_text(json.dumps(index))
    (snapshot / "transformer" / "0.safetensors").touch()
    assert not PrequantizedResolution._cached("org/model")  # one shard missing
    (snapshot / "transformer" / "1.safetensors").touch()
    assert PrequantizedResolution._cached("org/model")
    assert not PrequantizedResolution._cached("org/absent")
