import os
import re

import pytest

from mflux.models.common.config.model_config import AVAILABLE_MODELS
from mflux.models.common.pid_decoder import pid_decoder
from mflux.models.common.resolution import path_resolution
from mflux.models.common.resolution.model_revisions import ModelRevisions
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.flux2.model.flux2_text_encoder import small_text_encoder
from mflux.models.flux2.model.flux2_vae import small_decoder
from mflux.models.hed import hed
from mflux.models.lens.model.text_encoder import lens_gpt_oss_encoder
from mflux.models.lens.variants.txt2img import lens_image
from mflux.models.openpose import openpose
from mflux.models.qwen21.model.qwen21_small_text_encoder.zen_image_edit_weights import ZenImageEditWeights
from mflux.models.z_image.model.z_image_text_encoder import small_text_encoder as zimage_small_text_encoder

REPO = "org/model"
PINNED = "a" * 40
OTHER = "b" * 40


def _registry_repos() -> set[str]:
    repos = set()
    for config in AVAILABLE_MODELS.values():
        for field in ("model_name", "controlnet_model", "custom_transformer_model"):
            value = getattr(config, field, None)
            if value and "/" in value:
                repos.add(value)
    return repos


def _snapshot(cache, revision: str, mtime: float) -> None:
    snapshot = cache / f"models--{REPO.replace('/', '--')}" / "snapshots" / revision
    (snapshot / "transformer").mkdir(parents=True)
    (snapshot / "transformer" / "model.safetensors").write_bytes(b"")
    os.utime(snapshot, (mtime, mtime))


@pytest.mark.fast
def test_every_built_in_repo_is_pinned():
    # A repo that is not pinned loads whatever snapshot is cached, so a new model must add its pin.
    helper_repos = {
        pid_decoder.PID_REPO_DEFAULT,
        pid_decoder.GEMMA2_REPO,
        hed._HF_REPO,
        openpose._HF_REPO,
        lens_gpt_oss_encoder.DEFAULT_ENCODER_REPO,
        lens_image.VAE_REPO,
        small_text_encoder.ADAPTER_REPO,
        small_text_encoder.SMALL_ENCODER_REPO,
        small_decoder.SMALL_DECODER_REPO,
        zimage_small_text_encoder.SMALL_ENCODER_REPO,
        ZenImageEditWeights.REPO,
        "briaai/FIBO-vlm",
    }
    missing = sorted((_registry_repos() | helper_repos) - ModelRevisions.PINNED.keys())
    assert missing == []


@pytest.mark.fast
def test_every_pin_is_a_full_commit_hash():
    # A branch or tag name would move with the repo; the cache keeps snapshots under the full hash.
    assert [repo for repo, rev in ModelRevisions.PINNED.items() if not re.fullmatch(r"[0-9a-f]{40}", rev)] == []


@pytest.mark.fast
def test_a_pinned_repo_loads_only_its_pinned_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(path_resolution, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setitem(ModelRevisions.PINNED, REPO, PINNED)
    _snapshot(tmp_path, PINNED, mtime=1_000)
    _snapshot(tmp_path, OTHER, mtime=2_000)

    found = PathResolution._find_complete_cached_snapshot(REPO, ["transformer/*.safetensors"])

    assert found is not None and found.name == PINNED


@pytest.mark.fast
def test_a_pinned_repo_ignores_a_cached_snapshot_of_another_commit(tmp_path, monkeypatch):
    # The case that broke the Qwen-Image-2.1 golden: weights cached before the repo changed.
    monkeypatch.setattr(path_resolution, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setitem(ModelRevisions.PINNED, REPO, PINNED)
    _snapshot(tmp_path, OTHER, mtime=2_000)

    assert PathResolution._find_complete_cached_snapshot(REPO, ["transformer/*.safetensors"]) is None


@pytest.mark.fast
def test_an_unpinned_repo_keeps_the_newest_complete_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(path_resolution, "HF_HUB_CACHE", str(tmp_path))
    _snapshot(tmp_path, PINNED, mtime=1_000)
    _snapshot(tmp_path, OTHER, mtime=2_000)

    found = PathResolution._find_complete_cached_snapshot(REPO, ["transformer/*.safetensors"])

    assert found is not None and found.name == OTHER


@pytest.mark.fast
@pytest.mark.parametrize("pinned, expected", [(True, PINNED), (False, None)], ids=["pinned", "unpinned"])
def test_a_download_asks_for_the_pinned_commit(tmp_path, monkeypatch, pinned, expected):
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(path_resolution, "snapshot_download", fake_snapshot_download)
    if pinned:
        monkeypatch.setitem(ModelRevisions.PINNED, REPO, PINNED)

    PathResolution._execute(path_resolution.PathAction.HUGGINGFACE, REPO, ["*.safetensors"])

    assert calls[0]["revision"] == expected
