import inspect

import mlx.core as mx
import pytest
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
from mlx.utils import tree_flatten

from mflux.models.common.pid_decoder import pid_decoder as pid_decoder_module
from mflux.models.common.pid_decoder.pid_decoder import (
    PID_CHECKPOINT_VARIANTS,
    PidDecoder,
    _assert_full_weight_coverage,
)
from mflux.models.common.pid_decoder.pixdit.pixdit_network import PidNet


def test_decode_raises_on_channel_mismatch():
    net = PidNet(
        lq_latent_channels=16,
        hidden_size=32,
        pixel_hidden_size=8,
        patch_depth=1,
        pixel_depth=1,
        num_groups=4,
        patch_size=4,
        txt_embed_dim=16,
        txt_max_length=8,
        rope_ref_h=64,
        rope_ref_w=64,
        lq_hidden_dim=8,
        lq_num_res_blocks=1,
    )
    decoder = PidDecoder(pid_net=net, caption_encoder=None)  # caption path unreached before the shape check
    bad_latent = mx.random.normal((1, 4, 8, 8))  # 4 channels, expected 16
    with pytest.raises(ValueError, match="channels"):
        decoder.decode(bad_latent, caption="test")


def test_from_pretrained_raises_on_unknown_variant():
    """Test that from_pretrained raises ValueError for an unknown variant."""
    with pytest.raises(ValueError, match="Unknown PiD variant"):
        PidDecoder.from_pretrained(variant="invalid-variant")


def test_pid_checkpoint_variants_contains_expected_keys():
    """Test that PID_CHECKPOINT_VARIANTS contains the expected supported variants."""
    expected_variants = {"flux", "flux2", "qwen-image"}
    actual_variants = set(PID_CHECKPOINT_VARIANTS.keys())
    assert expected_variants == actual_variants, f"Expected variants {expected_variants}, got {actual_variants}"


def test_wired_variants_never_reach_the_lq_projection_fold_permute_branch():
    """LQProjection2D's channel-folding permute (pixdit_lq_projection.py::
    _align_latent_to_patch_grid) is the one place a correctly-shaped-but-wrongly-ordered
    channel wiring could hide from _assert_full_weight_coverage -- ruled out as the cause of
    a reported flux2-specific colour bias because it only runs when z_to_patch_ratio < 1.
    From_pretrained's real config (sr_scale=4, latent_spatial_down_factor=8, patch_size=16)
    gives z_to_patch_ratio=2 for every variant, so all three take the plain nearest-upsample
    path instead, unpermuted. Pin the config values so this conclusion doesn't go stale."""
    source = inspect.getsource(PidDecoder.from_pretrained)
    assert "sr_scale=4" in source
    assert "latent_spatial_down_factor=8" in source
    assert "patch_size=16" in source
    sr_scale, latent_spatial_down_factor, patch_size = 4, 8, 16
    z_to_patch_ratio = (sr_scale * latent_spatial_down_factor) / patch_size
    assert z_to_patch_ratio >= 1, (
        "a wired variant now takes the fold-permute branch -- re-check the colour bias analysis"
    )


def test_every_wired_vae_declares_a_variant_matching_its_latent_space():
    """Each VAE opting into --pid-decode names a real variant whose checkpoint expects exactly
    that VAE's latent shape. A mismatch here is the failure mode that only shows up after an
    ~8GB download, so pin it on class attributes -- no instantiation, no network."""
    from mflux.models.flux.model.flux_vae.vae import VAE as FluxVAE
    from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
    from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
    from mflux.models.z_image.model.z_image_vae.vae import VAE as ZImageVAE

    for vae_cls in (FluxVAE, Flux2VAE, QwenVAE, ZImageVAE):
        variant = vae_cls.pid_variant
        assert variant in PID_CHECKPOINT_VARIANTS, f"{vae_cls.__module__}.{vae_cls.__name__}: {variant!r}"
        _, lq_latent_channels = PID_CHECKPOINT_VARIANTS[variant]
        assert vae_cls.latent_channels == lq_latent_channels, (
            f"{vae_cls.__name__} has {vae_cls.latent_channels} latent channels but variant "
            f"{variant!r}'s checkpoint expects {lq_latent_channels}"
        )
        # PidDecoder derives the output size as zH * VAE_COMPRESSION * SR_SCALE, so a VAE with a
        # different spatial scale would silently super-resolve to the wrong dimensions.
        assert getattr(vae_cls, "spatial_scale", 8) == PidDecoder.VAE_COMPRESSION, vae_cls.__name__


def test_rope_reference_grid_comes_from_the_checkpoints_own_config():
    """PidNet's NTK-aware RoPE scales image-stream frequencies by (current_grid / ref_grid), so
    ref_grid decides every image position in the model.

    The reference takes it in *pixels* and divides by patch_size (pixeldit_official.py:1170),
    and the released res2kto4k checkpoints are trained with PID_SR4X_V1PT5, which sets
    rope_ref_h=2048 -> 128 patches. PidNet's own constructor default is PixDiT's 1024px
    pretraining resolution and does NOT describe these checkpoints; from_pretrained must pass
    2048 explicitly. A previous fix pinned 64 here, validated by a parity run whose CONFIG also
    omitted rope_ref_h -- so both sides fell back to 1024 and agreed on the wrong grid.
    """
    assert PidNet().rope_ref_grid_h == 1024 // 16
    assert PidNet(rope_ref_h=2048, rope_ref_w=2048).rope_ref_grid_h == 128
    # Pixels, not patches: halving patch_size doubles the grid the same reference resolution maps to.
    assert PidNet(rope_ref_h=2048, patch_size=8).rope_ref_grid_h == 256

    source = inspect.getsource(PidDecoder.from_pretrained)
    assert "rope_ref_h=2048" in source and "rope_ref_w=2048" in source


def _tiny_net_for_coverage() -> PidNet:
    return PidNet(
        hidden_size=32,
        pixel_hidden_size=8,
        patch_depth=1,
        pixel_depth=1,
        num_groups=4,
        pixel_attn_hidden_size=None,
        patch_size=4,
        txt_embed_dim=16,
        txt_max_length=8,
        rope_ref_h=64,
        rope_ref_w=64,
        lq_latent_channels=4,
        lq_hidden_dim=8,
        lq_num_res_blocks=1,
        sr_scale=4,
        latent_spatial_down_factor=8,
    )


def test_assert_full_weight_coverage_passes_when_complete():
    # Finding 1: coverage check must derive from the real parameter tree, not a hardcoded
    # count -- so supplying every real path (whatever the actual count is) must pass.
    net = _tiny_net_for_coverage()
    full = dict(tree_flatten(net.parameters()))
    _assert_full_weight_coverage(net, full, label="tiny net")  # must not raise


def test_assert_full_weight_coverage_raises_on_missing_param():
    # Drop one real parameter path from the supplied dict -- must raise, naming the path.
    net = _tiny_net_for_coverage()
    full = dict(tree_flatten(net.parameters()))
    missing_path = next(iter(full))
    partial = {k: v for k, v in full.items() if k != missing_path}

    with pytest.raises(ValueError, match="got no value"):
        _assert_full_weight_coverage(net, partial, label="tiny net")


def _http_error(cls, status_code: int, url: str = "https://huggingface.co/x"):
    httpx = pytest.importorskip("httpx")
    response = httpx.Response(status_code, request=httpx.Request("GET", url))
    return cls("boom", response=response)


def test_load_or_raise_friendly_gated_repo_names_repos_size_and_auth_step():
    def _raise():
        raise _http_error(GatedRepoError, 401, "https://huggingface.co/google/gemma-2-2b-it")

    with pytest.raises(RuntimeError) as exc_info:
        PidDecoder._load_or_raise_friendly("Gemma-2 weights download", "google/gemma-2-2b-it", _raise)

    message = str(exc_info.value)
    assert "nvidia/PiD" in message
    assert "google/gemma-2-2b-it" in message
    assert "8GB" in message
    assert "hf auth login" in message
    assert "license" in message


def test_load_or_raise_friendly_repository_not_found():
    def _raise():
        raise _http_error(RepositoryNotFoundError, 404)

    with pytest.raises(RuntimeError, match="was not found"):
        PidDecoder._load_or_raise_friendly("PidNet checkpoint download", "nvidia/PiD", _raise)


def test_load_or_raise_friendly_passes_through_return_value():
    assert PidDecoder._load_or_raise_friendly("step", "repo", lambda: 42) == 42


def test_release_pid_decoder_drops_the_cached_checkpoint(monkeypatch):
    """The ~8GB PidNet is cached so multi-seed runs don't reload it; release_pid_decoder is the
    only way a long-lived host gets that memory back, since MemorySaver can't see this cache."""
    loads = []
    monkeypatch.setattr(PidDecoder, "from_pretrained", classmethod(lambda cls, variant: loads.append(variant)))

    pid_decoder_module._load_decoder("flux")
    pid_decoder_module._load_decoder("flux")
    assert loads == ["flux"], "second call should have been served from the cache"

    pid_decoder_module.release_pid_decoder()
    assert pid_decoder_module._load_decoder.cache_info().currsize == 0

    pid_decoder_module._load_decoder("flux")
    assert loads == ["flux", "flux"], "release should force a reload"
    pid_decoder_module.release_pid_decoder()
