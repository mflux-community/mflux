import inspect

import mlx.core as mx
import pytest

from mflux.models.common.latent_creator.latent_creator import LatentCreator
from mflux.models.common.pid_decoder.pid_decoder import PID_MAX_DEGRADE_SIGMA, PidDecoder, pid_decode_latents
from mflux.models.common.pid_decoder.pixdit.pixdit_network import PidNet


class _FakeCaptionEncoder:
    def __call__(self, caption: str) -> mx.array:
        return mx.zeros((1, 6, 16))


def _tiny_decoder() -> PidDecoder:
    net = PidNet(
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
    return PidDecoder(pid_net=net, caption_encoder=_FakeCaptionEncoder())


def test_pid_decode_latents_rejects_degrade_sigma_out_of_range():
    with pytest.raises(ValueError, match="pid_degrade_sigma"):
        pid_decode_latents(vae=object(), latent=mx.zeros((1, 4, 4, 4)), caption="x", seed=0, degrade_sigma=0.9)
    with pytest.raises(ValueError, match="pid_degrade_sigma"):
        pid_decode_latents(vae=object(), latent=mx.zeros((1, 4, 4, 4)), caption="x", seed=0, degrade_sigma=-0.1)


def test_decode_rejects_degrade_sigma_out_of_range():
    # decode() is public, so the range check cannot live only in pid_decode_latents: a direct
    # call would otherwise noise the latent past the distribution the LQ gate was distilled on
    # and report that same out-of-range sigma to the gate.
    decoder = _tiny_decoder()
    latent = mx.zeros((1, 4, 4, 4))
    with pytest.raises(ValueError, match="pid_degrade_sigma"):
        decoder.decode(latent, caption="x", seed=0, degrade_sigma=0.9)
    with pytest.raises(ValueError, match="pid_degrade_sigma"):
        decoder.decode(latent, caption="x", seed=0, degrade_sigma=-0.1)


@pytest.mark.parametrize("bad_sigma", [None, "0.2", True])
def test_degrade_sigma_rejects_non_numeric_values(bad_sigma):
    # A sidecar from a non-PiD run stores `"pid_degrade_sigma": null`, so None can reach this
    # boundary from metadata rather than from a caller. Report it as a ValueError naming the
    # option, not as a bare TypeError out of the range comparison.
    with pytest.raises(ValueError, match="pid_degrade_sigma"):
        pid_decode_latents(vae=object(), latent=mx.zeros((1, 4, 4, 4)), caption="x", seed=0, degrade_sigma=bad_sigma)
    with pytest.raises(ValueError, match="pid_degrade_sigma"):
        _tiny_decoder().decode(mx.zeros((1, 4, 4, 4)), caption="x", seed=0, degrade_sigma=bad_sigma)


def test_max_degrade_sigma_matches_pid_training_distribution():
    # PiD's LQ gate was distilled on latents noised at sigma ~ U[0.0, 0.8] -- this bound is
    # load-bearing, not decorative, so pin it against silent drift.
    assert PID_MAX_DEGRADE_SIGMA == 0.8


def test_decode_is_deterministic_for_fixed_seed_and_degrade_sigma():
    decoder = _tiny_decoder()
    latent = mx.random.normal((1, 4, 4, 4))
    out_a = decoder.decode(latent, caption="a photo", seed=7, degrade_sigma=0.2)
    out_b = decoder.decode(latent, caption="a photo", seed=7, degrade_sigma=0.2)
    assert mx.array_equal(out_a, out_b)


def test_decode_degrade_sigma_changes_output_relative_to_clean_latent():
    decoder = _tiny_decoder()
    latent = mx.random.normal((1, 4, 4, 4))
    out_clean = decoder.decode(latent, caption="a photo", seed=7, degrade_sigma=0.0)
    out_degraded = decoder.decode(latent, caption="a photo", seed=7, degrade_sigma=0.4)
    assert not mx.array_equal(out_clean, out_degraded)


def test_decode_noising_follows_flow_matching_interpolation_convention(monkeypatch):
    # Capture the exact tensor decode() hands to the sampler, so we can invert
    # (1-s)*x0 + s*eps and confirm it matches an independently-drawn eps at the same key --
    # if this convention drifts, PiD would be handed noise from the wrong distribution.
    captured = {}

    def _fake_sample(net, caption_embs, lq_latent, sigma, *, target_h, target_w, seed):
        captured["lq_latent"] = lq_latent
        captured["sigma"] = sigma
        return mx.zeros((lq_latent.shape[0], 3, target_h, target_w))

    monkeypatch.setattr("mflux.models.common.pid_decoder.pid_decoder.sample", _fake_sample)

    decoder = _tiny_decoder()
    clean_latent = mx.random.normal((1, 4, 4, 4))
    seed = 11
    degrade_sigma = 0.3
    decoder.decode(clean_latent, caption="a photo", seed=seed, degrade_sigma=degrade_sigma)

    expected_eps = mx.random.normal(clean_latent.shape, key=mx.random.key(seed ^ 0x91D))
    expected_noised = LatentCreator.add_noise_by_interpolation(
        clean=clean_latent, noise=expected_eps, sigma=degrade_sigma
    )

    assert mx.allclose(
        captured["lq_latent"].astype(mx.float32), expected_noised.astype(mx.bfloat16).astype(mx.float32), atol=1e-2
    )
    assert float(captured["sigma"][0]) == pytest.approx(degrade_sigma)


def test_decode_skips_noising_when_degrade_sigma_is_zero(monkeypatch):
    captured = {}

    def _fake_sample(net, caption_embs, lq_latent, sigma, *, target_h, target_w, seed):
        captured["lq_latent"] = lq_latent
        return mx.zeros((lq_latent.shape[0], 3, target_h, target_w))

    monkeypatch.setattr("mflux.models.common.pid_decoder.pid_decoder.sample", _fake_sample)

    decoder = _tiny_decoder()
    clean_latent = mx.random.normal((1, 4, 4, 4))
    decoder.decode(clean_latent, caption="a photo", seed=11, degrade_sigma=0.0)

    assert mx.array_equal(captured["lq_latent"].astype(mx.float32), clean_latent.astype(mx.bfloat16).astype(mx.float32))


def test_degrade_sigma_noise_uses_an_independent_keyed_rng_stream():
    # The noise draw must come from an explicitly keyed generator (mx.random.key(seed ^ ...)),
    # never the unkeyed global stream sample() itself reseeds via mx.random.seed(seed) --
    # otherwise raising degrade_sigma would also reshuffle the sampler's own pixel noise,
    # making the knob impossible to evaluate (mirrors Krea2Sampler's seed ^ 0x5DE).
    source = inspect.getsource(PidDecoder.decode)
    assert "mx.random.key(seed ^ 0x91D)" in source
    assert "mx.random.seed(" not in source
