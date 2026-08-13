import mlx.core as mx

from mflux.models.common.pid_decoder.pid_sampler import sample
from mflux.models.common.pid_decoder.pixdit.pixdit_network import PidNet


def _make_net() -> PidNet:
    return PidNet(
        hidden_size=32,
        pixel_hidden_size=8,
        patch_depth=2,
        pixel_depth=1,
        num_groups=4,
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


def test_sample_output_shape():
    net = _make_net()
    caption_embs = mx.random.normal((1, 6, 16))
    lq_latent = mx.random.normal((1, 4, 4, 4))
    sigma = mx.array([0.0])
    out = sample(net, caption_embs, lq_latent, sigma, target_h=16, target_w=16, seed=0)
    assert out.shape == (1, 3, 16, 16)
    assert float(mx.max(mx.abs(out))) <= 1.0 + 1e-5


def test_sample_deterministic_for_fixed_seed():
    net = _make_net()
    caption_embs = mx.random.normal((1, 6, 16))
    lq_latent = mx.random.normal((1, 4, 4, 4))
    sigma = mx.array([0.0])
    out_a = sample(net, caption_embs, lq_latent, sigma, target_h=16, target_w=16, seed=42)
    out_b = sample(net, caption_embs, lq_latent, sigma, target_h=16, target_w=16, seed=42)
    assert mx.array_equal(out_a, out_b)


def test_sample_different_seeds_produce_different_output():
    net = _make_net()
    caption_embs = mx.random.normal((1, 6, 16))
    lq_latent = mx.random.normal((1, 4, 4, 4))
    sigma = mx.array([0.0])
    out_seed1 = sample(net, caption_embs, lq_latent, sigma, target_h=16, target_w=16, seed=1)
    out_seed2 = sample(net, caption_embs, lq_latent, sigma, target_h=16, target_w=16, seed=2)
    assert not mx.array_equal(out_seed1, out_seed2)


def test_sample_does_not_disturb_the_global_rng_stream():
    # The sampler threads explicit keys, so a decode must leave the process-global
    # stream where it found it: anything drawing unkeyed afterwards has to see the
    # same values whether or not PiD ran.
    def stub_net(x, t, caption_embs, lq_latent, sigma):
        return mx.zeros(x.shape)

    mx.random.seed(123)
    expected = mx.random.normal((8,))

    mx.random.seed(123)
    sample(
        stub_net,
        mx.zeros((1, 6, 16)),
        mx.zeros((1, 4, 4, 4)),
        mx.array(0.0),
        target_h=16,
        target_w=16,
        seed=999,
    )
    after = mx.random.normal((8,))

    assert mx.array_equal(expected, after)
