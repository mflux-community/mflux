import mlx.core as mx

from mflux.models.common.pid_decoder.pixdit.pixdit_network import PidNet


def _tiny_net(**overrides) -> PidNet:
    kwargs = dict(
        hidden_size=32,
        pixel_hidden_size=8,
        patch_depth=2,
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
    kwargs.update(overrides)
    return PidNet(**kwargs)


def test_pidnet_output_shape():
    net = _tiny_net()
    x = mx.random.normal((1, 3, 16, 16))  # H=W=16, patch_size=4 -> 4x4 grid
    t = mx.array([0.5])
    y = mx.random.normal((1, 6, 16))
    lq_latent = mx.random.normal((1, 4, 4, 4))  # zH=zW=4; z_to_patch_ratio = 4*8/4=8 -> upsample
    sigma = mx.array([0.0])
    out = net(x, t, y, lq_latent, sigma)
    assert out.shape == x.shape


def test_pidnet_output_shape_with_text_rope_disabled():
    # use_text_rope is a real constructor switch (defaults True to match the
    # qwenimage checkpoint) -- exercise the pos_txt=None path too.
    net = _tiny_net(use_text_rope=False)
    x = mx.random.normal((1, 3, 16, 16))
    t = mx.array([0.5])
    y = mx.random.normal((1, 6, 16))
    lq_latent = mx.random.normal((1, 4, 4, 4))
    sigma = mx.array([0.0])
    out = net(x, t, y, lq_latent, sigma)
    assert out.shape == x.shape


def test_pidnet_pit_lq_inject_output_shape():
    # Real qwenimage checkpoint (PID_SR4X_V1PT5) sets pit_lq_inject=True: a second LQ
    # injection point into the pixel-level (PiT) pathway, gating `s` right before it's
    # flattened into s_cond. Exercise the wiring end-to-end.
    net = _tiny_net(pit_lq_inject=True)
    assert net.pit_lq_gate is not None
    x = mx.random.normal((1, 3, 16, 16))
    t = mx.array([0.5])
    y = mx.random.normal((1, 6, 16))
    lq_latent = mx.random.normal((1, 4, 4, 4))
    sigma = mx.array([0.0])
    out = net(x, t, y, lq_latent, sigma)
    assert out.shape == x.shape


def test_pidnet_pit_lq_inject_changes_output():
    # Value-level check that the pit injection actually alters the forward pass, not just
    # that it runs. lq_proj is built with pit_output=True regardless (fixed at construction),
    # so toggling net.pit_lq_inject after construction isolates exactly the gate-application
    # branch in __call__ while holding every weight identical.
    net = _tiny_net(pit_lq_inject=True)
    x = mx.random.normal((1, 3, 16, 16))
    t = mx.array([0.5])
    y = mx.random.normal((1, 6, 16))
    lq_latent = mx.random.normal((1, 4, 4, 4))
    sigma = mx.array([0.0])

    out_with_pit = net(x, t, y, lq_latent, sigma)
    net.pit_lq_inject = False
    out_without_pit = net(x, t, y, lq_latent, sigma)

    assert out_with_pit.shape == out_without_pit.shape
    assert not bool(mx.allclose(out_with_pit, out_without_pit))


def test_pidnet_real_qwenimage_config_small_spatial():
    # Finding 2 (final integration review): every other test in this file uses shrunk
    # illustrative dims (hidden_size=32, 16x16 grids). The NTK-aware 2D RoPE
    # (precompute_freqs_cis_2d_ntk) is scale-dependent on height/rope_ref_grid_h, so it
    # behaves qualitatively differently at the real checkpoint's rope_ref_h=rope_ref_w=2048px
    # (ref_grid=128 patches at patch_size=16) than at a tiny 16x16 test grid -- and that math has
    # never been numerically exercised. Use the real confirmed qwenimage config (see
    # docs/superpowers/notes/pid-qwenimage-config.md and progress.md's "REAL QWENIMAGE
    # CONFIG" note) for every dim, but at a small 64x64 pixel input (not the real 2048px)
    # so the test stays fast.
    net = PidNet(
        patch_depth=14,
        pixel_depth=2,
        hidden_size=1536,
        pixel_hidden_size=16,
        pixel_attn_hidden_size=1152,
        num_groups=24,
        pixel_num_groups=16,
        patch_size=16,
        lq_latent_channels=16,
        lq_gate_type="sigma_aware_per_token",
        lq_interval=2,
        lq_hidden_dim=1024,
        lq_conv_padding_mode="replicate",
        pit_lq_inject=True,
        sr_scale=4,
        latent_spatial_down_factor=8,
        txt_embed_dim=2304,
        txt_max_length=300,
        use_text_rope=True,
        lq_num_res_blocks=4,
        rope_ref_h=2048,  # PID_SR4X_V1PT5 -> ref_grid 128 at patch_size 16
        rope_ref_w=2048,
    )
    # 64x64 @ patch_size=16 -> Hs=Ws=4 patch grid; real z_to_patch_ratio=2 (sr_scale=4 *
    # latent_spatial_down_factor=8 / patch_size=16) means lq_latent's spatial dims are
    # half the patch grid's: zH=zW=2.
    x = mx.random.normal((1, 3, 64, 64))
    t = mx.array([0.5])
    y = mx.random.normal((1, 6, 2304))  # arbitrary short caption length, well under txt_max_length=300
    lq_latent = mx.random.normal((1, 16, 2, 2))
    sigma = mx.array([0.0])

    out = net(x, t, y, lq_latent, sigma)

    assert out.shape == x.shape
    assert bool(mx.all(mx.isfinite(out)))


def test_patchify_unpatchify_round_trip():
    # Sanity check independent of the network weights: PixelTokenEmbedder's
    # NHWC fold-in (pixdit_embedders.py, minus the +pos_embed/proj, which are
    # additive/linear and don't affect the geometric fold) and PidNet's own
    # fold-out at the bottom of __call__ must be exact inverses of each other,
    # since both are pure reshape/transpose with no learned parameters.
    B, C, H, W, patch_size = 2, 3, 16, 16, 4
    Hs, Ws = H // patch_size, W // patch_size
    P2 = patch_size * patch_size
    x = mx.random.normal((B, C, H, W))

    # Fold-in, mirroring PixelTokenEmbedder.__call__.
    folded = x.transpose(0, 2, 3, 1)  # [B, H, W, C]
    folded = folded.reshape(B, Hs, patch_size, Ws, patch_size, C)
    folded = folded.transpose(0, 1, 3, 2, 4, 5)  # [B, Hs, Ws, ps, ps, C]
    folded = folded.reshape(B * Hs * Ws, P2, C)

    # Fold-out, mirroring the tail of PidNet.__call__.
    x_back = folded.reshape(B, Hs, Ws, patch_size, patch_size, C)
    x_back = x_back.transpose(0, 1, 3, 2, 4, 5)  # [B, Hs, ps, Ws, ps, C]
    x_back = x_back.reshape(B, H, W, C).transpose(0, 3, 1, 2)  # [B, C, H, W]

    assert x_back.shape == x.shape
    assert mx.allclose(x_back, x).item()
