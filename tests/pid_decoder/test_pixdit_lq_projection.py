import mlx.core as mx

from mflux.models.common.pid_decoder.pixdit.pixdit_lq_projection import LQProjection2D


def test_lq_projection_output_shapes():
    proj = LQProjection2D(
        latent_channels=16,
        hidden_dim=32,
        out_dim=64,
        patch_size=16,
        sr_scale=4,
        latent_spatial_down_factor=8,
        num_res_blocks=1,
        num_outputs=3,
        gate_type="sigma_aware_per_token_per_dim",
        interval=1,
    )
    lq_latent = mx.random.normal((1, 16, 8, 8))  # zH=zW=8
    outputs = proj(lq_latent, target_pH=16, target_pW=16)  # z_to_patch_ratio=2 -> upsample 2x: 8->16
    assert len(outputs) == 3
    for out in outputs:
        assert out.shape == (1, 256, 64)


def test_lq_projection_gate_and_real_checkpoint_gate_type():
    # Real qwenimage checkpoint uses "sigma_aware_per_token" (per-token scalar gate),
    # not the per-token-per-dim variant. Exercise it plus the .gate() entrypoint Task 7 uses.
    proj = LQProjection2D(
        latent_channels=16,
        hidden_dim=32,
        out_dim=64,
        patch_size=16,
        sr_scale=4,
        latent_spatial_down_factor=8,
        num_res_blocks=1,
        num_outputs=2,
        gate_type="sigma_aware_per_token",
        interval=2,
    )
    lq_latent = mx.random.normal((1, 16, 8, 8))
    outputs = proj(lq_latent, target_pH=16, target_pW=16)

    x = mx.random.normal((1, 256, 64))
    sigma = mx.array([0.5])
    gated = proj.gate(x, outputs[0], sigma, out_idx=0)
    assert gated.shape == (1, 256, 64)

    assert proj.is_gate_active(0) is True
    assert proj.is_gate_active(1) is False
    assert proj.output_index(2) == 1


def test_lq_projection_fold_branch_channel_order():
    # z_to_patch_ratio < 1 exercises the fold branch (not hit by the real qwenimage
    # checkpoint, but ported for completeness). Verify the channel-packing order matches
    # PyTorch's (C, fH, fW)-slowest-to-fastest layout by checking a hand-computed fold.
    latent_channels, f, pH, pW = 2, 2, 3, 3
    proj = LQProjection2D(
        latent_channels=latent_channels,
        hidden_dim=8,
        out_dim=8,
        patch_size=8,
        sr_scale=2,
        latent_spatial_down_factor=2,
        num_res_blocks=0,
        num_outputs=1,
        gate_type="sigma_aware_per_token",
        interval=1,
    )
    assert proj.z_to_patch_ratio < 1
    assert proj.latent_fold_factor == f

    zH, zW = pH * f, pW * f
    lq_nchw = mx.arange(latent_channels * zH * zW).reshape(1, latent_channels, zH, zW).astype(mx.float32)
    lq_nhwc = lq_nchw.transpose(0, 2, 3, 1)

    aligned = proj._align_latent_to_patch_grid(lq_nhwc, pH, pW)
    assert aligned.shape == (1, pH, pW, latent_channels * f * f)

    # Expected value at output token (i, j), packed-channel index (c, fh, fw) [C slowest]
    # is the source pixel at (i*f + fh, j*f + fw), channel c.
    i, j, c, fh, fw = 1, 2, 1, 0, 1
    packed_idx = (c * f + fh) * f + fw
    expected = lq_nchw[0, c, i * f + fh, j * f + fw]
    assert aligned[0, i, j, packed_idx].item() == expected.item()


def test_lq_projection_upsample_branch_nearest_neighbor_mapping():
    # z_to_patch_ratio > 1 is the branch that actually executes for the real qwenimage
    # checkpoint (sr_scale=4, latent_spatial_down_factor=8, patch_size=16 -> ratio=2). The
    # shape-only test above doesn't prove the mx.repeat upsample lands on the right source
    # pixel; verify a couple of concrete (h, w) coordinates by hand.
    latent_channels, ratio, pH, pW = 3, 2, 8, 8
    proj = LQProjection2D(
        latent_channels=latent_channels,
        hidden_dim=4,
        out_dim=4,
        patch_size=8,
        sr_scale=4,
        latent_spatial_down_factor=4,
        num_res_blocks=0,
        num_outputs=1,
        gate_type="sigma_aware_per_token",
        interval=1,
    )
    assert proj.z_to_patch_ratio > 1
    assert proj.latent_upsample_ratio == ratio

    zH = zW = pH // ratio
    lq_nchw = mx.arange(latent_channels * zH * zW).reshape(1, latent_channels, zH, zW).astype(mx.float32)
    lq_nhwc = lq_nchw.transpose(0, 2, 3, 1)

    aligned = proj._align_latent_to_patch_grid(lq_nhwc, pH, pW)
    assert aligned.shape == (1, pH, pW, latent_channels)

    # mx.repeat is nearest-neighbor (np.repeat semantics): output index h sources from
    # input index h // ratio (e.g. repeat([a, b], 2) == [a, a, b, b]).
    for h, w, c in [(3, 5, 2), (7, 7, 1), (0, 0, 0)]:
        expected = lq_nchw[0, c, h // ratio, w // ratio]
        assert aligned[0, h, w, c].item() == expected.item()


def test_lq_projection_pit_output_appends_extra_head():
    # pit_lq_inject (real qwenimage checkpoint: PID_SR4X_V1PT5) adds a second LQ injection
    # point into the pixel-level (PiT) pathway. LQProjection2D exposes this as an extra
    # trailing output from the same shared `tokens`, gated by pit_output=True.
    proj = LQProjection2D(
        latent_channels=16,
        hidden_dim=32,
        out_dim=64,
        patch_size=16,
        sr_scale=4,
        latent_spatial_down_factor=8,
        num_res_blocks=1,
        num_outputs=3,
        gate_type="sigma_aware_per_token_per_dim",
        interval=1,
        pit_output=True,
    )
    assert proj.pit_head is not None
    lq_latent = mx.random.normal((1, 16, 8, 8))
    outputs = proj(lq_latent, target_pH=16, target_pW=16)
    assert len(outputs) == 3 + 1
    for out in outputs:
        assert out.shape == (1, 256, 64)


def test_lq_projection_pit_output_disabled_by_default():
    proj = LQProjection2D(
        latent_channels=16,
        hidden_dim=32,
        out_dim=64,
        patch_size=16,
        sr_scale=4,
        latent_spatial_down_factor=8,
        num_res_blocks=1,
        num_outputs=3,
        gate_type="sigma_aware_per_token_per_dim",
        interval=1,
    )
    assert proj.pit_head is None
    lq_latent = mx.random.normal((1, 16, 8, 8))
    outputs = proj(lq_latent, target_pH=16, target_pW=16)
    assert len(outputs) == 3


def test_lq_projection_replicate_padding_differs_from_zeros():
    # conv_padding_mode="replicate" is required by the real qwenimage checkpoint
    # (PID_SR4X_V1PT5["lq_conv_padding_mode"]) but had no test coverage. Build the same
    # module twice with identical weights/input under each padding mode and confirm the
    # outputs actually diverge (proving replicate padding changes behavior, not just that
    # it runs).
    kwargs = dict(
        latent_channels=16,
        hidden_dim=32,
        out_dim=64,
        patch_size=16,
        sr_scale=4,
        latent_spatial_down_factor=8,
        num_res_blocks=1,
        num_outputs=1,
        gate_type="sigma_aware_per_token",
        interval=1,
    )
    proj_zeros = LQProjection2D(conv_padding_mode="zeros", **kwargs)
    proj_replicate = LQProjection2D(conv_padding_mode="replicate", **kwargs)
    proj_replicate.update(proj_zeros.parameters())

    lq_latent = mx.random.normal((1, 16, 8, 8))
    out_zeros = proj_zeros(lq_latent, target_pH=16, target_pW=16)[0]
    out_replicate = proj_replicate(lq_latent, target_pH=16, target_pW=16)[0]

    assert out_zeros.shape == out_replicate.shape == (1, 256, 64)
    assert not bool(mx.allclose(out_zeros, out_replicate))


def test_every_mflux_legal_size_lands_exactly_on_the_patch_grid():
    """mflux rounds all generations to a multiple of 16 (config.py:52-53), the VAE compresses
    by 8 and PiD super-resolves 4x at patch_size 16 -- so the LQ latent must always be exactly
    half the patch grid. Pin that chain: it is what lets the nearest upsample stay exact."""
    for base in range(16, 4096 + 1, 16):
        zH = base // 8
        patch_grid = (base * 4) // 16
        assert zH * 2 == patch_grid, base


def test_align_matches_torch_nearest_when_the_grid_is_not_a_multiple_of_the_latent():
    """Off-grid sizes must follow F.interpolate(mode="nearest")'s index map, out[i] = in[i*z//p],
    not repeat-by-integer-ratio-then-crop, which keeps the leading tiles and drops the tail."""
    proj = LQProjection2D(
        latent_channels=1, hidden_dim=8, out_dim=16, patch_size=16, sr_scale=4,
        latent_spatial_down_factor=8, num_res_blocks=1, num_outputs=1,
        gate_type="sigma_aware_per_token", interval=1,
    )  # fmt: skip
    latent = mx.arange(5, dtype=mx.float32).reshape(1, 1, 5, 1)  # rows 0..4, one column
    aligned = proj._align_latent_to_patch_grid(latent.transpose(0, 2, 3, 1), pH=9, pW=1)

    assert aligned.shape == (1, 9, 1, 1)
    expected = [i * 5 // 9 for i in range(9)]  # 0,0,1,1,2,2,3,3,4 -- reaches the last row
    assert [int(v) for v in aligned.reshape(-1)] == expected
