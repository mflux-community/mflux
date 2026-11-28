import numpy as np
import pytest

pytest.importorskip("mlx.core", exc_type=ImportError)

import mlx.core as mx
from mlx.utils import tree_flatten

from mflux.models.seedvr2.model.seedvr2_vae.common.conv3d import CausalConv3d
from mflux.models.seedvr2.model.seedvr2_vae.decoder.upsample_3d import Upsample3D


def _reference_conv3d(conv: CausalConv3d, x: mx.array) -> mx.array:
    kt, _, _ = conv.kernel_size
    pt, ph, pw = conv.padding
    if conv.causal_temporal and kt > 1:
        causal_pad = (2 * pt) if conv.use_padding_causal else kt - 1
        if causal_pad > 0:
            first_frame = x[:, :, :1, :, :]
            x = mx.concatenate([mx.repeat(first_frame, causal_pad, axis=2), x], axis=2)
        temporal_padding = 0
    else:
        temporal_padding = pt
    x = x.transpose(0, 2, 3, 4, 1)
    out = mx.conv_general(x, conv.weight, stride=conv.stride, padding=(temporal_padding, ph, pw))
    return (out + conv.bias).transpose(0, 4, 1, 2, 3)


def _reference_upsample3d(up: Upsample3D, x: mx.array) -> mx.array:
    B, C, T, H, W = x.shape
    x = _reference_conv3d(up.upscale_conv, x)
    sf, tf = up.spatial_factor, up.temporal_factor
    x = x.reshape(B, sf, sf, tf, C, T, H, W)
    x = x.transpose(0, 4, 5, 3, 6, 1, 7, 2)
    x = x.reshape(B, C, T * tf, H * sf, W * sf)
    if T == 1 and tf > 1:
        x = x[:, :, :1, :, :]
    return _reference_conv3d(up.conv, x)


@pytest.mark.fast
@pytest.mark.parametrize(
    ("kernel_size", "stride", "padding", "use_padding_causal"),
    [
        (3, 1, 1, False),
        (1, 1, 0, False),
        ((1, 3, 3), (1, 2, 2), (0, 0, 0), False),
        ((3, 3, 3), (2, 2, 2), (1, 0, 0), False),
        (3, 1, 1, True),
    ],
)
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_causal_conv3d_t1_matches_3d_reference(kernel_size, stride, padding, use_padding_causal, dtype):
    mx.random.seed(0)
    conv = CausalConv3d(
        in_channels=8,
        out_channels=12,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        causal_temporal=True,
        use_padding_causal=use_padding_causal,
    )
    conv.weight = (mx.random.normal(conv.weight.shape) * 0.1).astype(dtype)
    conv.bias = (mx.random.normal(conv.bias.shape) * 0.1).astype(dtype)
    x = mx.random.normal((1, 8, 1, 9, 9)).astype(dtype)
    mx.eval(conv.weight, conv.bias, x)

    actual = conv(x)
    expected = _reference_conv3d(conv, x)
    mx.eval(actual, expected)

    assert actual.shape == expected.shape
    assert actual.dtype == dtype
    tol = 1e-5 if dtype == mx.float32 else 2e-2
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)), np.array(expected.astype(mx.float32)), rtol=tol, atol=tol
    )


@pytest.mark.fast
def test_causal_conv3d_multiframe_and_non_causal_preserve_3d_path():
    mx.random.seed(1)
    for causal, t_frames in [(True, 3), (False, 1)]:
        conv = CausalConv3d(4, 6, kernel_size=3, stride=1, padding=1, causal_temporal=causal)
        conv.weight = mx.random.normal(conv.weight.shape) * 0.1
        conv.bias = mx.random.normal(conv.bias.shape) * 0.1
        x = mx.random.normal((1, 4, t_frames, 6, 6))
        mx.eval(conv.weight, conv.bias, x)

        actual = conv(x)
        expected = _reference_conv3d(conv, x)
        mx.eval(actual, expected)

        np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=1e-5, atol=1e-5)


@pytest.mark.fast
@pytest.mark.parametrize("temporal_up", [False, True])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_upsample3d_t1_matches_3d_reference_and_keeps_parameters_intact(temporal_up, dtype):
    mx.random.seed(2)
    up = Upsample3D(channels=8, temporal_up=temporal_up)
    for layer in (up.conv, up.upscale_conv):
        layer.weight = (mx.random.normal(layer.weight.shape) * 0.1).astype(dtype)
        layer.bias = (mx.random.normal(layer.bias.shape) * 0.1).astype(dtype)
    x = mx.random.normal((1, 8, 1, 6, 6)).astype(dtype)
    mx.eval(up.parameters(), x)

    param_keys_before = {k: v.shape for k, v in tree_flatten(up.parameters())}
    actual = up(x)
    expected = _reference_upsample3d(up, x)
    mx.eval(actual, expected)
    param_keys_after = {k: v.shape for k, v in tree_flatten(up.parameters())}

    assert param_keys_after == param_keys_before
    assert actual.shape == expected.shape
    assert actual.dtype == dtype
    tol = 1e-5 if dtype == mx.float32 else 2e-2
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)), np.array(expected.astype(mx.float32)), rtol=tol, atol=tol
    )


@pytest.mark.fast
def test_causal_conv3d_t1_preserves_promoted_precision_for_mixed_dtypes():
    mx.random.seed(3)
    conv = CausalConv3d(4, 6, kernel_size=3, stride=1, padding=1, causal_temporal=True)
    conv.weight = (mx.random.normal(conv.weight.shape) * 0.1).astype(mx.bfloat16)
    conv.bias = (mx.random.normal(conv.bias.shape) * 0.1).astype(mx.bfloat16)
    x = mx.random.normal((1, 4, 1, 6, 6)).astype(mx.float32)
    mx.eval(conv.weight, conv.bias, x)

    actual = conv(x)
    expected = _reference_conv3d(conv, x)
    mx.eval(actual, expected)

    assert actual.dtype == mx.float32
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=1e-5, atol=1e-5)
