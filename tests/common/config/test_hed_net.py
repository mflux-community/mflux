import mlx.core as mx
import numpy as np
import pytest

from mflux.models.hed.hed import _BLOCKS, HED, _HEDNet


def _fake_state():
    # torch-layout (OIHW) tensors with the real channel dims but tiny random values, so the net wiring
    # can be exercised without downloading the 28MB checkpoint.
    dims = {"block1": (3, 64), "block2": (64, 128), "block3": (128, 256), "block4": (256, 512), "block5": (512, 512)}
    state = {"norm": mx.zeros((1, 3, 1, 1))}
    for name, n_conv in _BLOCKS:
        cin, cout = dims[name]
        for c in range(n_conv):
            i = cin if c == 0 else cout
            state[f"{name}.convs.{c}.weight"] = mx.zeros((cout, i, 3, 3))
            state[f"{name}.convs.{c}.bias"] = mx.zeros((cout,))
        state[f"{name}.projection.weight"] = mx.zeros((1, cout, 1, 1))
        state[f"{name}.projection.bias"] = mx.zeros((1,))
    return state


@pytest.mark.fast
def test_hed_net_emits_five_downsampled_single_channel_side_outputs():
    net = _HEDNet(_fake_state())
    projections = net(mx.zeros((1, 64, 64, 3)))
    mx.eval(projections)
    # one side output per block, each a single channel, halving in resolution after the first block
    assert len(projections) == 5
    expected = [64, 32, 16, 8, 4]
    for p, size in zip(projections, expected):
        assert p.shape == (1, size, size, 1)


@pytest.mark.fast
def test_fuse_side_outputs_gives_an_rgb_edge_at_the_requested_size():
    # Side outputs at their own resolutions fuse to a single 8-bit RGB edge at the requested size,
    # with sigmoid applied (values spread across 0..255, not a flat map).
    projections = [
        mx.array(np.linspace(-6, 6, 1 * h * w * 1, dtype=np.float32).reshape(1, h, w, 1))
        for h, w in [(8, 8), (4, 4), (2, 2)]
    ]
    edge = HED._fuse_side_outputs(projections, (32, 24))
    assert edge.size == (32, 24)
    assert edge.mode == "RGB"
    arr = np.array(edge)
    # sigmoid spread, not a flat image
    assert arr.min() < 64
    assert arr.max() > 192
