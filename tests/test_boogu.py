"""Weight-free unit tests for the Boogu-Image-Turbo port.

These validate the numerically tricky pieces (complex RoPE), the transformer
forward shape, and weight-mapping invariants without downloading the ~38 GB
checkpoint. End-to-end image quality is validated separately against real weights.
"""

import mlx.core as mx
import numpy as np

from mflux.models.boogu.model.boogu_transformer.boogu_rope import BooguImageRoPE, apply_rotary_emb
from mflux.models.boogu.model.boogu_transformer.boogu_transformer import BooguImageTransformer
from mflux.models.boogu.weights.boogu_weight_mapping import BooguWeightMapping


def _reference_rope(cap_len, h, w, axes_dim, axes_lens, theta):
    """Reference complex freqs_cis path (diffusers get_1d_rotary_pos_embed + gather)."""

    def axis_complex(dim, length):
        scales = np.arange(0, dim, 2, dtype=np.float64) / dim
        omega = 1.0 / (theta**scales)
        pos = np.arange(length, dtype=np.float64)
        freqs = np.outer(pos, omega)
        return np.cos(freqs) + 1j * np.sin(freqs)

    tables = [axis_complex(d, e) for d, e in zip(axes_dim, axes_lens)]
    seq = cap_len + h * w
    pid = np.zeros((seq, 3), dtype=np.int64)
    tp = np.arange(cap_len)
    pid[:cap_len, 0] = pid[:cap_len, 1] = pid[:cap_len, 2] = tp
    pid[cap_len:, 0] = cap_len
    pid[cap_len:, 1] = np.repeat(np.arange(h), w)
    pid[cap_len:, 2] = np.tile(np.arange(w), h)
    return np.concatenate([tables[a][pid[:, a]] for a in range(3)], axis=-1)


def test_rope_matches_reference_complex_rotation():
    axes_dim, axes_lens, theta = (40, 40, 40), (2048, 1664, 1664), 10000
    rope = BooguImageRoPE(theta=theta, axes_dim=axes_dim, axes_lens=axes_lens)
    cap_len, h, w = 7, 4, 5
    emb = rope(cap_len, h, w)
    seq = cap_len + h * w
    assert emb.joint[0].shape == (seq, sum(axes_dim) // 2)

    freqs_cis = _reference_rope(cap_len, h, w, axes_dim, axes_lens, theta)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, seq, 2, sum(axes_dim))).astype(np.float32)
    xc = x.reshape(1, seq, 2, sum(axes_dim) // 2, 2)
    xc = xc[..., 0] + 1j * xc[..., 1]
    out_c = xc * freqs_cis[None, :, None, :]
    ref = np.stack([out_c.real, out_c.imag], axis=-1).reshape(1, seq, 2, sum(axes_dim))

    mine = np.array(apply_rotary_emb(mx.array(x), emb.joint))
    assert np.abs(mine - ref).max() < 1e-4


def test_transformer_forward_shape():
    model = BooguImageTransformer(
        hidden_size=3360,
        num_layers=4,
        num_double_stream_layers=2,
        num_refiner_layers=1,
        num_attention_heads=28,
        num_kv_heads=7,
        instruction_feat_dim=4096,
    )
    latent = mx.random.normal((1, 16, 16, 24))
    out = model(latent, mx.array([0.5]), mx.random.normal((1, 10, 4096)))
    mx.eval(out)
    assert out.shape == (1, 16, 16, 24)


def test_transformer_mapping_is_identity_and_unique():
    targets = BooguWeightMapping.get_transformer_mapping()
    # Every transformer target is identity (MLX key == checkpoint key).
    assert all(t.to_pattern == t.from_pattern[0] for t in targets)
    # No duplicate target patterns.
    patterns = [t.to_pattern for t in targets]
    assert len(patterns) == len(set(patterns))


def test_text_encoder_mapping_strips_language_model_prefix():
    targets = BooguWeightMapping.get_text_encoder_mapping()
    for t in targets:
        assert t.from_pattern[0].startswith("model.language_model.")
        assert not t.to_pattern.startswith("model.")


def test_block_targets_pin_max_blocks():
    # The loader defaults {block} expansion to 4, so every {block} target must
    # set max_blocks or higher-index layers silently keep random init.
    for targets in (
        BooguWeightMapping.get_transformer_mapping(),
        BooguWeightMapping.get_text_encoder_mapping(),
    ):
        for t in targets:
            if "{block}" in t.to_pattern:
                assert t.max_blocks is not None, t.to_pattern

    # The biggest stacks must expand to their full depth.
    transformer = {t.to_pattern: t.max_blocks for t in BooguWeightMapping.get_transformer_mapping()}
    assert transformer["single_stream_layers.{block}.norm2.weight"] == 32
    assert transformer["double_stream_layers.{block}.img_attn_norm.weight"] == 8
