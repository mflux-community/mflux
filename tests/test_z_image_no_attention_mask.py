"""The Z-Image main DiT path must not put an all-True mask into SDPA.

transformer.py used to build mx.ones(...) masks for all three block streams. attention.py:83-88
turns any mask into an additive one, so an all-True mask became an all-zero one: mathematically a
no-op, but SDPA still reads the broadcast. Isolated attention measured that read at 2.0% to 4.3%
across three sequence lengths and two MLX versions. The block-level effect did not reproduce: one
run said +1.21% and another -1.63%. So no step-level saving is claimed here; see
spikes/zimage_dit/README.md for both runs.
These tests pin what is exact rather than what is small: the output is bitwise identical, and no
mask reaches the attention kernel.

Scoped to ZImageTransformer. ZImageControlNetTransformer (transformer_controlnet.py:342, 400, 418)
still builds the same all-True masks: its blocks take the mask positionally, so dropping it there is
a wider change than this saving justifies.
"""

import mlx.core as mx
import pytest
from mlx.core.fast import scaled_dot_product_attention

from mflux.models.z_image.model.z_image_transformer import attention as attention_module
from mflux.models.z_image.model.z_image_transformer.attention import ZImageAttention
from mflux.models.z_image.model.z_image_transformer.transformer import ZImageTransformer
from mflux.models.z_image.model.z_image_transformer.transformer_block import ZImageTransformerBlock

DIM = 64
HEADS = 2
HEAD_DIM = DIM // HEADS  # 32
ROPE_HALF = HEAD_DIM // 2  # the rotary table holds one cos/sin pair per two head dims
SEQ = 37
T_EMB = min(DIM, 256)
# The shapes the real model runs, so the equality claim is tested in the regime that picks the kernel.
PROD_HEADS, PROD_HEAD_DIM, PROD_SEQ = 30, 128, 4224


def _inputs():
    mx.random.seed(0)
    x = mx.random.normal((1, SEQ, DIM)).astype(mx.bfloat16)
    angles = mx.random.uniform(-3.14159, 3.14159, shape=(SEQ, ROPE_HALF))
    freqs_cis = mx.stack([mx.cos(angles), mx.sin(angles)], axis=-1)
    all_true = mx.ones((1, SEQ), dtype=mx.bool_)
    t_emb = mx.random.normal((1, T_EMB)).astype(mx.bfloat16)
    return x, all_true, freqs_cis, t_emb


def _module(cls):
    mx.random.seed(7)
    module = (
        cls(dim=DIM, n_heads=HEADS, norm_eps=1e-5, qk_norm=True)
        if cls is ZImageTransformerBlock
        else cls(dim=DIM, n_heads=HEADS)
    )
    module.set_dtype(mx.bfloat16)
    mx.eval(module.parameters())
    return module


@pytest.mark.fast
def test_all_true_mask_is_bitwise_identical_to_no_mask_in_attention():
    x, all_true, freqs_cis, _ = _inputs()
    attn = _module(ZImageAttention)
    with_mask = attn(x, attention_mask=all_true, freqs_cis=freqs_cis)
    without = attn(x, attention_mask=None, freqs_cis=freqs_cis)
    mx.eval(with_mask, without)
    assert with_mask.dtype == mx.bfloat16
    assert mx.array_equal(with_mask, without)


@pytest.mark.fast
def test_all_true_mask_is_bitwise_identical_to_no_mask_in_a_block():
    x, all_true, freqs_cis, t_emb = _inputs()
    block = _module(ZImageTransformerBlock)
    with_mask = block(x=x, attn_mask=all_true, freqs_cis=freqs_cis, t_emb=t_emb)
    without = block(x=x, attn_mask=None, freqs_cis=freqs_cis, t_emb=t_emb)
    mx.eval(with_mask, without)
    assert mx.array_equal(with_mask, without)


@pytest.mark.fast
def test_transformer_passes_no_mask_to_sdpa(monkeypatch):
    mx.random.seed(0)
    transformer = ZImageTransformer(
        dim=DIM,
        n_layers=2,
        n_refiner_layers=1,
        n_heads=HEADS,
        cap_feat_dim=32,
        axes_dims=[8, 12, 12],
        axes_lens=[64, 32, 32],
    )
    transformer.set_dtype(mx.bfloat16)

    seen = []
    real = attention_module.scaled_dot_product_attention

    def record(q, k, v, *, scale, mask=None, **kwargs):
        seen.append(mask)
        return real(q, k, v, scale=scale, mask=mask, **kwargs)

    monkeypatch.setattr(attention_module, "scaled_dot_product_attention", record)
    latents = mx.random.normal((16, 1, 8, 8)).astype(mx.bfloat16)
    out = transformer(
        x=latents,
        timestep=mx.array([0.5]),
        sigmas=mx.array([0.5]),
        cap_feats=mx.random.normal((7, 32)).astype(mx.bfloat16),
    )
    mx.eval(out)

    # 1 noise refiner + 1 context refiner + 2 main layers.
    assert len(seen) == 4
    assert seen == [None] * len(seen), "an all-True mask reached SDPA"
    assert out.dtype == mx.bfloat16


@pytest.mark.fast
def test_all_zero_mask_is_bitwise_identical_at_production_sdpa_shape():
    # The equality the release note promises is about the kernel MLX picks for the real shapes
    # (head_dim 128, 30 heads, S=4224, bf16), not for the toy config above. Kernel selection keys on
    # exactly these parameters, so assert it where it is claimed.
    mx.random.seed(4)
    shape = (1, PROD_HEADS, PROD_SEQ, PROD_HEAD_DIM)
    q = mx.random.normal(shape=shape).astype(mx.bfloat16)
    k = mx.random.normal(shape=shape).astype(mx.bfloat16)
    v = mx.random.normal(shape=shape).astype(mx.bfloat16)
    zeros = mx.zeros((1, 1, 1, PROD_SEQ), dtype=mx.bfloat16)
    scale = PROD_HEAD_DIM**-0.5
    mx.eval(q, k, v, zeros)

    masked = scaled_dot_product_attention(q, k, v, scale=scale, mask=zeros)
    unmasked = scaled_dot_product_attention(q, k, v, scale=scale, mask=None)
    mx.eval(masked, unmasked)
    assert mx.array_equal(masked, unmasked)


@pytest.mark.fast
def test_attention_spreads_a_mask_over_the_key_axis_not_the_query_axis(monkeypatch):
    # attention.py:85 indexes the boolean mask [:, None, None, :], so the additive mask it builds
    # selects keys and shares that choice across every query row. Every mask the model ever built
    # was all-True, which is invariant under any permutation of those axes, so an all-True test
    # cannot tell the two apart. A partial mask can, and this PR removes the last production
    # producer of masks while keeping the conversion, so the conversion keeps needing the proof.
    #
    # The mask must be captured from the call itself. Building an additive mask here and handing it
    # straight to SDPA would test SDPA and leave line 85 unexercised.
    n_valid = 24
    x, _all_true, freqs_cis, _t_emb = _inputs()
    partial = mx.array([[True] * n_valid + [False] * (SEQ - n_valid)])
    attn = _module(ZImageAttention)

    captured = []
    real = attention_module.scaled_dot_product_attention

    def spy(q, k, v, *, scale, mask=None, **kwargs):
        captured.append(mask)
        return real(q, k, v, scale=scale, mask=mask, **kwargs)

    monkeypatch.setattr(attention_module, "scaled_dot_product_attention", spy)
    masked = attn(x, attention_mask=partial, freqs_cis=freqs_cis)
    unmasked = attn(x, attention_mask=None, freqs_cis=freqs_cis)
    mx.eval(masked, unmasked)

    assert len(captured) == 2
    additive = captured[0]
    assert additive is not None, "the partial mask never reached SDPA"
    # Key axis last and broadcast across queries. A permutation onto the query axis gives
    # (1, 1, SEQ, 1) instead, which fails here rather than silently passing.
    assert additive.shape == (1, 1, 1, SEQ), f"mask spread over the wrong axis: {additive.shape}"

    head = additive[0, 0, 0, :]
    assert mx.all(head[:n_valid] == 0), "a valid key was masked out"
    assert mx.all(head[n_valid:] == float("-inf")), "an invalid key was left unmasked"
    assert not mx.array_equal(masked, unmasked), "the mask changed nothing, so it is not applied"


@pytest.mark.fast
def test_gradient_checkpointing_takes_the_absent_mask():
    # transformer.py:145 wraps each block in nn.utils.checkpoint when the training adapter turns it
    # on, and every block call now passes attn_mask=None through it.
    mx.random.seed(0)
    transformer = ZImageTransformer(
        dim=DIM,
        n_layers=2,
        n_refiner_layers=1,
        n_heads=HEADS,
        cap_feat_dim=32,
        axes_dims=[8, 12, 12],
        axes_lens=[64, 32, 32],
    )
    transformer.set_dtype(mx.bfloat16)
    latents = mx.random.normal((16, 1, 8, 8)).astype(mx.bfloat16)
    cap_feats = mx.random.normal((7, 32)).astype(mx.bfloat16)
    args = {"x": latents, "timestep": mx.array([0.5]), "sigmas": mx.array([0.5]), "cap_feats": cap_feats}

    transformer.gradient_checkpointing = False
    plain = transformer(**args)
    transformer.gradient_checkpointing = True
    checkpointed = transformer(**args)
    mx.eval(plain, checkpointed)
    assert mx.array_equal(plain, checkpointed)
