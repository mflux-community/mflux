import mlx.core as mx
import pytest

from mflux.models.common.pid_decoder.gemma2.gemma2_config import Gemma2Config
from mflux.models.common.pid_decoder.gemma2.gemma2_model import Gemma2Model

pytestmark = pytest.mark.fast


def test_gemma2_model_output_shape():
    config = Gemma2Config(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
    )
    model = Gemma2Model(config)
    input_ids = mx.array([[1, 2, 3, 0, 0]])
    attention_mask = mx.array([[1, 1, 1, 0, 0]])
    out = model(input_ids, attention_mask)
    assert out.shape == (1, 5, 16)


def test_gemma2_model_gqa_batched_with_padding_mask_does_not_crash():
    # num_attention_heads > num_key_value_heads => repeats > 1, which reshapes
    # attention scores to 5D (B, n_kv_heads, repeats, L, L). Batch size (3) is
    # deliberately != n_kv_heads (2) and != 1: a stale 4D mask right-aligned
    # against 5D scores aliases its batch axis with n_kv_heads and is only
    # broadcastable when B in {1, n_kv_heads} — B=3 makes the pre-fix code
    # raise a broadcast error instead of silently mis-applying the mask.
    config = Gemma2Config(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    model = Gemma2Model(config)
    input_ids = mx.array([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0], [6, 7, 8, 9, 0]])
    attention_mask = mx.array([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 0]])
    out = model(input_ids, attention_mask)
    assert out.shape == (3, 5, 16)


def test_gemma2_model_is_causal():
    # Changing a later token must not change an earlier token's output hidden
    # state. This is the standard causal-masking property test: run the model
    # twice with a different trailing token and compare the earlier positions.
    config = Gemma2Config(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
    )
    model = Gemma2Model(config)
    input_ids_a = mx.array([[1, 2, 3, 4]])
    input_ids_b = mx.array([[1, 2, 3, 99]])

    out_a = model(input_ids_a)
    out_b = model(input_ids_b)

    assert mx.allclose(out_a[:, :3, :], out_b[:, :3, :], atol=1e-5).item()
    assert not mx.allclose(out_a[:, 3, :], out_b[:, 3, :], atol=1e-5).item()
