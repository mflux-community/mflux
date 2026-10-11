import mlx.core as mx
import pytest

from mflux.models.common.cli.save import MODEL_CLASSES
from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_text_encoder import Qwen21TextEncoder
from mflux.models.qwen21.model.qwen21_vae.qwen21_vae import Qwen21VAE
from mflux.models.qwen21.qwen21_initializer import Qwen21Initializer
from mflux.models.qwen21.variants.edit.qwen_image_21_edit import QwenImage21Edit

# Small text-only encoder: only its attributes matter here.
TEXT_ONLY = Qwen21TextEncoder(
    vocab_size=8,
    hidden_size=64,
    num_hidden_layers=1,
    num_attention_heads=2,
    num_key_value_heads=1,
    intermediate_size=64,
    head_dim=32,
    mrope_section=[6, 5, 5],
)


@pytest.mark.fast
@pytest.mark.parametrize("key", ["qwen-image-2.1", "qwen-image-2.1-turbo"])
def test_mflux_save_writes_the_edit_capable_checkpoint(key):
    assert MODEL_CLASSES[key] is QwenImage21Edit


@pytest.mark.fast
def test_text_encoder_keys_of_an_edit_save_map_to_the_text_only_encoder():
    one = mx.zeros((1,))
    supplied = {
        "language_model.embed_tokens.weight": one,
        "language_model.layers.0.mlp.up_proj.scales": one,
        "language_model.norm.weight": one,
        "visual.blocks.0.attn.qkv.weight": one,
        "lm_head.weight": one,
    }
    normalized = Qwen21Initializer._normalize_text_encoder_weights(supplied, TEXT_ONLY)
    assert sorted(normalized) == ["embed_tokens.weight", "layers.0.mlp.up_proj.scales", "norm.weight"]


@pytest.mark.fast
def test_text_encoder_keys_of_a_text_only_checkpoint_stay_the_same():
    supplied = {"embed_tokens.weight": mx.zeros((1,)), "norm.weight": mx.zeros((1,))}
    assert Qwen21Initializer._normalize_text_encoder_weights(supplied, TEXT_ONLY) is supplied


@pytest.mark.fast
def test_vae_drops_the_video_time_conv_layers_of_an_edit_save():
    one = mx.zeros((1,))
    supplied = {
        "decoder.up_blocks.0.upsampler.time_conv.weight": one,
        "encoder.down_blocks.1.downsampler.time_conv.bias": one,
        "decoder.conv_in.weight": one,
    }
    assert sorted(Qwen21Initializer._normalize_vae_weights(supplied, Qwen21VAE())) == ["decoder.conv_in.weight"]
