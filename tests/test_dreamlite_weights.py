from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from mlx.utils import tree_flatten, tree_map

from mflux.models.common.compute_precision import ComputePrecision
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.dreamlite.dreamlite_initializer import DreamLiteInitializer
from mflux.models.dreamlite.model.dreamlite_text_encoder.dreamlite_text_encoder import DreamLiteTextEncoder
from mflux.models.dreamlite.model.dreamlite_unet.attention import DreamLiteAttention
from mflux.models.dreamlite.model.dreamlite_unet.timestep_embedding import DreamLiteTimestepEmbedding
from mflux.models.dreamlite.model.dreamlite_unet.transformer_block import DreamLiteTransformerBlock
from mflux.models.dreamlite.model.dreamlite_unet.unet import DreamLiteUNet
from mflux.models.dreamlite.model.dreamlite_vae.vae import DreamLiteVAE
from mflux.models.dreamlite.variants.dreamlite_core import _DreamLiteCore
from mflux.models.dreamlite.weights.dreamlite_weight_definition import DreamLiteWeightDefinition

pytestmark = pytest.mark.fast

# Tensor names read from the safetensors headers of the official checkpoints (no weights), captured 2026-11-17 from
# carlofkl/DreamLite-base @ a9a0f151ffd9: unet/, vae/ and text_encoder/. carlofkl/DreamLite-mobile @ a48c656c291f
# has the same names in all three.
KEYS_DIR = Path(__file__).parent / "resources" / "checkpoint_keys"


class _Names:
    @staticmethod
    def official(component: str) -> set[str]:
        definition = next(c for c in DreamLiteWeightDefinition.get_components() if c.name == component)
        names = (KEYS_DIR / f"dreamlite_{component}.txt").read_text().split()
        transform = definition.key_transform or (lambda name: name)
        return {transform(name) for name in names}

    @staticmethod
    def module(module) -> set[str]:
        # Building the modules allocates nothing: MLX evaluates the initial weights lazily, and nothing reads them
        return {name for name, _ in tree_flatten(module.parameters())}


def test_the_unet_modules_carry_the_official_checkpoint_names():
    assert _Names.module(DreamLiteUNet()) == _Names.official("unet")


def test_the_vae_modules_carry_the_official_checkpoint_names():
    assert _Names.module(DreamLiteVAE()) == _Names.official("vae")


def test_the_text_encoder_modules_carry_the_official_checkpoint_names():
    # The rotary frequencies are computed, not stored
    computed = {"language_model.rotary_emb.inv_freq", "visual.rotary_pos_emb.inv_freq"}

    assert _Names.module(DreamLiteTextEncoder()) - computed == _Names.official("text_encoder")


def test_the_components_load_from_the_upstream_directories_at_their_precisions():
    components = {c.name: c for c in DreamLiteWeightDefinition.get_components()}

    assert {name: c.hf_subdir for name, c in components.items()} == {"vae": "vae", "unet": "unet", "text_encoder": "text_encoder"}  # fmt: skip
    assert components["unet"].precision == mx.float32
    assert components["vae"].precision == mx.float32
    assert components["text_encoder"].precision == ModelConfig.precision


def test_convolution_weights_move_to_the_mlx_layout_and_other_weights_stay():
    transform = next(c for c in DreamLiteWeightDefinition.get_components() if c.name == "unet").weight_transform
    conv = mx.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7)

    moved = transform("conv_in.weight", conv)

    assert moved.shape == (2, 5, 7, 3)
    assert moved[1, 4, 6, 2].item() == conv[1, 2, 4, 6].item()
    assert transform("time_embedding.linear_1.weight", mx.zeros((4, 3))).shape == (4, 3)


def test_upstream_bfloat16_weights_load_as_float32_for_the_unet_and_vae_only(tmp_path):
    # The official layout, one tensor per component, stored in bfloat16 with PyTorch's convolution layout
    stored = mx.ones((2, 3, 5, 7), dtype=mx.bfloat16)
    for subdir, name in (("unet", "conv_in.weight"), ("vae", "decoder.layers.0.weight")):
        (tmp_path / subdir).mkdir()
        mx.save_safetensors(str(tmp_path / subdir / "diffusion_pytorch_model.safetensors"), {name: stored})
    (tmp_path / "text_encoder").mkdir()
    mx.save_safetensors(
        str(tmp_path / "text_encoder" / "model.safetensors"),
        {"model.language_model.norm.weight": mx.ones((4,), dtype=mx.bfloat16)},
    )

    weights = WeightLoader.load(weight_definition=DreamLiteWeightDefinition, model_path=str(tmp_path)).components

    assert weights["unet"]["conv_in"]["weight"].dtype == mx.float32
    assert weights["unet"]["conv_in"]["weight"].shape == (2, 5, 7, 3)
    assert weights["vae"]["decoder"]["layers"][0]["weight"].dtype == mx.float32
    assert weights["text_encoder"]["language_model"]["norm"]["weight"].dtype == mx.bfloat16


def test_the_text_encoder_computes_in_float32_over_bfloat16_weights():
    # bfloat16 activations through 28 layers show in the image, because the UNet reads the pre-norm hidden states
    text_config = {**DreamLiteTextEncoder.TEXT_CONFIG, "vocab_size": 64, "hidden_size": 128, "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 64, "intermediate_size": 128, "rope_scaling": {"mrope_section": [12, 10, 10]}}  # fmt: skip
    vision_config = {**DreamLiteTextEncoder.VISION_CONFIG, "depth": 1, "hidden_size": 128, "intermediate_size": 128, "num_heads": 2, "out_hidden_size": 128, "num_position_embeddings": 64, "deepstack_visual_indexes": [0]}  # fmt: skip
    encoder = DreamLiteTextEncoder(text_config, vision_config)
    encoder.update(tree_map(lambda p: p.astype(mx.bfloat16), encoder.parameters()))

    hidden = encoder(mx.array([[1, 2, 3, 4, 5]]))

    assert hidden.shape == (1, 5, 128)
    assert hidden.dtype == mx.float32
    assert encoder.language_model.layers[0].self_attn.q_proj.weight.dtype == mx.bfloat16


def test_saving_writes_the_image_processor_config_beside_the_tokenizer(tmp_path, monkeypatch):
    saved = {}
    monkeypatch.setattr(ModelSaver, "save_model", staticmethod(lambda **kwargs: saved.update(kwargs)))
    core = _DreamLiteCore.__new__(_DreamLiteCore)
    nn.Module.__init__(core)
    core.bits = 8
    core.processor = SimpleNamespace(image_processor=SimpleNamespace(save_pretrained=lambda path: saved.update(processor=path)))  # fmt: skip

    core.save_model(str(tmp_path / "out"))

    assert saved["weight_definition"] is DreamLiteWeightDefinition
    assert (saved["bits"], saved["base_path"]) == (8, str(tmp_path / "out"))
    assert saved["processor"] == str(tmp_path / "out" / "processor")


def test_the_timestep_and_size_embedders_are_not_quantized():
    predicate = DreamLiteWeightDefinition.quantization_predicate
    linear = nn.Linear(64, 64)

    assert predicate("time_embedding.linear_1", linear) is False
    assert predicate("add_embedding.linear_2", linear) is False
    assert predicate("encoder_hid_proj.0", linear) is True
    assert predicate("down_blocks.0.attentions.0.proj_in", linear) is True
    assert predicate("conv_in", nn.Conv2d(4, 64, 3)) is False


def test_compute_precision_casts_the_whole_unet_and_keeps_it_from_being_saved():
    unet = DreamLiteUNet(block_channels=(64, 128, 128), num_heads=(1, 2, 2), transformer_layers=(1, 1, 1), time_proj_dim=64, time_embed_dim=128, text_dim=128, context_dim=128)  # fmt: skip

    ComputePrecision(mx.float16).apply(unet, (DreamLiteUNet,))
    out = unet(mx.random.normal((1, 4, 16, 32)), mx.array(999.0), mx.random.normal((1, 5, 128)) * 50, (1024, 1024))

    assert {value.dtype for _, value in tree_flatten(unet.parameters())} == {mx.float16}
    assert out.dtype == mx.float16
    assert mx.all(mx.isfinite(out)).item()
    with pytest.raises(ValueError, match="Cannot save a model built with compute_precision"):
        ComputePrecision.ensure_savable(unet)


def test_an_unsupported_compute_precision_fails_before_any_weight_is_read():
    with pytest.raises(ValueError, match="Unsupported compute precision"):
        DreamLiteInitializer.init(model=None, model_config=ModelConfig.dreamlite(), quantize=None, compute_precision=mx.bfloat16)  # fmt: skip


def test_only_the_processor_and_the_three_components_are_downloaded():
    patterns = DreamLiteWeightDefinition.get_download_patterns()

    assert not any(pattern.startswith(("tokenizer/", "scheduler/")) for pattern in patterns)
    assert {pattern.split("/")[0] for pattern in patterns} == {"vae", "unet", "text_encoder", "processor"}


def test_the_timestep_embedding_puts_cosines_before_sines():
    embedding = DreamLiteTimestepEmbedding.sinusoidal(mx.array([0.0, 1000.0]), 8)

    assert np.allclose(np.asarray(embedding[0]), [1, 1, 1, 1, 0, 0, 0, 0])
    assert np.isclose(embedding[1, 0].item(), np.cos(1000.0), atol=1e-4)
    assert np.isclose(embedding[1, 4].item(), np.sin(1000.0), atol=1e-4)


def test_attention_shares_one_key_value_head_across_the_query_heads():
    attention = DreamLiteAttention(dim=128, num_heads=2, context_dim=192)

    assert attention.to_q.weight.shape == (128, 128)
    assert attention.to_k.weight.shape == (64, 192)
    assert attention.to_v.weight.shape == (64, 192)
    assert attention(mx.zeros((1, 6, 128)), mx.zeros((1, 3, 192))).shape == (1, 6, 128)


def test_a_block_without_self_attention_continues_from_the_normalized_states():
    # The reference drops the residual with the self-attention: the block's input only survives through norm1
    block = DreamLiteTransformerBlock(dim=64, num_heads=1, context_dim=64, self_attention=False)
    hidden = mx.random.normal((1, 4, 64), key=mx.random.key(0))
    context = mx.random.normal((1, 3, 64), key=mx.random.key(1))

    normed = block.norm1(hidden)
    expected = normed + block.attn2(block.norm2(normed), context)
    expected = expected + block.ff(block.norm3(expected))

    assert "attn1" not in block
    assert mx.allclose(block(hidden, context), expected, atol=1e-5).item()


def test_the_unet_attends_to_itself_only_at_the_lowest_resolution():
    unet = DreamLiteUNet()

    def has_self_attention(block) -> bool:
        return "attn1" in block.attentions[0].transformer_blocks[0]

    assert [has_self_attention(block) for block in unet.down_blocks] == [False, False, True]
    assert [has_self_attention(block) for block in unet.up_blocks[:2]] == [True, False]
    assert "attentions" not in unet.up_blocks[2]
    assert has_self_attention(unet.mid_block)


def test_the_vae_decodes_eight_times_the_latent_size_and_encodes_back():
    vae = DreamLiteVAE()

    decoded = vae.decode(mx.zeros((1, 4, 4, 6)))

    assert decoded.shape == (1, 3, 32, 48)
    assert vae.encode(decoded).shape == (1, 4, 4, 6)
    assert DreamLiteVAE.supports_implicit_tiling is False
