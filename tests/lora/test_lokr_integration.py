import mlx.core as mx
import pytest
from mlx import nn

from mflux.models.common.lora.layer.linear_lokr_layer import LoKrLinear
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.flux.weights.flux_lora_mapping import FluxLoRAMapping
from mflux.models.flux2.weights.flux2_lora_mapping import Flux2LoRAMapping


class _KleinAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_qkv_mlp_proj = nn.Linear(8, 4, bias=False)
        self.to_out = nn.Linear(4, 4, bias=False)
        self.to_qkv_mlp_proj.weight = mx.zeros((4, 8))
        self.to_out.weight = mx.zeros((4, 4))


class _KleinBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _KleinAttn()


class _KleinTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.single_transformer_blocks = [_KleinBlock()]


class _Flux1Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(4, 2, bias=False)
        self.to_q.weight = mx.zeros((2, 4))


class _Flux1Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _Flux1Attn()


class _Flux1Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.single_transformer_blocks = [_Flux1Block()]


@pytest.mark.fast
def test_flux2_klein_lokr_safetensors_apply_layers(tmp_path):
    lora_path = tmp_path / "klein_lycoris_lokr.safetensors"
    mx.save_safetensors(
        str(lora_path),
        {
            "lycoris_single_transformer_blocks_0_attn_to_qkv_mlp_proj.lokr_w1": mx.ones((1, 1)),
            "lycoris_single_transformer_blocks_0_attn_to_qkv_mlp_proj.lokr_w2": mx.ones((4, 8)),
            "lycoris_single_transformer_blocks_0_attn_to_out.lokr_w1": mx.ones((1, 1)),
            "lycoris_single_transformer_blocks_0_attn_to_out.lokr_w2": mx.eye(4),
            "lora_unet_single_transformer_blocks_0_attn_to_out.dora_scale": mx.ones((4, 1)),
        },
    )

    transformer = _KleinTransformer()
    LoRALoader._apply_single_lora(
        transformer,
        str(lora_path),
        scale=1.0,
        lora_mapping=Flux2LoRAMapping.get_mapping(),
        role=None,
    )

    assert isinstance(transformer.single_transformer_blocks[0].attn.to_qkv_mlp_proj, LoKrLinear)
    assert isinstance(transformer.single_transformer_blocks[0].attn.to_out, LoKrLinear)


def test_flux2_klein_lokr_load_and_bake(tmp_path):
    lora_path = tmp_path / "klein_lycoris_lokr.safetensors"
    mx.save_safetensors(
        str(lora_path),
        {
            "lycoris_single_transformer_blocks_0_attn_to_qkv_mlp_proj.lokr_w1": mx.ones((1, 1)),
            "lycoris_single_transformer_blocks_0_attn_to_qkv_mlp_proj.lokr_w2": mx.ones((4, 8)),
            "lycoris_single_transformer_blocks_0_attn_to_out.lokr_w1": mx.ones((1, 1)),
            "lycoris_single_transformer_blocks_0_attn_to_out.lokr_w2": mx.eye(4),
        },
    )

    transformer = _KleinTransformer()
    LoRALoader.load_and_apply_lora(
        lora_mapping=Flux2LoRAMapping.get_mapping(),
        transformer=transformer,
        lora_paths=[str(lora_path)],
        lora_scales=[1.0],
        bake_lora=True,
    )

    assert isinstance(transformer.single_transformer_blocks[0].attn.to_qkv_mlp_proj, nn.Linear)
    assert isinstance(transformer.single_transformer_blocks[0].attn.to_out, nn.Linear)


def test_flux2_klein_lokr_load_skips_bake_when_disabled(tmp_path):
    lora_path = tmp_path / "klein_lycoris_lokr.safetensors"
    mx.save_safetensors(
        str(lora_path),
        {
            "lycoris_single_transformer_blocks_0_attn_to_qkv_mlp_proj.lokr_w1": mx.ones((1, 1)),
            "lycoris_single_transformer_blocks_0_attn_to_qkv_mlp_proj.lokr_w2": mx.ones((4, 8)),
        },
    )

    transformer = _KleinTransformer()
    LoRALoader.load_and_apply_lora(
        lora_mapping=Flux2LoRAMapping.get_mapping(),
        transformer=transformer,
        lora_paths=[str(lora_path)],
        lora_scales=[1.0],
        bake_lora=False,
    )

    assert isinstance(transformer.single_transformer_blocks[0].attn.to_qkv_mlp_proj, LoKrLinear)


@pytest.mark.fast
def test_flux1_lycoris_lokr_safetensors_apply_layers(tmp_path):
    lora_path = tmp_path / "flux1_lycoris_lokr.safetensors"
    mx.save_safetensors(
        str(lora_path),
        {
            "lycoris_single_transformer_blocks_0_attn_to_q.lokr_w1": mx.ones((1, 1)),
            "lycoris_single_transformer_blocks_0_attn_to_q.lokr_w2": mx.ones((2, 4)),
            "lycoris_single_transformer_blocks_0_attn_to_q.dora_scale": mx.ones((2, 1)),
        },
    )

    transformer = _Flux1Transformer()
    LoRALoader._apply_single_lora(
        transformer,
        str(lora_path),
        scale=0.5,
        lora_mapping=FluxLoRAMapping.get_mapping(),
        role=None,
    )

    assert isinstance(transformer.single_transformer_blocks[0].attn.to_q, LoKrLinear)
