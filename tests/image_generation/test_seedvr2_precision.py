import pytest

pytest.importorskip("mlx.core", exc_type=ImportError)

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_map, tree_map_with_path

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.seedvr2.model.seedvr2_transformer.transformer import SeedVR2Transformer
from mflux.models.seedvr2.model.seedvr2_vae.vae import SeedVR2VAE
from mflux.models.seedvr2.weights.seedvr2_weight_definition import (
    SeedVR2WeightDefinition,
    SeedVR2WeightDefinition3B,
    SeedVR2WeightDefinition7B,
)


def _cast_module(module: nn.Module, dtype: mx.Dtype, *, keep_rope_freqs_fp16: bool = False) -> None:
    def cast_leaf(path: str, array: mx.array) -> mx.array:
        if keep_rope_freqs_fp16 and path.endswith(".rope.freqs"):
            return array.astype(mx.float16)
        return array.astype(dtype)

    module.update(tree_map_with_path(cast_leaf, module.parameters()))


@pytest.mark.fast
@pytest.mark.parametrize("weight_def", [SeedVR2WeightDefinition3B, SeedVR2WeightDefinition7B])
def test_seedvr2_weight_definitions_cast_weights_but_preserve_rope_freqs(weight_def):
    components = {c.name: c for c in weight_def.get_components()}

    assert components["vae"].precision == ModelConfig.precision
    transform = components["transformer"].weight_transform
    assert transform is not None

    linear_weight = mx.ones((4, 4), dtype=mx.float16)
    rope_freqs = mx.array([1.0, 402.123], dtype=mx.float16)

    assert transform("blocks.0.attn.proj_qkv.vid.weight", linear_weight).dtype == ModelConfig.precision
    assert transform("blocks.0.attn.rope.rope.freqs", rope_freqs).dtype == mx.float16


@pytest.mark.fast
@pytest.mark.parametrize("weight_dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("quantize_bits", [None, 8])
def test_seedvr2_transformer_preserves_weight_dtype_with_mixed_inputs(weight_dtype, quantize_bits):
    mx.random.seed(0)
    transformer = SeedVR2Transformer(
        vid_in_channels=16,
        vid_out_channels=16,
        vid_dim=64,
        txt_in_dim=64,
        heads=1,
        head_dim=64,
        rope_dim=64,
        num_layers=2,
        mm_layers=1,
        window=(2, 2, 2),
    )
    _cast_module(transformer, weight_dtype, keep_rope_freqs_fp16=True)
    if quantize_bits is not None:
        nn.quantize(
            transformer,
            bits=quantize_bits,
            class_predicate=SeedVR2WeightDefinition.quantization_predicate,
        )
    mx.eval(transformer.parameters())

    vid = mx.random.normal((1, 16, 1, 4, 4)).astype(mx.float32)
    txt = mx.random.normal((1, 4, 64)).astype(mx.float16)
    timestep = mx.array(1000.0, dtype=mx.float32)

    emb = transformer.emb_in(timestep)
    assert emb.dtype == weight_dtype

    out = transformer(vid=vid, txt=txt, timestep=timestep)
    assert out.dtype == weight_dtype


@pytest.mark.fast
@pytest.mark.parametrize("weight_dtype", [mx.bfloat16, mx.float16])
def test_seedvr2_vae_preserves_weight_dtype_with_fp32_inputs(weight_dtype):
    mx.random.seed(0)
    vae = SeedVR2VAE(block_out_channels=(32, 32), latent_channels=16)
    vae.update(tree_map(lambda a: a.astype(weight_dtype), vae.parameters()))
    mx.eval(vae.parameters())

    image = mx.random.normal((1, 3, 16, 16)).astype(mx.float32)
    latent = vae.encode(image)
    assert latent.dtype == weight_dtype

    decoded = vae.decode(latent.astype(mx.float32))
    assert decoded.dtype == weight_dtype
