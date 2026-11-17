import json

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mflux.models.common.quantization import w8a8
from mflux.models.common.quantization.w8a8 import GROUP, H256, W8A8Linear
from mflux.models.qwen21.model.qwen21_transformer.qwen_image21_transformer import QwenImage21Transformer
from mflux.models.qwen21.weights import qwen21_w8a8

# W8A8 needs input widths that are multiples of 256: hidden 4 x 64 = 256, feed-forward 512.
TINY = dict(
    num_layers=2,
    num_attention_heads=4,
    attention_head_dim=64,
    axes_dims_rope=(16, 24, 24),
    context_in_dim=64,
    in_channels=4,
    out_channels=4,
    mlp_ratio=2,
    eps=1e-6,
    causal_condition=True,
)
QUANT = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GROUP})


def convrot(weight: mx.array) -> tuple[mx.array, mx.array]:
    # What a ComfyUI int8_convrot file stores: per-row int8 of W H256 (block-diagonal along the input) and the scales.
    out, width = weight.shape
    rotated = (weight.astype(mx.float32).reshape(out, width // GROUP, GROUP) @ H256).reshape(out, width)
    scale = mx.maximum(mx.max(mx.abs(rotated), axis=1) / 127, 1e-20)
    return mx.clip(mx.round(rotated / scale[:, None]), -127, 127).astype(mx.int8), scale.reshape(-1, 1)


def rel(a: mx.array, b: mx.array) -> float:
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return (mx.linalg.norm(a - b) / mx.linalg.norm(b)).item()


@pytest.fixture(params=["fallback", "tensor_ops"])
def backend(request, monkeypatch):
    if request.param == "tensor_ops" and not w8a8.tensor_ops_available():
        pytest.skip("the INT8 TensorOps GEMM needs an Apple M5 GPU")
    if request.param == "fallback":
        monkeypatch.setattr(w8a8, "tensor_ops_available", lambda: False)
    return request.param


@pytest.mark.fast
def test_w8a8_linear_matches_the_float_projection(backend):
    mx.random.seed(0)
    weight = mx.random.normal((384, 512)) / 20
    x = mx.random.normal((2, 7, 512)).astype(mx.bfloat16)
    layer = W8A8Linear(512, 384)
    layer.weight, scale = convrot(weight)
    layer.scale = scale.reshape(-1)
    y = layer(x)
    assert y.shape == (2, 7, 384) and y.dtype == mx.bfloat16
    assert rel(y, x.astype(mx.float32) @ weight.T) < 0.02


@pytest.mark.fast
def test_rotate_quantize_kernel_matches_the_reference():
    if not w8a8.tensor_ops_available():
        pytest.skip("checked on Apple M5 GPUs, where the kernel runs")
    mx.random.seed(1)
    x = (mx.random.normal((33, 768)) * 3).astype(mx.bfloat16)
    q, s = w8a8.rotate_quantize(x)
    q_ref, s_ref = w8a8.rotate_quantize_reference(x)
    assert mx.allclose(s, s_ref, rtol=1e-5).item()
    diff = mx.abs(q.astype(mx.int32) - q_ref.astype(mx.int32))
    assert mx.max(diff).item() <= 1  # rounding ties only
    assert mx.mean((diff > 0).astype(mx.float32)).item() < 1e-3


@pytest.mark.fast
def test_load_reads_an_int8_convrot_dit(tmp_path, backend):
    mx.random.seed(2)
    reference = QwenImage21Transformer(TINY)
    stored = {}
    for key, value in tree_flatten(reference.parameters()):
        prefix = key.removesuffix(".weight")
        if key.endswith(".weight") and any(prefix.endswith("." + p) for p in qwen21_w8a8.PROJECTIONS):
            if prefix.endswith(("img_mlp.gate_layer", "img_mlp.proj")):
                continue  # stored fused as gate_up below
            stored[key], stored[prefix + ".weight_scale"] = convrot(value)
            stored[prefix + ".comfy_quant"] = mx.array(np.frombuffer(QUANT.encode(), np.uint8))
        else:
            stored[key] = value
    for index, block in enumerate(reference.transformer_blocks):
        prefix = f"transformer_blocks.{index}.img_mlp.gate_up"
        gate_up = mx.concatenate([block.img_mlp.gate_layer.weight, block.img_mlp.proj.weight])
        stored[prefix + ".weight"], stored[prefix + ".weight_scale"] = convrot(gate_up)
        stored[prefix + ".comfy_quant"] = mx.array(np.frombuffer(QUANT.encode(), np.uint8))
    path = tmp_path / "qwen_image_2.1_int8_convrot.safetensors"
    mx.save_safetensors(str(path), stored)

    model = QwenImage21Transformer(TINY)
    qwen21_w8a8.convert(model)
    qwen21_w8a8.load(model, path)

    block, ref_block = model.transformer_blocks[1], reference.transformer_blocks[1]
    loaded = dict(tree_flatten(model.parameters()))
    floats = [key for key, value in tree_flatten(reference.parameters()) if key in stored and value.dtype != mx.int8]
    assert floats and all(mx.array_equal(loaded[key], stored[key]).item() for key in floats)
    x = mx.random.normal((1, 9, 256)).astype(mx.bfloat16)
    for got, want in zip(block.attn._project_qkv(x), ref_block.attn._project_qkv(x)):
        assert rel(got, want) < 0.03
    assert rel(block.img_mlp(x), ref_block.img_mlp(x)) < 0.03
    # The packed QKV weight is the only copy: each projection is a view of its rows.
    packed, _, sizes = block.attn._w8a8_qkv
    assert sizes == [256, 256, 256]
    assert mx.array_equal(block.attn.to_k.weight, packed[256:512]).item()


@pytest.mark.fast
def test_load_rejects_another_quantization(tmp_path):
    model = QwenImage21Transformer(TINY)
    qwen21_w8a8.convert(model)
    bad = json.dumps({"format": "float8_e4m3fn"})
    path = tmp_path / "other.safetensors"
    mx.save_safetensors(
        str(path), {"transformer_blocks.0.attn.to_q.comfy_quant": mx.array(np.frombuffer(bad.encode(), np.uint8))}
    )
    with pytest.raises(ValueError, match="unsupported quantization"):
        qwen21_w8a8.load(model, path)
