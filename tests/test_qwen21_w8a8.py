import json

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mflux.models.common.quantization import w8a8
from mflux.models.common.quantization.w8a8 import GROUP, H256, W8A8Linear
from mflux.models.qwen21.model.qwen21_transformer.qwen_image21_transformer import QwenImage21Transformer
from mflux.models.qwen21.weights.qwen21_w8a8 import PROJECTIONS, Qwen21W8A8

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
@pytest.mark.parametrize("rows", [1, 14, 77, 129])
@pytest.mark.parametrize("out", [64, 96, 200, 384])
def test_gemm_partial_tiles_match_the_integer_product(rows, out):
    # Rows and output widths that are not multiples of the tile (BM 128, BN 64) must give the exact
    # int8 product, with every tile reading only inside xq and the weight.
    if not w8a8.tensor_ops_available():
        pytest.skip("the INT8 TensorOps GEMM needs an Apple M5 GPU")
    mx.random.seed(3)
    x = mx.random.normal((rows, 512)).astype(mx.bfloat16)
    weight, scale = convrot(mx.random.normal((out, 512)) / 20)
    scale = scale.reshape(-1)
    y = w8a8.matmul(x, weight, scale)
    xq, xs = w8a8.rotate_quantize(x)
    acc = np.array(xq).astype(np.int64) @ np.array(weight).astype(np.int64).T
    ref = (acc.astype(np.float32) * np.array(xs)[:, None]) * np.array(scale)[None, :]
    assert y.shape == (rows, out)
    assert mx.array_equal(y, mx.array(ref).astype(mx.bfloat16)).item()


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


def write_int8_convrot(tmp_path, reference) -> tuple:
    # What ComfyUI stores for a DiT: int8 convrot block projections (gate/up fused), everything else as is.
    stored = {}
    for key, value in tree_flatten(reference.parameters()):
        prefix = key.removesuffix(".weight")
        if key.endswith(".weight") and any(prefix.endswith("." + p) for p in PROJECTIONS):
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
    return path, stored


@pytest.mark.fast
def test_load_reads_an_int8_convrot_dit(tmp_path, backend):
    mx.random.seed(2)
    reference = QwenImage21Transformer(TINY)
    path, stored = write_int8_convrot(tmp_path, reference)

    model = QwenImage21Transformer(TINY)
    Qwen21W8A8.convert(model)
    Qwen21W8A8.load(model, path)

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
    Qwen21W8A8.convert(model)
    bad = json.dumps({"format": "float8_e4m3fn"})
    path = tmp_path / "other.safetensors"
    mx.save_safetensors(
        str(path), {"transformer_blocks.0.attn.to_q.comfy_quant": mx.array(np.frombuffer(bad.encode(), np.uint8))}
    )
    with pytest.raises(ValueError, match="unsupported quantization"):
        Qwen21W8A8.load(model, path)


@pytest.mark.fast
@pytest.mark.parametrize("drop", ["transformer_blocks.1.attn.to_k.weight_scale", "norm_out.linear.weight"])
def test_load_rejects_an_incomplete_checkpoint(tmp_path, drop):
    # A missing scale would otherwise stay at zero and silence the layer without an error.
    path, stored = write_int8_convrot(tmp_path, QwenImage21Transformer(TINY))
    del stored[drop]
    mx.save_safetensors(str(path), stored)
    model = QwenImage21Transformer(TINY)
    Qwen21W8A8.convert(model)
    with pytest.raises(ValueError, match="does not match the transformer"):
        Qwen21W8A8.load(model, path)


@pytest.mark.fast
def test_w8a8_skips_the_float_dit_download_and_refuses_to_save(tmp_path, monkeypatch):
    patterns = ["vae/*.safetensors", "transformer/*.safetensors", "transformer/*.json"]
    monkeypatch.delenv("MFLUX_QWEN21_W8A8", raising=False)
    assert Qwen21W8A8.download_patterns(patterns) == patterns
    monkeypatch.setenv("MFLUX_QWEN21_W8A8", str(tmp_path))
    assert Qwen21W8A8.download_patterns(patterns) == ["vae/*.safetensors", "transformer/*.json"]

    class Model:
        transformer = QwenImage21Transformer(TINY)

    model = Model()
    Qwen21W8A8.check_savable(model)  # an ordinary model saves
    Qwen21W8A8.prepare(model, object, quantize=None, lora_paths=None)
    with pytest.raises(ValueError, match="cannot be saved"):
        Qwen21W8A8.check_savable(model)


@pytest.mark.fast
def test_load_rejects_an_integer_scale(tmp_path):
    # W8A8Linear multiplies by its scale: an int32 scale must be refused like a missing one.
    path, stored = write_int8_convrot(tmp_path, QwenImage21Transformer(TINY))
    key = "transformer_blocks.0.attn.to_q.weight_scale"
    stored[key] = stored[key].astype(mx.int32)
    mx.save_safetensors(str(path), stored)
    model = QwenImage21Transformer(TINY)
    Qwen21W8A8.convert(model)
    with pytest.raises(ValueError, match="wrong shape or dtype"):
        Qwen21W8A8.load(model, path)


@pytest.mark.fast
@pytest.mark.parametrize("initializer", ["text_to_image", "edit"])
def test_w8a8_settings_fail_before_any_download(tmp_path, monkeypatch, initializer):
    from mflux.models.common.resolution.path_resolution import PathResolution
    from mflux.models.qwen21.qwen21_initializer import Qwen21Initializer
    from mflux.models.qwen21.qwen_image21_initializer import QwenImage21Initializer

    def no_download(*args, **kwargs):
        raise AssertionError("downloaded before the W8A8 settings were checked")

    monkeypatch.setattr(PathResolution, "resolve", staticmethod(no_download))
    monkeypatch.setenv("MFLUX_QWEN21_W8A8", str(tmp_path / "missing.safetensors"))
    init = Qwen21Initializer.init if initializer == "text_to_image" else QwenImage21Initializer.init
    kwargs = dict(quantize=None, model_path=None, model_config=None)
    with pytest.raises(FileNotFoundError, match="missing file"):
        init(object(), **kwargs)
    monkeypatch.setenv("MFLUX_QWEN21_W8A8", str(tmp_path))
    with pytest.raises(ValueError, match="quantize or LoRAs"):
        init(object(), **{**kwargs, "quantize": 8})
