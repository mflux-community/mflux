import mlx.core as mx
import mlx.nn as nn
import pytest

from mflux.models.common.lora.layer.dense_weight import dense_weight
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.lora.mapping.lora_mapping import LoRATarget
from mflux.models.z_image.model.z_image_transformer.transformer import ZImageTransformer
from mflux.models.z_image.weights.z_image_lora_mapping import ZImageLoRAMapping


@pytest.mark.fast
@pytest.mark.parametrize("layer_type", ["layers", "noise_refiner", "context_refiner"])
@pytest.mark.parametrize("bits", [None, 4, 8])
@pytest.mark.parametrize("bake_lora", [False, True])
def test_fused_qkv_matches_dense_reference(tmp_path, layer_type, bits, bake_lora):
    mx.random.seed(0)
    model = ZImageTransformer(dim=64, n_layers=1, n_refiner_layers=1, n_heads=2, cap_feat_dim=64)
    attention = getattr(model, layer_type)[0].attention
    if bits:
        nn.quantize(attention, group_size=64, bits=bits)
    down = mx.random.normal((2, 64)) * 0.05
    up = mx.random.normal((192, 2)) * 0.05
    x = mx.random.normal((3, 64))
    weight = mx.concatenate([dense_weight(getattr(attention, name)) for name in ("to_q", "to_k", "to_v")])
    merged = weight + 0.7 * ((up * (3 / 2)) @ down)
    if bits and bake_lora:
        merged = mx.dequantize(*mx.quantize(merged, group_size=64, bits=8), group_size=64, bits=8)
    expected = x @ merged.T
    mx.eval(expected)
    source = f"diffusion_model.{layer_type}.0.attention.qkv"
    adapter = tmp_path / "qkv.safetensors"
    mx.save_safetensors(
        str(adapter),
        {f"{source}.lora_down.weight": down, f"{source}.lora_up.weight": up, f"{source}.alpha": mx.array(3.0)},
    )

    LoRALoader.load_and_apply_lora(ZImageLoRAMapping.get_mapping(), model, [str(adapter)], [0.7], bake_lora=bake_lora)

    actual = mx.concatenate([getattr(attention, name)(x) for name in ("to_q", "to_k", "to_v")], axis=-1)
    assert mx.allclose(actual, expected, atol=1e-6).item()
    if bits and bake_lora:
        assert attention.to_q.bits == 8


@pytest.mark.fast
@pytest.mark.parametrize(
    "source,path",
    [
        ("x_embedder", "all_x_embedder.2-1"),
        ("final_layer.linear", "all_final_layer.2-1.linear"),
        ("final_layer.adaLN_modulation.1", "all_final_layer.2-1.adaLN_modulation.0"),
        ("cap_embedder.1", "cap_embedder.1"),
        ("t_embedder.mlp.0", "t_embedder.linear1"),
        ("t_embedder.mlp.2", "t_embedder.linear2"),
        ("layers.0.adaLN_modulation.0", "layers.0.adaLN_modulation.0"),
        ("noise_refiner.0.adaLN_modulation.0", "noise_refiner.0.adaLN_modulation.0"),
    ],
)
@pytest.mark.parametrize("bits", [None, 4, 8])
def test_linear_lora_and_bias_delta(tmp_path, source, path, bits):
    mx.random.seed(0)
    model = ZImageTransformer(dim=64, n_layers=1, n_refiner_layers=1, n_heads=2, cap_feat_dim=64)
    layer = LoRALoader._get_target_module(model, path)
    if bits:
        layer = nn.QuantizedLinear.from_linear(layer, group_size=64, bits=bits)
        LoRALoader._replace_target_module(model, path, layer)
    weight = dense_weight(layer)
    down = mx.random.normal((2, weight.shape[1])) * 0.05
    up = mx.random.normal((weight.shape[0], 2)) * 0.05
    bias_delta = mx.random.normal(layer.bias.shape) * 0.05
    expected_bias = layer.bias - 0.4 * bias_delta
    x = mx.random.normal((2, weight.shape[1]))
    expected = x @ (weight - 0.4 * (up @ down)).T + layer.bias - 0.4 * bias_delta
    mx.eval(expected)
    source = f"diffusion_model.{source}"
    adapter = tmp_path / "linear.safetensors"
    mx.save_safetensors(
        str(adapter),
        {f"{source}.lora_down.weight": down, f"{source}.lora_up.weight": up, f"{source}.diff_b": bias_delta},
    )

    # Exercise both an existing LoRALinear and the fused wrapper before baking.
    LoRALoader.load_and_apply_lora(ZImageLoRAMapping.get_mapping(), model, [str(adapter)] * 3, [0.7, -0.3, -0.8])

    actual = LoRALoader._get_target_module(model, path)(x)
    assert mx.allclose(actual, expected, atol=0.02 if bits else 2e-6).item()
    assert mx.allclose(LoRALoader._get_target_module(model, path).bias, expected_bias, atol=1e-6).item()


@pytest.mark.fast
@pytest.mark.parametrize(
    "source,path",
    [
        ("cap_embedder.0", "cap_embedder.0"),
        ("layers.0.attention.q_norm", "layers.0.attention.norm_q"),
        ("layers.0.attention.k_norm", "layers.0.attention.norm_k"),
        ("noise_refiner.0.attention_norm1", "noise_refiner.0.attention_norm1"),
        ("context_refiner.0.attention_norm2", "context_refiner.0.attention_norm2"),
        ("layers.0.ffn_norm1", "layers.0.ffn_norm1"),
        ("layers.0.ffn_norm2", "layers.0.ffn_norm2"),
    ],
)
@pytest.mark.parametrize("scale", [0, -0.5, 1.0])
def test_norm_delta_matches_reference(tmp_path, source, path, scale):
    model = ZImageTransformer(dim=64, n_layers=1, n_refiner_layers=1, n_heads=2, cap_feat_dim=64)
    norm = LoRALoader._get_target_module(model, path)
    delta = mx.arange(norm.weight.size, dtype=mx.float32) / 100
    x = mx.ones((2, norm.weight.size))
    expected = mx.fast.rms_norm(x, norm.weight + scale * delta, norm.eps)
    mx.eval(expected)
    adapter = tmp_path / "norm.safetensors"
    mx.save_safetensors(str(adapter), {f"diffusion_model.{source}.diff": delta})

    LoRALoader.load_and_apply_lora(ZImageLoRAMapping.get_mapping(), model, [str(adapter)], [scale])

    assert mx.allclose(norm(x), expected, atol=1e-6).item()


@pytest.mark.fast
@pytest.mark.parametrize("bake_lora,role", [(False, None), (True, "train")])
def test_direct_patch_rejects_unbaked_or_role_before_changes(tmp_path, bake_lora, role):
    model = ZImageTransformer(dim=64, n_layers=1, n_refiner_layers=1, n_heads=2, cap_feat_dim=64)
    original_linear = model.all_x_embedder["2-1"]
    adapter = tmp_path / "patch.safetensors"
    mx.save_safetensors(
        str(adapter),
        {
            "diffusion_model.all_x_embedder.2-1.lora_A.weight": mx.ones((2, 64)),
            "diffusion_model.all_x_embedder.2-1.lora_B.weight": mx.ones((64, 2)),
            "diffusion_model.cap_embedder.0.diff": mx.ones((64,)),
        },
    )
    with pytest.raises(ValueError, match="require bake_lora=True"):
        LoRALoader.load_and_apply_lora(
            ZImageLoRAMapping.get_mapping(), model, [str(adapter)], [1], bake_lora=bake_lora, role=role
        )
    assert mx.array_equal(model.cap_embedder[0].weight, mx.ones((64,))).item()
    assert model.all_x_embedder["2-1"] is original_linear


@pytest.mark.fast
@pytest.mark.parametrize("bake_lora,role", [(False, None), (True, "train")])
@pytest.mark.parametrize("suffix", ["diff", "diff_b"])
def test_unmatched_direct_patch_does_not_block_lora(tmp_path, capsys, bake_lora, role, suffix):
    model = nn.Module()
    model.proj = nn.Linear(4, 4)
    x = mx.ones((1, 4))
    expected = model.proj(x) + 8
    mx.eval(expected)
    target = LoRATarget(
        "proj",
        ["proj.lora_up.weight"],
        ["proj.lora_down.weight"],
        possible_diff_patterns=["proj.diff"],
        possible_diff_b_patterns=["proj.diff_b"],
    )
    adapter = tmp_path / "adapter.safetensors"
    mx.save_safetensors(
        str(adapter),
        {
            "proj.lora_down.weight": mx.ones((2, 4)),
            "proj.lora_up.weight": mx.ones((4, 2)),
            f"unrelated.proj.{suffix}": mx.ones((4,)),
        },
    )

    LoRALoader.load_and_apply_lora([target], model, [str(adapter)], [1], bake_lora=bake_lora, role=role)

    assert mx.allclose(model.proj(x), expected, atol=1e-6).item()
    assert f"unrelated.proj.{suffix}" in capsys.readouterr().out


@pytest.mark.fast
@pytest.mark.parametrize("suffix,shape", [("diff", (1,)), ("diff", (1, 64)), ("diff_b", (64,))])
def test_direct_patch_rejects_broadcasting_and_missing_bias(tmp_path, suffix, shape):
    model = nn.Module()
    model.norm = nn.RMSNorm(64)
    target = LoRATarget("norm", [], [], possible_diff_patterns=["norm.diff"], possible_diff_b_patterns=["norm.diff_b"])
    adapter = tmp_path / "bad.safetensors"
    mx.save_safetensors(str(adapter), {f"norm.{suffix}": mx.ones(shape)})
    with pytest.raises(ValueError, match="Direct patch shape mismatch"):
        LoRALoader.load_and_apply_lora([target], model, [str(adapter)], [1])


@pytest.mark.fast
def test_fused_qkv_rejects_unequal_chunks(tmp_path):
    model = ZImageTransformer(dim=64, n_layers=1, n_refiner_layers=1, n_heads=2, cap_feat_dim=64)
    adapter = tmp_path / "bad-qkv.safetensors"
    source = "diffusion_model.layers.0.attention.qkv"
    mx.save_safetensors(
        str(adapter), {f"{source}.lora_down.weight": mx.ones((2, 64)), f"{source}.lora_up.weight": mx.ones((193, 2))}
    )
    with pytest.raises(ValueError, match="three equal parts"):
        LoRALoader.load_and_apply_lora(ZImageLoRAMapping.get_mapping(), model, [str(adapter)], [1])


@pytest.mark.fast
def test_failed_load_leaves_direct_patches_unapplied(tmp_path):
    model = ZImageTransformer(dim=64, n_layers=1, n_refiner_layers=1, n_heads=2, cap_feat_dim=64)
    norm_weight = model.cap_embedder[0].weight
    bias = model.all_x_embedder["2-1"].bias
    adapter = tmp_path / "partial.safetensors"
    mx.save_safetensors(
        str(adapter),
        {
            "diffusion_model.cap_embedder.0.diff": mx.ones((64,)),
            "diffusion_model.x_embedder.diff_b": mx.ones(bias.shape),
            "diffusion_model.x_embedder.lora_down.weight": mx.ones((2, 64)),
        },
    )
    with pytest.raises(ValueError, match="could not be applied"):
        LoRALoader.load_and_apply_lora(ZImageLoRAMapping.get_mapping(), model, [str(adapter)], [1])
    assert mx.array_equal(model.cap_embedder[0].weight, norm_weight).item()
    assert mx.array_equal(model.all_x_embedder["2-1"].bias, bias).item()


@pytest.mark.fast
def test_direct_patch_on_missing_target_reports_failure(tmp_path):
    model = nn.Module()
    model.norm = nn.RMSNorm(4)
    targets = [
        LoRATarget("norm", [], [], possible_diff_patterns=["norm.diff"]),
        LoRATarget("blocks.3.norm", [], [], possible_diff_patterns=["blocks.3.norm.diff"]),
    ]
    adapter = tmp_path / "missing.safetensors"
    mx.save_safetensors(str(adapter), {"norm.diff": mx.ones((4,)), "blocks.3.norm.diff": mx.ones((4,))})
    with pytest.raises(ValueError, match="could not be applied"):
        LoRALoader.load_and_apply_lora(targets, model, [str(adapter)], [1])
    assert mx.array_equal(model.norm.weight, mx.ones((4,))).item()


@pytest.mark.fast
def test_fused_and_separate_projection_keys_conflict(tmp_path):
    model = ZImageTransformer(dim=64, n_layers=1, n_refiner_layers=1, n_heads=2, cap_feat_dim=64)
    original = model.layers[0].attention.to_q
    adapter = tmp_path / "mixed.safetensors"
    prefix = "diffusion_model.layers.0.attention"
    mx.save_safetensors(
        str(adapter),
        {
            f"{prefix}.qkv.lora_down.weight": mx.ones((2, 64)),
            f"{prefix}.qkv.lora_up.weight": mx.ones((192, 2)),
            f"{prefix}.to_q.lora_down.weight": mx.ones((2, 64)),
            f"{prefix}.to_q.lora_up.weight": mx.ones((64, 2)),
        },
    )
    with pytest.raises(ValueError, match="both map to layers.0.attention.to_q"):
        LoRALoader.load_and_apply_lora(ZImageLoRAMapping.get_mapping(), model, [str(adapter)], [1])
    assert model.layers[0].attention.to_q is original


@pytest.mark.fast
def test_later_file_failure_leaves_earlier_direct_patches_unapplied(tmp_path):
    model = ZImageTransformer(dim=64, n_layers=1, n_refiner_layers=1, n_heads=2, cap_feat_dim=64)
    norm_weight = model.cap_embedder[0].weight
    good = tmp_path / "good.safetensors"
    bad = tmp_path / "bad.safetensors"
    mx.save_safetensors(str(good), {"diffusion_model.cap_embedder.0.diff": mx.ones((64,))})
    mx.save_safetensors(str(bad), {"diffusion_model.x_embedder.lora_down.weight": mx.ones((2, 64))})
    with pytest.raises(ValueError, match="could not be applied"):
        LoRALoader.load_and_apply_lora(ZImageLoRAMapping.get_mapping(), model, [str(good), str(bad)], [1, 1])
    assert mx.array_equal(model.cap_embedder[0].weight, norm_weight).item()
