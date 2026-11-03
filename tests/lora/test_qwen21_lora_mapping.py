import json
import sys
from unittest.mock import Mock

import mlx.core as mx
import pytest
from mlx import nn

from mflux.cli.capabilities import describe_command
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.qwen21.cli import qwen21_edit_generate, qwen21_generate
from mflux.models.qwen21.model.qwen21_transformer.qwen21_time_text_embed import Qwen21TimeTextEmbed
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer_block import Qwen21TransformerBlock
from mflux.models.qwen21.weights.qwen21_lora_mapping import Qwen21LoRAMapping


@pytest.mark.fast
@pytest.mark.parametrize("bake_lora", [False, True])
@pytest.mark.parametrize("prefix,suffix", [("", ".default"), ("transformer.", ""), ("diffusion_model.", "")])
@pytest.mark.parametrize(
    "name",
    ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "img_mlp.proj", "img_mlp.gate_layer", "img_mlp.out"],
)
def test_qwen21_lora_updates_target_layer(tmp_path, name, prefix, suffix, bake_lora):
    model = nn.Module()
    model.transformer_blocks = [Qwen21TransformerBlock(dim=8, num_attention_heads=2, attention_head_dim=4)]
    path = f"transformer_blocks.0.{name}"
    linear = LoRALoader._get_target_module(model, path)
    output_dims, input_dims = linear.weight.shape
    down = mx.arange(2 * input_dims, dtype=mx.float32).reshape(2, input_dims) / 100
    up = mx.arange(output_dims * 2, dtype=mx.float32).reshape(output_dims, 2) / 100
    x = mx.ones((1, input_dims))
    expected = linear(x) + 0.7 * (x @ down.T @ up.T)
    mx.eval(expected)
    adapter = tmp_path / "adapter.safetensors"
    mx.save_safetensors(
        str(adapter), {f"{prefix}{path}.lora_A{suffix}.weight": down, f"{prefix}{path}.lora_B{suffix}.weight": up}
    )

    paths, scales = LoRALoader.load_and_apply_lora(
        Qwen21LoRAMapping.get_mapping(), model, [str(adapter)], [0.7], bake_lora=bake_lora
    )

    actual = LoRALoader._get_target_module(model, path)
    assert isinstance(actual, LoRALinear) is (not bake_lora)
    assert mx.allclose(actual(x), expected, atol=1e-6).item()
    assert paths == [str(adapter)]
    assert scales == [0.7]


@pytest.mark.fast
@pytest.mark.parametrize("bake_lora", [False, True])
@pytest.mark.parametrize(
    "name",
    ["modulation.layers.1", "time_text_embed.timestep_embedder.linear_1", "time_text_embed.timestep_embedder.linear_2"],
)
@pytest.mark.parametrize("prefix", ["transformer", "diffusion_model"])
def test_qwen21_lora_updates_global_layer(tmp_path, name, bake_lora, prefix):
    model = nn.Module()
    model.modulation = nn.Sequential(nn.SiLU(), nn.Linear(8, 32, bias=False))
    model.time_text_embed = Qwen21TimeTextEmbed(embedding_dim=8)
    linear = LoRALoader._get_target_module(model, name)
    output_dims, input_dims = linear.weight.shape
    down = mx.arange(2 * input_dims, dtype=mx.float32).reshape(2, input_dims) / 100
    up = mx.arange(output_dims * 2, dtype=mx.float32).reshape(output_dims, 2) / 100
    x = mx.ones((1, input_dims))
    expected = linear(x) + 0.7 * (x @ down.T @ up.T)
    mx.eval(expected)
    adapter = tmp_path / "adapter.safetensors"
    source = f"{prefix}." + name.replace("modulation.layers.1", "modulation.1")
    mx.save_safetensors(str(adapter), {f"{source}.lora_A.weight": down, f"{source}.lora_B.weight": up})

    LoRALoader.load_and_apply_lora(Qwen21LoRAMapping.get_mapping(), model, [str(adapter)], [0.7], bake_lora=bake_lora)

    actual = LoRALoader._get_target_module(model, name)
    assert isinstance(actual, LoRALinear) is (not bake_lora)
    assert mx.allclose(actual(x), expected, atol=1e-6).item()


@pytest.mark.fast
@pytest.mark.parametrize("bake_lora", [False, True])
@pytest.mark.parametrize("prefix", ["transformer", "diffusion_model"])
def test_qwen21_lora_matches_fused_mlp(tmp_path, bake_lora, prefix):
    mx.random.seed(0)
    model = nn.Module()
    model.transformer_blocks = [Qwen21TransformerBlock(dim=8, num_attention_heads=2, attention_head_dim=4)]
    mlp = model.transformer_blocks[0].img_mlp
    output_dims, input_dims = mlp.gate_layer.weight.shape
    down = mx.arange(2 * input_dims, dtype=mx.float32).reshape(2, input_dims) / 100 - 0.1
    up = mx.arange(2 * output_dims * 2, dtype=mx.float32).reshape(2 * output_dims, 2) / 100 - 0.5
    x = mx.arange(3 * input_dims, dtype=mx.float32).reshape(3, input_dims) / 10 - 1
    # The fused reference computes [gate, up] before SiLU(gate) * up.
    fused_weight = mx.concatenate([mlp.gate_layer.weight, mlp.proj.weight], axis=0) + 0.7 * (up @ down)
    expected_gate, expected_proj = mx.split(x @ fused_weight.T, 2, axis=-1)
    expected = mlp.out(nn.silu(expected_gate) * expected_proj)
    mx.eval(expected_gate, expected_proj, expected)
    adapter = tmp_path / "adapter.safetensors"
    source = f"{prefix}.transformer_blocks.0.img_mlp.gate_up"
    mx.save_safetensors(str(adapter), {f"{source}.lora_A.weight": down, f"{source}.lora_B.weight": up})

    LoRALoader.load_and_apply_lora(Qwen21LoRAMapping.get_mapping(), model, [str(adapter)], [0.7], bake_lora=bake_lora)

    assert isinstance(mlp.gate_layer, LoRALinear) is (not bake_lora)
    assert isinstance(mlp.proj, LoRALinear) is (not bake_lora)
    assert mx.allclose(mlp.gate_layer(x), expected_gate, atol=1e-6).item()
    assert mx.allclose(mlp.proj(x), expected_proj, atol=1e-6).item()
    assert mx.allclose(mlp(x), expected, atol=1e-6).item()


@pytest.mark.fast
@pytest.mark.parametrize(
    "magnitude_key",
    ["dora_scale", "lora_magnitude_vector", "lora_magnitude_vector.weight", "lora_magnitude_vector.default.weight"],
)
@pytest.mark.parametrize(
    "down,up",
    [
        ("lora_A", "lora_B"),
        ("lora_A.weight", "lora_B.weight"),
        ("lora_down.weight", "lora_up.weight"),
        ("lora.down", "lora.up"),
    ],
)
def test_qwen21_dora_rejected_before_applying_weights(tmp_path, magnitude_key, down, up):
    model = nn.Module()
    model.transformer_blocks = [Qwen21TransformerBlock(dim=8, num_attention_heads=2, attention_head_dim=4)]
    attn = model.transformer_blocks[0].attn
    adapter = tmp_path / "dora.safetensors"
    prefix = "diffusion_model.transformer_blocks.0.attn"
    mx.save_safetensors(
        str(adapter),
        {
            f"{prefix}.to_k.{down}": mx.ones((2, 8)),
            f"{prefix}.to_k.{up}": mx.ones((8, 2)),
            f"{prefix}.to_q.{down}": mx.ones((2, 8)),
            f"{prefix}.to_q.{up}": mx.ones((8, 2)),
            f"{prefix}.to_q.{magnitude_key}": mx.ones((8, 1)),
        },
    )

    with pytest.raises(ValueError, match="DoRA with LoRA matrices is not supported"):
        LoRALoader.load_and_apply_lora(Qwen21LoRAMapping.get_mapping(), model, [str(adapter)], [1.0], bake_lora=False)

    assert isinstance(attn.to_k, nn.Linear)
    assert isinstance(attn.to_q, nn.Linear)


@pytest.mark.fast
@pytest.mark.parametrize("flags", [[], ["--no-bake-lora"]])
@pytest.mark.parametrize("cli", [qwen21_generate, qwen21_edit_generate])
def test_qwen21_cli_accepts_lora(monkeypatch, tmp_path, flags, cli):
    adapter = tmp_path / "a.safetensors"
    adapter.touch()
    monkeypatch.setattr(
        sys,
        "argv",
        ["mflux-generate-qwen-2.1", "--prompt", "test", "--lora", str(adapter), "0.9", *flags],
    )
    args = cli.build_parser().parse_args()
    assert args.lora_paths == [str(adapter)]
    assert args.lora_scales == [0.9]
    assert args.bake_lora is (not flags)


@pytest.mark.fast
@pytest.mark.parametrize(
    "cli,model_name", [(qwen21_generate, "QwenImage21"), (qwen21_edit_generate, "QwenImage21Edit")]
)
def test_qwen21_cli_warns_for_lora_style(monkeypatch, cli, model_name):
    monkeypatch.setattr(sys, "argv", ["prog", "--prompt", "test", "--lora-style", "storyboard"])
    monkeypatch.setattr(cli, model_name, Mock(side_effect=RuntimeError("stop before loading weights")))
    with (
        pytest.warns(UserWarning, match="--lora-style is ignored"),
        pytest.raises(RuntimeError, match="stop before loading weights"),
    ):
        cli.main()


@pytest.mark.fast
@pytest.mark.parametrize(
    "name,cli",
    [("mflux-generate-qwen-2.1", qwen21_generate), ("mflux-generate-qwen-2.1-edit", qwen21_edit_generate)],
)
def test_qwen21_capabilities_report_lora_style_as_ignored(name, cli):
    command = describe_command(name, cli.__name__)
    options = {option["flag"]: option for option in command["options"]}
    assert command["traits"]["lora"] is True
    assert options["--lora"]["status"] == "honored"
    assert options["--lora-style"]["status"] == "ignored"


@pytest.mark.fast
def test_edit_cli_forwards_adapter_and_baking_policy(monkeypatch, tmp_path):
    adapter = tmp_path / "adapter.safetensors"
    adapter.touch()
    monkeypatch.setattr(sys, "argv", ["prog", "--prompt", "test", "--lora", str(adapter), "0.4", "--no-bake-lora"])
    constructor = Mock(side_effect=RuntimeError("stop before loading weights"))
    monkeypatch.setattr(qwen21_edit_generate, "QwenImage21Edit", constructor)
    with pytest.raises(RuntimeError, match="stop before loading weights"):
        qwen21_edit_generate.main()
    assert constructor.call_args.kwargs["lora_paths"] == [str(adapter)]
    assert constructor.call_args.kwargs["lora_scales"] == [0.4]
    assert constructor.call_args.kwargs["bake_lora"] is False


@pytest.mark.fast
def test_edit_metadata_restores_lora_with_explicit_override(monkeypatch, tmp_path):
    adapters = [tmp_path / "old.safetensors", tmp_path / "new.safetensors"]
    for adapter in adapters:
        adapter.touch()
    sidecar = tmp_path / "image.json"
    sidecar.write_text(json.dumps({"prompt": "test", "lora_paths": [str(adapters[0])], "lora_scales": [0.3]}))
    monkeypatch.setattr(sys, "argv", ["prog", "--config-from-metadata", str(sidecar)])
    args = qwen21_edit_generate.build_parser().parse_args()
    assert args.lora_paths == [str(adapters[0])]
    assert args.lora_scales == [0.3]
    monkeypatch.setattr(
        sys, "argv", ["prog", "--config-from-metadata", str(sidecar), "--lora", str(adapters[1]), "0.8"]
    )
    args = qwen21_edit_generate.build_parser().parse_args()
    assert args.lora_paths == [str(adapters[1])]
    assert args.lora_scales == [0.8]


def _block_model():
    model = nn.Module()
    model.transformer_blocks = [Qwen21TransformerBlock(dim=8, num_attention_heads=2, attention_head_dim=4)]
    return model


def _check_lora(tmp_path, model, path, keys, scale, factor):
    # keys maps each tensor name in the file to "down", "up" or a value; factor is the expected update size.
    linear = LoRALoader._get_target_module(model, path)
    output_dims, input_dims = linear.weight.shape
    down = mx.arange(2 * input_dims, dtype=mx.float32).reshape(2, input_dims) / 1000
    up = mx.arange(output_dims * 2, dtype=mx.float32).reshape(output_dims, 2) / 1000
    x = mx.ones((1, input_dims))
    expected = linear(x) + factor * (x @ down.T @ up.T)
    mx.eval(expected)
    adapter = tmp_path / "adapter.safetensors"
    tensors = {"down": down, "up": up}
    mx.save_safetensors(str(adapter), {k: tensors.get(v, v) for k, v in keys.items()})

    LoRALoader.load_and_apply_lora(Qwen21LoRAMapping.get_mapping(), model, [str(adapter)], [scale])

    assert mx.allclose(LoRALoader._get_target_module(model, path)(x), expected, atol=1e-4).item()


@pytest.mark.fast
@pytest.mark.parametrize(
    "prefix,down_name,up_name",
    [
        ("", "lora_A.weight", "lora_B.weight"),  # Comfy-Org 5/8-step
        ("diffusion_model.", "lora_down.weight", "lora_up.weight"),  # Comfy-Org avg_rank_178
        ("", "lora_down", "lora_up"),  # Fun-Acc
        ("transformer.", "lora.down.weight", "lora.up.weight"),  # lightx2v Lightning
    ],
)
def test_qwen21_lora_accepts_comfyui_spellings(tmp_path, prefix, down_name, up_name):
    path = "transformer_blocks.0.attn.to_q"
    keys = {f"{prefix}{path}.{down_name}": "down", f"{prefix}{path}.{up_name}": "up"}
    _check_lora(tmp_path, _block_model(), path, keys, scale=0.7, factor=0.7)


@pytest.mark.fast
def test_qwen21_lora_applies_per_layer_alpha(tmp_path):
    # rank 2, alpha 4: the update is applied at twice its stored size
    path = "transformer_blocks.0.attn.to_k"
    keys = {f"{path}.lora_A.weight": "down", f"{path}.lora_B.weight": "up", f"{path}.alpha": mx.array(4.0)}
    _check_lora(tmp_path, _block_model(), path, keys, scale=1.0, factor=2.0)


@pytest.mark.fast
def test_qwen21_lora_gate_up_alpha_scales_both_halves(tmp_path):
    model = _block_model()
    mlp = model.transformer_blocks[0].img_mlp
    output_dims, input_dims = mlp.gate_layer.weight.shape
    down = mx.arange(2 * input_dims, dtype=mx.float32).reshape(2, input_dims) / 100 - 0.1
    up = mx.arange(2 * output_dims * 2, dtype=mx.float32).reshape(2 * output_dims, 2) / 100 - 0.5
    x = mx.arange(3 * input_dims, dtype=mx.float32).reshape(3, input_dims) / 10 - 1
    delta = 3.0 * (up @ down)
    expected_gate = x @ (mlp.gate_layer.weight + delta[:output_dims]).T
    expected_proj = x @ (mlp.proj.weight + delta[output_dims:]).T
    mx.eval(expected_gate, expected_proj)
    adapter = tmp_path / "adapter.safetensors"
    source = "transformer_blocks.0.img_mlp.gate_up"
    mx.save_safetensors(
        str(adapter),
        {f"{source}.lora_A.weight": down, f"{source}.lora_B.weight": up, f"{source}.alpha": mx.array(6.0)},
    )

    LoRALoader.load_and_apply_lora(Qwen21LoRAMapping.get_mapping(), model, [str(adapter)], [1.0])

    assert mx.allclose(mlp.gate_layer(x), expected_gate, atol=1e-5).item()
    assert mx.allclose(mlp.proj(x), expected_proj, atol=1e-5).item()


@pytest.mark.fast
@pytest.mark.parametrize("name", ["img_in", "proj_out", "norm_out.linear", "txt_in.in_layer", "txt_in.out_layer"])
def test_qwen21_lora_updates_input_and_output_layers(tmp_path, name):
    model = Qwen21Transformer(num_layers=1, num_attention_heads=2, attention_head_dim=4, context_in_dim=16)
    keys = {f"diffusion_model.{name}.lora_A": "down", f"diffusion_model.{name}.lora_B": "up"}
    _check_lora(tmp_path, model, name, keys, scale=0.5, factor=0.5)


@pytest.mark.fast
def test_qwen21_lora_applies_norm_weight_diff(tmp_path):
    model = _block_model()
    norm = model.transformer_blocks[0].attn.norm_q
    diff = mx.arange(norm.weight.shape[0], dtype=mx.float32)
    expected = norm.weight + 0.5 * diff
    mx.eval(expected)
    adapter = tmp_path / "adapter.safetensors"
    mx.save_safetensors(str(adapter), {"diffusion_model.transformer_blocks.0.attn.norm_q.diff": diff})

    LoRALoader.load_and_apply_lora(Qwen21LoRAMapping.get_mapping(), model, [str(adapter)], [0.5])

    assert mx.allclose(model.transformer_blocks[0].attn.norm_q.weight, expected, atol=1e-6).item()
