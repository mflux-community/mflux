import sys
from unittest.mock import Mock

import mlx.core as mx
import pytest
from mlx import nn

from mflux.cli.capabilities import describe_command
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.qwen21.cli import qwen21_generate
from mflux.models.qwen21.model.qwen21_transformer.qwen21_time_text_embed import Qwen21TimeTextEmbed
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
@pytest.mark.parametrize("suffix", ["", ".weight"])
def test_qwen21_dora_rejected_before_applying_weights(tmp_path, magnitude_key, suffix):
    model = nn.Module()
    model.transformer_blocks = [Qwen21TransformerBlock(dim=8, num_attention_heads=2, attention_head_dim=4)]
    attn = model.transformer_blocks[0].attn
    adapter = tmp_path / "dora.safetensors"
    prefix = "diffusion_model.transformer_blocks.0.attn"
    mx.save_safetensors(
        str(adapter),
        {
            f"{prefix}.to_k.lora_A{suffix}": mx.ones((2, 8)),
            f"{prefix}.to_k.lora_B{suffix}": mx.ones((8, 2)),
            f"{prefix}.to_q.lora_A{suffix}": mx.ones((2, 8)),
            f"{prefix}.to_q.lora_B{suffix}": mx.ones((8, 2)),
            f"{prefix}.to_q.{magnitude_key}": mx.ones((8, 1)),
        },
    )

    with pytest.raises(ValueError, match="DoRA with LoRA matrices is not supported"):
        LoRALoader.load_and_apply_lora(Qwen21LoRAMapping.get_mapping(), model, [str(adapter)], [1.0], bake_lora=False)

    assert isinstance(attn.to_k, nn.Linear)
    assert isinstance(attn.to_q, nn.Linear)


@pytest.mark.fast
@pytest.mark.parametrize("flags", [[], ["--no-bake-lora"]])
def test_qwen21_cli_accepts_lora(monkeypatch, tmp_path, flags):
    adapter = tmp_path / "a.safetensors"
    adapter.touch()
    monkeypatch.setattr(
        sys,
        "argv",
        ["mflux-generate-qwen-2.1", "--prompt", "test", "--lora", str(adapter), "0.9", *flags],
    )
    args = qwen21_generate.build_parser().parse_args()
    assert args.lora_paths == [str(adapter)]
    assert args.lora_scales == [0.9]
    assert args.bake_lora is (not flags)


@pytest.mark.fast
def test_qwen21_cli_warns_for_lora_style(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--prompt", "test", "--lora-style", "storyboard"])
    monkeypatch.setattr(qwen21_generate, "QwenImage21", Mock(side_effect=RuntimeError("stop before loading weights")))
    with (
        pytest.warns(UserWarning, match="--lora-style is ignored"),
        pytest.raises(RuntimeError, match="stop before loading weights"),
    ):
        qwen21_generate.main()


@pytest.mark.fast
def test_qwen21_capabilities_report_lora_style_as_ignored():
    command = describe_command("mflux-generate-qwen-2.1", qwen21_generate.__name__)
    options = {option["flag"]: option for option in command["options"]}
    assert command["traits"]["lora"] is True
    assert options["--lora"]["status"] == "honored"
    assert options["--lora-style"]["status"] == "ignored"
