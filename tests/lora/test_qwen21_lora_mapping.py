import sys
from unittest.mock import Mock

import mlx.core as mx
import pytest
from mlx import nn

from mflux.cli.capabilities import describe_command
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.qwen21.cli import qwen21_generate
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer_block import Qwen21TransformerBlock
from mflux.models.qwen21.weights.qwen21_lora_mapping import Qwen21LoRAMapping


@pytest.mark.fast
@pytest.mark.parametrize("bake_lora", [False, True])
@pytest.mark.parametrize(
    "name",
    ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "img_mlp.proj", "img_mlp.gate_layer", "img_mlp.out"],
)
def test_qwen21_peft_updates_target_layer(tmp_path, name, bake_lora):
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
    mx.save_safetensors(str(adapter), {f"{path}.lora_A.default.weight": down, f"{path}.lora_B.default.weight": up})

    paths, scales = LoRALoader.load_and_apply_lora(
        Qwen21LoRAMapping.get_mapping(), model, [str(adapter)], [0.7], bake_lora=bake_lora
    )

    actual = LoRALoader._get_target_module(model, path)
    assert isinstance(actual, LoRALinear) is (not bake_lora)
    assert mx.allclose(actual(x), expected, atol=1e-6).item()
    assert paths == [str(adapter)]
    assert scales == [0.7]


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
