import sys
from unittest.mock import Mock

import mlx.core as mx
import pytest
from mlx import nn

from mflux.cli.capabilities import describe_command
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.qwen21.cli import qwen21_generate
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer_block import Qwen21TransformerBlock
from mflux.models.qwen21.pdd_loader import Qwen21PDDLoader
from mflux.models.qwen21.qwen21_pdd_scheduler import Qwen21PDDScheduler
from mflux.models.qwen21.weights.qwen21_lora_mapping import Qwen21LoRAMapping


@pytest.mark.fast
@pytest.mark.parametrize("bake_lora", [False, True])
@pytest.mark.parametrize("adapter_name", [".default", ""])
@pytest.mark.parametrize(
    "name",
    ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "img_mlp.proj", "img_mlp.gate_layer", "img_mlp.out"],
)
def test_qwen21_peft_updates_target_layer(tmp_path, name, adapter_name, bake_lora):
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
        str(adapter), {f"{path}.lora_A{adapter_name}.weight": down, f"{path}.lora_B{adapter_name}.weight": up}
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
def test_qwen21_peft_mapping_supports_transformer_prefixed_global_targets(tmp_path):
    targets = {target.model_path: target for target in Qwen21LoRAMapping.get_mapping()}

    assert (
        "transformer.transformer_blocks.{block}.attn.to_q.lora_A.default.weight"
        in targets["transformer_blocks.{block}.attn.to_q"].possible_down_patterns
    )
    assert (
        "transformer.transformer_blocks.{block}.attn.to_q.lora_A.weight"
        in targets["transformer_blocks.{block}.attn.to_q"].possible_down_patterns
    )
    assert (
        "transformer.time_text_embed.timestep_embedder.linear_1.lora_A.default.weight"
        in targets["time_text_embed.timestep_embedder.linear_1"].possible_down_patterns
    )
    assert (
        "transformer.time_text_embed.timestep_embedder.linear_2.lora_B.default.weight"
        in targets["time_text_embed.timestep_embedder.linear_2"].possible_up_patterns
    )
    assert "transformer.modulation.1.lora_B.default.weight" in targets["modulation.layers.1"].possible_up_patterns

    model = nn.Module()
    model.transformer_blocks = [Qwen21TransformerBlock(dim=8, num_attention_heads=2, attention_head_dim=4)]
    model.time_text_embed = nn.Module()
    model.time_text_embed.timestep_embedder = nn.Module()
    model.time_text_embed.timestep_embedder.linear_1 = nn.Linear(4, 4, bias=False)
    model.modulation = nn.Sequential(nn.SiLU(), nn.Linear(4, 16, bias=False))
    aliases = {
        "transformer.transformer_blocks.0.attn.to_q": "transformer_blocks.0.attn.to_q",
        "transformer.time_text_embed.timestep_embedder.linear_1": "time_text_embed.timestep_embedder.linear_1",
        "transformer.modulation.1": "modulation.layers.1",
    }
    weights = {}
    for source_path, target_path in aliases.items():
        linear = LoRALoader._get_target_module(model, target_path)
        output_dims, input_dims = linear.weight.shape
        weights[f"{source_path}.lora_A.default.weight"] = mx.ones((2, input_dims))
        weights[f"{source_path}.lora_B.default.weight"] = mx.ones((output_dims, 2))
    adapter = tmp_path / "prefixed-and-global.safetensors"
    mx.save_safetensors(str(adapter), weights)

    LoRALoader.load_and_apply_lora(Qwen21LoRAMapping.get_mapping(), model, [str(adapter)], bake_lora=False)

    for target_path in aliases.values():
        assert isinstance(LoRALoader._get_target_module(model, target_path), LoRALinear)


@pytest.mark.fast
def test_qwen21_pdd_loader_applies_prefused_output_heads(tmp_path):
    transformer = Qwen21Transformer(
        in_channels=4,
        out_channels=2,
        num_layers=1,
        attention_head_dim=4,
        num_attention_heads=2,
        context_in_dim=6,
    )
    adapter = tmp_path / "fun-acc.safetensors"
    mx.save_safetensors(
        str(adapter),
        {
            "img_in.lora_down": mx.ones((2, 4)),
            "img_in.lora_up": mx.ones((8, 2)),
            "proj_out.weight": mx.ones((4, 2, 8)),
        },
        {"format": Qwen21PDDLoader.FORMAT},
    )

    Qwen21PDDLoader.load_and_apply(transformer, str(adapter), scale=1.0, bake_lora=True)

    assert transformer.has_pdd
    assert transformer.pdd_proj_out_weights.shape == (4, 2, 8)
    assert mx.array_equal(transformer.pdd_sigmas, Qwen21PDDLoader.SIGMAS)
    assert "pdd_proj_out_weights" in transformer.parameters()


@pytest.mark.fast
def test_qwen21_pdd_scheduler_uses_fp32_state():
    scheduler = Qwen21PDDScheduler(mx.array([1.0, 0.5, 0.0]))
    result = scheduler.step(mx.ones((1,), dtype=mx.bfloat16), 0, mx.ones((1,), dtype=mx.bfloat16))
    assert result.dtype == mx.float32
    assert mx.allclose(result, mx.array([0.5], dtype=mx.float32)).item()


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
