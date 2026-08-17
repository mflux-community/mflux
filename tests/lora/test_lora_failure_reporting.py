from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx import nn

from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.lora.mapping.lora_mapping import LoRATarget
from mflux.models.common.lora.mapping.lora_saver import LoRASaver
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.common.weights.saving.model_saver import ModelSaver


class _Fixtures:
    @staticmethod
    def projection_mapping(*names: str) -> list[LoRATarget]:
        return [
            LoRATarget(
                model_path=name,
                possible_up_patterns=[f"{name}.lora_B.weight"],
                possible_down_patterns=[f"{name}.lora_A.weight"],
            )
            for name in names
        ]

    @staticmethod
    def adapter(path, *names: str) -> str:
        mx.save_safetensors(
            str(path),
            {
                key: value
                for name in names
                for key, value in {
                    f"{name}.lora_A.weight": mx.ones((1, 4)),
                    f"{name}.lora_B.weight": mx.ones((2, 1)),
                }.items()
            },
        )
        return str(path)


class _Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 2, bias=False)
        self.proj.weight = mx.zeros((2, 4))


def test_loader_raises_on_missing_lora_file(tmp_path, capsys):
    # Resolution used to drop the path with a warning, which both generated from the base
    # model and shifted the remaining scales onto the wrong adapters.
    present = _Fixtures.adapter(tmp_path / "present.safetensors", "proj")

    with pytest.raises(FileNotFoundError, match="LoRA file not found"):
        LoRALoader.load_and_apply_lora(
            lora_mapping=_Fixtures.projection_mapping("proj"),
            transformer=_Transformer(),
            lora_paths=[str(tmp_path / "gone.safetensors"), present],
            lora_scales=[0.8, 0.5],
            bake_lora=False,
        )

    assert "applied successfully" not in capsys.readouterr().out


def test_apply_single_lora_raises_on_missing_lora_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="LoRA file not found"):
        LoRALoader._apply_single_lora(
            _Transformer(),
            str(tmp_path / "gone.safetensors"),
            scale=1.0,
            lora_mapping=_Fixtures.projection_mapping("proj"),
            role=None,
        )


def test_loader_raises_on_unreadable_lora_file(tmp_path):
    lora_path = tmp_path / "corrupt.safetensors"
    lora_path.write_bytes(b"not a safetensors file")

    with pytest.raises(ValueError, match="Failed to load LoRA file"):
        LoRALoader.load_and_apply_lora(
            lora_mapping=_Fixtures.projection_mapping("proj"),
            transformer=_Transformer(),
            lora_paths=[str(lora_path)],
            lora_scales=[1.0],
            bake_lora=False,
        )


def test_loader_raises_when_only_some_targets_apply(tmp_path, capsys):
    # The file names a layer this model does not have: applying the rest would leave a
    # model that is neither the base nor the adapter.
    lora_path = _Fixtures.adapter(tmp_path / "half.safetensors", "proj", "absent")

    with pytest.raises(ValueError, match=r"1 of the 2 layers named by half.safetensors .* \(absent\)"):
        LoRALoader.load_and_apply_lora(
            lora_mapping=_Fixtures.projection_mapping("proj", "absent"),
            transformer=_Transformer(),
            lora_paths=[lora_path],
            lora_scales=[1.0],
            bake_lora=False,
        )

    assert "applied successfully" not in capsys.readouterr().out


def test_saver_raises_on_shape_mismatch_and_names_the_layer():
    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.wq = nn.Linear(4, 2, bias=False)
            self.wq.weight = mx.zeros((2, 4))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = Attention()

    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = [Block()]

    transformer = Transformer()
    attention = transformer.blocks[0].attn
    lora = LoRALinear.from_linear(attention.wq, r=1)
    lora.lora_A = mx.ones((4, 1))
    lora.lora_B = mx.ones((1, 3))  # delta is (3, 4), the base weight is (2, 4)
    attention.wq = lora

    with pytest.raises(ValueError, match=r"LoRA shape mismatch at blocks.0.attn.wq: .*\(2, 4\) vs .*\(3, 4\)"):
        LoRASaver.bake_and_strip_lora(transformer)


def test_saver_raises_when_requantizing_the_merged_weight_fails(monkeypatch):
    linear = nn.Linear(64, 32, bias=False)
    linear.weight = mx.zeros((32, 64))
    quantized = linear.to_quantized(group_size=32, bits=8)

    def _boom(*args, **kwargs):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(LoRASaver, "_quantize_dense", _boom)

    with pytest.raises(RuntimeError, match="Failed to bake a LoRA into QuantizedLinear at blocks.0"):
        LoRASaver._bake_delta_into_linear(quantized, mx.zeros((32, 64)), path="blocks.0")


def test_model_saver_writes_nothing_when_a_lora_cannot_be_baked(tmp_path):
    class Tokenizer:
        @staticmethod
        def save_pretrained(path):
            raise AssertionError("nothing may be written before every component has baked")

    class Model:
        def __init__(self):
            self.vae = nn.Linear(4, 2, bias=False)
            self.transformer = _Transformer()
            self.tokenizers = {"tokenizer": SimpleNamespace(tokenizer=Tokenizer())}
            lora = LoRALinear.from_linear(self.transformer.proj, r=1)
            lora.lora_A = mx.ones((4, 1))
            lora.lora_B = mx.ones((1, 3))
            self.transformer.proj = lora

    class WeightDefinition:
        @staticmethod
        def get_tokenizers():
            return [TokenizerDefinition(name="tokenizer", hf_subdir="tokenizer")]

        @staticmethod
        def get_components():
            return [
                ComponentDefinition(name="vae", hf_subdir="vae"),
                ComponentDefinition(name="transformer", hf_subdir="transformer"),
            ]

    with pytest.raises(ValueError, match="LoRA shape mismatch at proj"):
        ModelSaver.save_model(Model(), bits=8, base_path=str(tmp_path), weight_definition=WeightDefinition)

    assert list(tmp_path.iterdir()) == []
