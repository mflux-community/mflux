import json
import math
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn

from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.lora.mapping.lora_mapping import LoRATarget
from mflux.models.common.lora.mapping.peft_lora_metadata import PeftLoRAMetadata

pytestmark = pytest.mark.fast

# LoRA files written by diffusers' save_lora_weights from adapters PEFT added to a toy model, with the outputs PEFT
# computed for them: "alpha_pattern" is r=4, lora_alpha=2 and alpha_pattern {"to_k": 8} (scales 0.5, 2.0, 0.5),
# "rslora" is r=4, lora_alpha=4 with rsLoRA (scale 4 / sqrt(4) = 2.0)
FIXTURE = Path(__file__).parent.parent / "resources" / "lora" / "peft_lora_metadata.npz"
MODULES = ("blocks.0.to_q", "blocks.0.to_k", "proj_out")


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(8, 8, bias=False)
        self.to_k = nn.Linear(8, 8, bias=False)


class _Toy(nn.Module):
    def __init__(self, base: dict[str, mx.array]):
        super().__init__()
        self.blocks = [_Block()]
        self.proj_out = nn.Linear(8, 8, bias=False)
        self.blocks[0].to_q.weight = base["blocks.0.to_q"]
        self.blocks[0].to_k.weight = base["blocks.0.to_k"]
        self.proj_out.weight = base["proj_out"]

    def module(self, path: str) -> nn.Module:
        return self.proj_out if path == "proj_out" else getattr(self.blocks[0], path.split(".")[-1])


class _Fixture:
    def __init__(self, name: str):
        data = np.load(FIXTURE)
        self.x = mx.array(data["x"])
        self.base = {m: mx.array(data[f"{name}/base/{m}"]) for m in MODULES}
        self.expected = {m: np.array(data[f"{name}/expected/{m}"]) for m in MODULES}
        self.scaling = {m: float(data[f"{name}/scaling/{m}"]) for m in MODULES}
        self.lora = {k.split("/lora/")[1]: mx.array(data[k]) for k in data.files if k.startswith(f"{name}/lora/")}
        self.metadata = str(data[f"{name}/metadata"])

    def write(self, path: Path, with_metadata: bool = True, extra: dict | None = None) -> str:
        metadata = {"format": "pt", "lora_adapter_metadata": self.metadata} if with_metadata else {"format": "pt"}
        mx.save_safetensors(str(path), {**self.lora, **(extra or {})}, metadata=metadata)
        return str(path)

    @staticmethod
    def mapping() -> list[LoRATarget]:
        return [
            LoRATarget(
                model_path=path,
                possible_up_patterns=[f"transformer.{path}.lora_B.weight"],
                possible_down_patterns=[f"transformer.{path}.lora_A.weight"],
                possible_alpha_patterns=[f"transformer.{path}.alpha"],
            )
            for path in ("blocks.{block}.to_q", "blocks.{block}.to_k", "proj_out")
        ]

    def outputs(self, lora_file: str, bake: bool) -> dict[str, np.ndarray]:
        model = _Toy(self.base)
        LoRALoader.load_and_apply_lora(
            lora_mapping=_Fixture.mapping(),
            transformer=model,
            lora_paths=[lora_file],
            lora_scales=[1.0],
            bake_lora=bake,
        )
        return {m: np.array(model.module(m)(self.x)) for m in MODULES}

    def unscaled(self, module: str) -> np.ndarray:
        # The base output plus the LoRA update at scale 1, the strength used when no alpha is known
        down = np.array(self.lora[f"transformer.{module}.lora_A.weight"])
        up = np.array(self.lora[f"transformer.{module}.lora_B.weight"])
        x = np.array(self.x)
        return x @ np.array(self.base[module]).T + x @ down.T @ up.T


@pytest.mark.parametrize("name", ["alpha_pattern", "rslora"])
@pytest.mark.parametrize("bake", [False, True], ids=["runtime", "baked"])
def test_lora_saved_by_diffusers_matches_peft(tmp_path, name, bake):
    fixture = _Fixture(name)
    outputs = fixture.outputs(fixture.write(tmp_path / "lora.safetensors"), bake=bake)
    for module in MODULES:
        np.testing.assert_allclose(outputs[module], fixture.expected[module], rtol=1e-5, atol=1e-5)


def test_without_metadata_the_lora_loads_at_scale_one_as_before(tmp_path):
    fixture = _Fixture("alpha_pattern")
    outputs = fixture.outputs(fixture.write(tmp_path / "lora.safetensors", with_metadata=False), bake=False)
    for module in MODULES:
        np.testing.assert_allclose(outputs[module], fixture.unscaled(module), rtol=1e-5, atol=1e-5)


def test_an_alpha_tensor_in_the_file_wins_over_the_metadata(tmp_path):
    # Rank 4 and an .alpha of 4 for to_q: scale 1, whatever the metadata says
    fixture = _Fixture("alpha_pattern")
    extra = {"transformer.blocks.0.to_q.alpha": mx.array(4.0)}
    outputs = fixture.outputs(fixture.write(tmp_path / "lora.safetensors", extra=extra), bake=False)
    np.testing.assert_allclose(outputs["blocks.0.to_q"], fixture.unscaled("blocks.0.to_q"), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(outputs["proj_out"], fixture.expected["proj_out"], rtol=1e-5, atol=1e-5)


class TestPeftLoRAMetadata:
    def test_no_or_unreadable_metadata_gives_no_configs(self):
        assert PeftLoRAMetadata.configs(None) == {}
        assert PeftLoRAMetadata.configs({"format": "pt"}) == {}
        assert PeftLoRAMetadata.configs({"lora_adapter_metadata": "not json"}) == {}
        assert PeftLoRAMetadata.configs({"lora_adapter_metadata": json.dumps({"transformer.r": 4})}) == {}

    def test_pipeline_save_is_read_per_component(self):
        configs = PeftLoRAMetadata.configs(_metadata({"transformer.r": 16, "transformer.lora_alpha": 8}))
        assert PeftLoRAMetadata.alpha("transformer.layers.0.to_q.lora_A.weight", configs, rank=16) == 8.0

    def test_model_save_without_prefix(self):
        configs = PeftLoRAMetadata.configs(_metadata({"r": 16, "lora_alpha": 8}))
        assert PeftLoRAMetadata.alpha("layers.0.to_q.lora_A.weight", configs, rank=16) == 8.0

    def test_peft_adapter_name_in_the_key(self):
        configs = PeftLoRAMetadata.configs(_metadata({"transformer.r": 16, "transformer.lora_alpha": 8}))
        assert PeftLoRAMetadata.alpha("transformer.layers.0.to_q.lora_A.default.weight", configs, rank=16) == 8.0

    def test_alpha_pattern_matches_whole_module_name_parts(self):
        config = {"transformer.r": 4, "transformer.lora_alpha": 4, "transformer.alpha_pattern": {"to_q": 2}}
        configs = PeftLoRAMetadata.configs(_metadata(config))
        assert PeftLoRAMetadata.alpha("transformer.layers.0.to_q.lora_A.weight", configs, rank=4) == 2.0
        assert PeftLoRAMetadata.alpha("transformer.layers.0.add_to_q.lora_A.weight", configs, rank=4) == 4.0

    def test_rslora_divides_by_the_square_root_of_the_rank(self):
        config = {"transformer.r": 16, "transformer.lora_alpha": 8, "transformer.use_rslora": True}
        configs = PeftLoRAMetadata.configs(_metadata(config))
        alpha = PeftLoRAMetadata.alpha("transformer.layers.0.to_q.lora_A.weight", configs, rank=16)
        assert alpha / 16 == pytest.approx(8 / math.sqrt(16))

    def test_keys_of_another_component_are_left_alone(self):
        config = {"transformer.r": 4, "transformer.lora_alpha": 2, "text_encoder.r": 4, "text_encoder.lora_alpha": 8}
        configs = PeftLoRAMetadata.configs(_metadata(config))
        assert PeftLoRAMetadata.alpha("unet.down.0.lora_A.weight", configs, rank=4) is None
        assert PeftLoRAMetadata.alpha("text_encoder.layers.0.q.lora_A.weight", configs, rank=4) == 8.0

    def test_a_key_that_is_not_a_lora_matrix_has_no_alpha(self):
        configs = PeftLoRAMetadata.configs(_metadata({"transformer.r": 4, "transformer.lora_alpha": 2}))
        assert PeftLoRAMetadata.alpha("transformer.layers.0.to_q.alpha", configs, rank=4) is None


def _metadata(config: dict) -> dict[str, str]:
    return {"format": "pt", "lora_adapter_metadata": json.dumps(config)}
