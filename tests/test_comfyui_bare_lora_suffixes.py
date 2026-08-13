import pytest

from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.ernie_image.weights.ernie_lora_mapping import ErnieLoRAMapping
from mflux.models.flux.weights.flux_lora_mapping import FluxLoRAMapping
from mflux.models.flux2.weights.flux2_lora_mapping import Flux2LoRAMapping
from mflux.models.ideogram4.weights.ideogram4_lora_mapping import Ideogram4LoRAMapping
from mflux.models.krea2.weights.krea2_lora_mapping import Krea2LoRAMapping
from mflux.models.qwen.weights.qwen_lora_mapping import QwenLoRAMapping
from mflux.models.z_image.weights.z_image_lora_mapping import ZImageLoRAMapping

pytestmark = pytest.mark.fast

MAPPINGS = [
    FluxLoRAMapping,
    Flux2LoRAMapping,
    QwenLoRAMapping,
    ZImageLoRAMapping,
    ErnieLoRAMapping,
    Ideogram4LoRAMapping,
    Krea2LoRAMapping,
]


@pytest.mark.parametrize("mapping_cls", MAPPINGS, ids=lambda m: m.__name__)
def test_every_weight_pattern_also_matches_its_bare_form(mapping_cls):
    # ComfyUI-format adapters name the tensor itself "lora_B", not "lora_B.weight".
    # The matcher accepts the bare spelling for every ".weight" pattern centrally, so a
    # file in that format matches in every family, not only where someone hand-spelled
    # both forms. A key derived from the pattern itself is exactly what such a file holds.
    checked = 0
    for target in mapping_cls.get_mapping():
        for pattern in list(target.possible_up_patterns) + list(target.possible_down_patterns):
            if not pattern.endswith(".weight"):
                continue
            bare_key = pattern[: -len(".weight")].replace("{block}", "0")
            assert LoRALoader._match_pattern(bare_key, pattern) is not None, (
                f"{mapping_cls.__name__}: bare form of {pattern!r} does not match"
            )
            checked += 1
    assert checked > 0, f"{mapping_cls.__name__} exposes no .weight patterns to check"


def test_reported_comfyui_krea2_keys_match():
    # The exact key shapes from the field report: prefix diffusion_model., module paths
    # attn.wk / attn.gate / mlp.up, tensors named lora_A / lora_B with no .weight.
    reported = [
        "diffusion_model.blocks.0.attn.wk.lora_A",
        "diffusion_model.blocks.0.attn.wk.lora_B",
        "diffusion_model.blocks.0.attn.gate.lora_A",
        "diffusion_model.blocks.0.mlp.up.lora_B",
    ]
    all_patterns = [
        (pattern, direction)
        for target in Krea2LoRAMapping.get_mapping()
        for direction, patterns in (("up", target.possible_up_patterns), ("down", target.possible_down_patterns))
        for pattern in patterns
    ]
    for key in reported:
        assert any(LoRALoader._match_pattern(key, p) is not None for p, _ in all_patterns), (
            f"reported key {key!r} matches no krea2 pattern"
        )
