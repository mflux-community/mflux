from pathlib import Path

import mlx.core as mx
import pytest

from mflux.models.common.config import ModelConfig
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
from mflux.models.common.weights.mapping.weight_mapper import WeightMapper
from mflux.models.common.weights.mapping.weight_mapping import WeightTarget
from mflux.models.common_models.qwen3_5 import Qwen35TextModel
from mflux.models.fibo.model.fibo_transformer.transformer import FiboTransformer
from mflux.models.fibo.weights.fibo_weight_definition import FIBOWeightDefinition
from mflux.models.fibo2.model.fibo2_transformer.fibo2_transformer import Fibo2Transformer
from mflux.models.fibo2.weights.fibo2_weight_definition import Fibo2WeightDefinition
from mflux.models.fibo_vlm.weights.fibo_vlm_weight_definition import FIBOVLMWeightDefinition
from mflux.models.flux.model.flux_transformer.ada_layer_norm_continuous import AdaLayerNormContinuous
from mflux.models.flux.model.flux_transformer.transformer import Transformer
from mflux.models.flux.weights.flux_weight_definition import FluxControlnetWeightDefinition, FluxWeightDefinition
from mflux.models.flux.weights.flux_weight_mapping import FluxWeightMapping
from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer
from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition
from mflux.models.flux3.model.flux3_transformer.flux3_transformer import Flux3Transformer
from mflux.models.flux3.weights.flux3_weight_definition import (
    Flux3KleinBaseWeightDefinition,
    Flux3KleinWeightDefinition,
)
from mflux.models.iris.model.iris_transformer.iris_transformer import IrisTransformer
from mflux.models.iris.weights.iris_weight_definition import IrisTurboWeightDefinition, IrisWeightDefinition
from mflux.models.ming_image.weights.ming_image_weight_definition import MingImageWeightDefinition
from mflux.models.moss_image.model.moss_image_transformer.moss_image_transformer import MOSSImageTransformer
from mflux.models.moss_image.weights.moss_image_weight_definition import MOSSImageWeightDefinition
from mflux.models.qwen.model.qwen_transformer.qwen_transformer import QwenTransformer
from mflux.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition
from mflux.models.qwen21.weights.qwen21_weight_definition import Qwen21WeightDefinition
from mflux.models.seedvr2.weights.seedvr2_weight_definition import (
    SeedVR2WeightDefinition3B,
    SeedVR2WeightDefinition7B,
)
from mflux.models.z_image.weights.z_image_controlnet_weight_definition import ZImageControlnetWeightDefinition
from mflux.models.z_image.weights.z_image_weight_definition import ZImageWeightDefinition

# Tensor names read from the safetensors headers of each official checkpoint (no weights), after the component's
# own prefix filter and key transform, i.e. what the loader hands the check. Captured 2026-09-24 from:
#   flux1_schnell_vae        black-forest-labs/FLUX.1-schnell @ 741f7c3ce8b3  vae/
#   flux1_controlnet_canny   InstantX/FLUX.1-dev-Controlnet-Canny @ e7cee4b2afa3  diffusion_pytorch_model.safetensors
#   fibo_transformer         briaai/FIBO @ 7c7e00ca081f  transformer/
#   fibo_vlm_decoder         briaai/FIBO-vlm @ bd13da5e0b73
#   seedvr2_7b_transformer   numz/SeedVR2_comfyUI @ 09ced7102363  seedvr2_ema_7b_fp16.safetensors
#   qwen21_text_encoder      Qwen/Qwen-Image-2.1 @ 790c92633540  text_encoder/
#   qwen21_transformer       Qwen/Qwen-Image-2.1 @ 790c92633540  transformer/
#   qwen_image_vae           Qwen/Qwen-Image-2512 @ 25468b98e327  vae/
#   z_image_controlnet_union alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 @ 5155fc56d178
#                            Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors
#   ming_image_transformer   inclusionAI/Ming-Image-0.1-Design @ 208087ada148  transformer/
#   flux1_schnell_transformer  black-forest-labs/FLUX.1-schnell @ 741f7c3ce8b3  transformer/  (captured 2026-10-07)
#   qwen_image_transformer     Qwen/Qwen-Image-2512 @ 25468b98e327  transformer/  (captured 2026-10-07)
#   flux2_klein_4b_transformer black-forest-labs/FLUX.2-klein-4B @ e7b7dc27f91d  transformer/  (captured 2026-10-07)
#   flux3_klein_transformer    black-forest-labs/flux-3-image-klein @ 8454017f9eba  transformer/  (captured 2026-10-15)
#   flux3_klein_base_transformer  black-forest-labs/flux-3-image-klein-base @ 1469d27cfb9b  transformer/  (captured 2026-10-15)
#   flux3_klein_text_encoder   black-forest-labs/flux-3-image-klein @ 8454017f9eba  text_encoder/, after the
#                              component's key_transform (vision tower and layers >= 32 dropped)
#   flux3_klein_vae            black-forest-labs/flux-3-image-klein @ 8454017f9eba  vae/
#   iris_3b_transformer       speridlabs/iris-3b @ adc05b909fb0  turbo/model.safetensors (the base model's root file has the same names)  (captured 2026-10-18)
#   iris_3b_text_encoder      Qwen/Qwen3-VL-4B-Instruct @ ebb281ec70b0  model-0000{1,2}-of-00002.safetensors, after the
#                             component's key_transform (vision tower dropped, all 36 layers kept)  (captured 2026-10-18)
#   moss_image_transformer    OpenMOSS-Team/MOSS-Image-SFT @ f972bd4b4b9e  transformer/, after the component's
#                             key_transform (rope.bands dropped)  (captured 2026-10-21; the transformer of
#                             OpenMOSS-Team/MOSS-Image-Base @ 85975793d8b3 has the same names, checked 2026-10-26)
#   moss_image_text_encoder   OpenMOSS-Team/MOSS-Image-SFT @ f972bd4b4b9e  text_encoder/, after the component's
#                             prefix filter (vision tower dropped)  (captured 2026-10-21)
#   fibo2_transformer         briaai/Fibo-2.0-dev @ fb2b781f9e9e  transformer/  (captured 2026-10-23)
#   fibo2_vae                 briaai/Fibo-2.0-dev @ fb2b781f9e9e  vae/  (captured 2026-10-23)
# To regenerate one, take the keys of mx.load(<file>) (lazy, reads no weights) or of the shard index's weight_map;
# only fibo_vlm_decoder is then filtered, to the "model.language_model" and "lm_head" prefixes.
# A required mapping entry these names cannot satisfy would reject the official checkpoint at load time.


class _OfficialNames:
    DIR = Path(__file__).parent.parent / "resources" / "checkpoint_keys"

    @staticmethod
    def weights(fixture: str) -> dict[str, mx.array]:
        names = (_OfficialNames.DIR / f"{fixture}.txt").read_text().split()
        return {name: mx.zeros((1,)) for name in names}


class _Component:
    @staticmethod
    def of(definition, name: str) -> ComponentDefinition:
        return next(component for component in definition.get_components() if component.name == name)

    @staticmethod
    def missing(weights: dict[str, mx.array], component: ComponentDefinition) -> list[WeightTarget]:
        # The same mapping and block/layer counts WeightLoader._load_component passes to the check.
        return WeightMapper.missing_required_targets(
            weights, component.mapping_getter(), component.num_blocks, component.num_layers
        )


@pytest.mark.fast
@pytest.mark.parametrize(
    ("fixture", "component"),
    [
        ("flux1_schnell_vae", _Component.of(FluxWeightDefinition, "vae")),
        ("flux1_controlnet_canny", FluxControlnetWeightDefinition.get_controlnet_component()),
        ("fibo_transformer", _Component.of(FIBOWeightDefinition, "transformer")),
        ("fibo_vlm_decoder", _Component.of(FIBOVLMWeightDefinition, "decoder")),
        ("seedvr2_7b_transformer", _Component.of(SeedVR2WeightDefinition7B, "transformer")),
        ("qwen21_text_encoder", _Component.of(Qwen21WeightDefinition, "text_encoder")),
        ("qwen21_transformer", _Component.of(Qwen21WeightDefinition, "transformer")),
        ("qwen_image_vae", _Component.of(QwenWeightDefinition, "vae")),
        ("z_image_controlnet_union", ZImageControlnetWeightDefinition.get_controlnet_component()),
        ("ming_image_transformer", _Component.of(MingImageWeightDefinition, "transformer")),
        ("flux3_klein_transformer", _Component.of(Flux3KleinWeightDefinition, "transformer")),
        ("flux3_klein_base_transformer", _Component.of(Flux3KleinBaseWeightDefinition, "transformer")),
        ("flux3_klein_text_encoder", _Component.of(Flux3KleinWeightDefinition, "text_encoder")),
        ("flux3_klein_vae", _Component.of(Flux3KleinWeightDefinition, "vae")),
        ("iris_3b_transformer", IrisTurboWeightDefinition.source_transformer()),
        ("iris_3b_text_encoder", IrisWeightDefinition.source_text_encoder()),
        ("moss_image_transformer", _Component.of(MOSSImageWeightDefinition, "transformer")),
        ("moss_image_text_encoder", _Component.of(MOSSImageWeightDefinition, "text_encoder")),
        ("fibo2_transformer", _Component.of(Fibo2WeightDefinition, "transformer")),
        ("fibo2_vae", _Component.of(Fibo2WeightDefinition, "vae")),
    ],
)
def test_an_official_checkpoint_carries_every_required_weight(fixture, component):
    missing = _Component.missing(_OfficialNames.weights(fixture), component)

    assert [target.to_pattern for target in missing] == []


class _Placement:
    @staticmethod
    def unplaced(module, mapped, path: str = "") -> list[str]:
        # The names in a mapped weight tree that the module has no parameter for. Module.update(strict=False)
        # skips them without a word, which is how a model comes to run without part of its checkpoint.
        if isinstance(mapped, mx.array):
            return []
        items = mapped.items() if isinstance(mapped, dict) else enumerate(mapped)
        names = []
        for key, value in items:
            present = key in module if isinstance(module, dict) else isinstance(module, list) and key < len(module)
            if present:
                names += _Placement.unplaced(module[key], value, f"{path}{key}.")
            else:
                names.append(f"{path}{key}")
        return names


@pytest.mark.fast
@pytest.mark.parametrize(
    ("fixture", "component", "build"),
    [
        ("flux1_schnell_transformer", _Component.of(FluxWeightDefinition, "transformer"), lambda: Transformer(ModelConfig.schnell())),
        ("fibo_transformer", _Component.of(FIBOWeightDefinition, "transformer"), FiboTransformer),
        ("qwen_image_transformer", _Component.of(QwenWeightDefinition, "transformer"), QwenTransformer),
        ("flux2_klein_4b_transformer", _Component.of(Flux2KleinWeightDefinition, "transformer"), Flux2Transformer),
        ("flux3_klein_transformer", _Component.of(Flux3KleinWeightDefinition, "transformer"), Flux3Transformer),
        (
            "flux3_klein_base_transformer",
            _Component.of(Flux3KleinBaseWeightDefinition, "transformer"),
            lambda: Flux3Transformer(flow_map=False),
        ),
        ("iris_3b_transformer", IrisTurboWeightDefinition.source_transformer(), IrisTransformer),
        ("moss_image_transformer", _Component.of(MOSSImageWeightDefinition, "transformer"), MOSSImageTransformer),
        ("moss_image_text_encoder", _Component.of(MOSSImageWeightDefinition, "text_encoder"), Qwen35TextModel),
        ("fibo2_transformer", _Component.of(Fibo2WeightDefinition, "transformer"), Fibo2Transformer),
    ],
)  # fmt: skip
def test_a_transformer_has_a_place_for_every_weight_of_its_official_checkpoint(fixture, component, build):
    # FLUX.1, FIBO and Qwen-Image ran without norm_out.linear.bias from 0.16.0 to 0.21.0: the output norm they share
    # with FLUX.2 lost its bias for FLUX.2, whose checkpoint has none, and the loader dropped theirs silently.
    weights, mapping = _OfficialNames.weights(fixture), component.mapping_getter()
    mapped = WeightMapper.apply_mapping(weights, mapping, component.num_blocks, component.num_layers)

    # Both halves: a name the mapping leaves out never reaches the module either.
    assert WeightMapper.unmapped_names(weights, mapping, component.num_blocks, component.num_layers) == []
    assert _Placement.unplaced(build(), mapped) == []


@pytest.mark.fast
def test_the_shared_output_norm_applies_its_bias_and_starts_it_at_zero():
    # Zero by default, so a model saved without the bias loads as it was saved; the checkpoint's value when it has one.
    norm = AdaLayerNormContinuous(4, 4)
    norm.linear.weight = mx.zeros((8, 4))
    x, embedding = mx.ones((1, 2, 4)) * mx.array([1.0, 2.0, 3.0, 4.0]), mx.ones((1, 4))
    assert mx.array_equal(norm.linear.bias, mx.zeros((8,))).item()
    # In the model's precision: a float32 zero would promote a bf16 stream, and a model saved without the bias
    # would stop giving the pixels it gave.
    assert norm.linear.bias.dtype == ModelConfig.precision
    unbiased = norm(x, embedding)

    norm.linear.bias = mx.array([1.0] * 4 + [0.5] * 4)  # scale 1 and shift 0.5

    assert mx.allclose(norm(x, embedding), unbiased * 2 + 0.5).item()
    # FLUX.2 checkpoints have no bias for this layer, so its transformer does not grow one.
    assert "bias" not in Flux2Transformer().norm_out.linear


@pytest.mark.fast
def test_the_flux1_transformer_still_requires_its_output_head():
    # The ControlNet relaxation must stay inside the ControlNet mapping; FLUX.1 itself always has a head.
    required = {target.to_pattern for target in FluxWeightMapping.get_transformer_mapping() if target.required}

    assert {"proj_out.weight", "norm_out.linear.weight"} <= required


@pytest.mark.fast
def test_seedvr2_3b_still_requires_its_output_ada():
    # 3B builds and applies vid_out_norm, out_shift and out_scale; only 7B (use_output_ada=False) runs without.
    component = _Component.of(SeedVR2WeightDefinition3B, "transformer")
    required = {target.to_pattern for target in component.mapping_getter() if target.required}

    assert {"vid_out_norm.weight", "out_shift", "out_scale"} <= required


@pytest.mark.fast
def test_the_issue_748_text_encoder_layout_misses_every_required_weight():
    # mlx-community/Qwen-Image-2.1-MLX-4bit names the text encoder language_model.model.* where the official
    # checkpoint uses model.language_model.*; mflux used to load it with random weights and no error.
    component = _Component.of(Qwen21WeightDefinition, "text_encoder")
    official = _OfficialNames.weights("qwen21_text_encoder")
    swapped = {name.replace("model.language_model.", "language_model.model.", 1): v for name, v in official.items()}
    required = [target for target in component.mapping_getter() if target.required]

    assert len(_Component.missing(swapped, component)) == len(required)


@pytest.mark.fast
def test_the_issue_748_transformer_rename_is_caught_on_its_own():
    # The same checkpoint stores the shared modulation layer as modulation.0.* instead of modulation.1.*. Every other
    # transformer weight matches, so this one renamed layer is the only thing the check may report.
    component = _Component.of(Qwen21WeightDefinition, "transformer")
    renamed = {
        name.replace("modulation.1.", "modulation.0.", 1): v
        for name, v in _OfficialNames.weights("qwen21_transformer").items()
    }

    missing = _Component.missing(renamed, component)

    assert [target.from_pattern for target in missing] == [["modulation.1.weight"]]


@pytest.mark.fast
def test_the_z_image_transformer_still_requires_its_pad_tokens():
    # Ming-Image reuses this name list without pad tokens; Z-Image itself builds and uses them.
    mapping = ZImageWeightDefinition.get_components()
    transformer = next(component for component in mapping if component.name == "transformer")
    required = {target.to_pattern for target in transformer.mapping_getter() if target.required}

    assert {"x_pad_token", "cap_pad_token"} <= required


@pytest.mark.fast
def test_the_saved_layout_controlnet_component_uses_the_controlnet_list():
    component = _Component.of(FluxControlnetWeightDefinition, "transformer_controlnet")

    assert _Component.missing(_OfficialNames.weights("flux1_controlnet_canny"), component) == []


@pytest.mark.fast
@pytest.mark.parametrize(
    ("removed", "reported"),
    [
        ("single_transformer_blocks.0.proj_out.weight", "single_transformer_blocks.{block}.proj_out.weight"),
        ("controlnet_single_blocks.0.bias", "controlnet_single_blocks.{block}.bias"),
    ],
)
def test_a_controlnet_that_has_single_blocks_needs_each_of_their_weights(removed, reported):
    # Canny ships no single blocks; give it a complete single block 0 and its head, then take one weight away.
    component = FluxControlnetWeightDefinition.get_controlnet_component()
    weights = _OfficialNames.weights("flux1_controlnet_canny")
    for target in component.mapping_getter():
        for pattern in target.from_pattern:
            if pattern.startswith(("single_transformer_blocks.{block}.", "controlnet_single_blocks.{block}.")):
                weights[pattern.replace("{block}", "0")] = mx.zeros((1,))
    assert _Component.missing(weights, component) == []

    del weights[removed]

    assert [target.to_pattern for target in _Component.missing(weights, component)] == [reported]


@pytest.mark.fast
def test_flux3_distilled_needs_the_target_time_embedding_and_base_has_none():
    # The Base checkpoint is the distilled layout minus target_timestep_embedder; loading it with the distilled
    # mapping must be reported, not silently run on a randomly initialised embedder.
    base_weights = _OfficialNames.weights("flux3_klein_base_transformer")

    missing = _Component.missing(base_weights, _Component.of(Flux3KleinWeightDefinition, "transformer"))

    assert sorted(target.to_pattern for target in missing) == [
        "target_timestep_embedder.linear_1.weight",
        "target_timestep_embedder.linear_2.weight",
    ]
    assert Flux3Transformer(flow_map=False).target_timestep_embedder is None
