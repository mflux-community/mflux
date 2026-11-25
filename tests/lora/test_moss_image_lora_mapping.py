import inspect

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten

from mflux.cli.capabilities import describe_command
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.moss_image.cli import moss_image_generate
from mflux.models.moss_image.model.moss_image_transformer.moss_image_block import MOSSImageBlock
from mflux.models.moss_image.model.moss_image_transformer.moss_image_transformer import MOSSImageTransformer
from mflux.models.moss_image.variants.txt2img.moss_image import MOSSImage
from mflux.models.moss_image.weights.moss_image_lora_mapping import MOSSImageLoRAMapping

pytestmark = pytest.mark.fast

TOY = dict(dim=96, ffn_dim=160, num_heads=4, num_layers=3, text_dim=48, freq_dim=32, latent_channels=128)
RANK = 4


def _resolve(key: str) -> set[tuple[str, str]]:
    targets = set()
    for mapping in LoRALoader._build_pattern_mappings(MOSSImageLoRAMapping.get_mapping()):
        block = LoRALoader._match_pattern(key, mapping.source_pattern)
        if block is not None:
            targets.add((mapping.target_path.format(block=block), mapping.matrix_name))
    return targets


def _apply(transformer: nn.Module, path, weights: dict[str, mx.array], scale: float = 1.0, bake_lora: bool = False):
    mx.save_safetensors(str(path), weights)
    return LoRALoader.load_and_apply_lora(
        lora_mapping=MOSSImageLoRAMapping.get_mapping(),
        transformer=transformer,
        lora_paths=[str(path)],
        lora_scales=[scale],
        bake_lora=bake_lora,
    )


def _update(layer: LoRALinear) -> np.ndarray:
    return np.array(layer.scale * (layer.lora_A @ layer.lora_B).T)


@pytest.mark.parametrize(
    ("key", "target"),
    [
        # kohya, as lodestones/MOSS-Image-SFT-8-steps-dmd and darask0/MOSS-Image_4-step_adapter_lora write it.
        ("diffusion_model.layers.0.proj.q.lora_down.weight", ("layers.0.proj.q", "lora_A")),
        ("diffusion_model.layers.41.proj.ffn_up_gate.lora_up.weight", ("layers.41.proj.ffn_up_gate", "lora_B")),
        ("diffusion_model.layers.7.proj.o.alpha", ("layers.7.proj.o", "alpha")),
        # PEFT matrices without a .weight suffix, as chinoll/Moss-Image-Edit writes them.
        ("layers.12.proj.ffn_down.lora_A", ("layers.12.proj.ffn_down", "lora_A")),
        ("layers.3.proj.v.lora_B", ("layers.3.proj.v", "lora_B")),
        # The diffusers and PEFT save_pretrained prefixes.
        ("transformer.layers.5.proj.k.lora_A.weight", ("layers.5.proj.k", "lora_A")),
        ("base_model.model.layers.2.proj.q.lora_B.default.weight", ("layers.2.proj.q", "lora_B")),
    ],
)
def test_published_key_formats_reach_their_module(key, target):
    assert _resolve(key) == {target}


def test_the_targets_are_the_projections_of_a_layer():
    block = MOSSImageBlock(dim=96, ffn_dim=160, heads=4)
    assert sorted(block.proj) == sorted(MOSSImageLoRAMapping.PROJECTIONS)
    assert _resolve("text_embedder.blocks.0.qkv.lora_A.weight") == set()


def test_kohya_and_peft_files_apply_the_same_update(tmp_path):
    kohya_model = MOSSImageTransformer(**TOY)
    peft_model = MOSSImageTransformer(**TOY)
    peft_model.update(tree_unflatten(tree_flatten(kohya_model.parameters())))
    rng = np.random.default_rng(0)
    kohya, peft, expected = {}, {}, {}
    for layer in range(TOY["num_layers"]):
        for name in MOSSImageLoRAMapping.PROJECTIONS:
            out_dim, in_dim = kohya_model.layers[layer].proj[name].weight.shape
            down = rng.standard_normal((RANK, in_dim)).astype(np.float32)
            up = rng.standard_normal((out_dim, RANK)).astype(np.float32)
            # alpha = 2 * rank doubles the update. A PEFT file has no alpha, so its up matrix carries the factor.
            kohya[f"diffusion_model.layers.{layer}.proj.{name}.lora_down.weight"] = mx.array(down)
            kohya[f"diffusion_model.layers.{layer}.proj.{name}.lora_up.weight"] = mx.array(up)
            kohya[f"diffusion_model.layers.{layer}.proj.{name}.alpha"] = mx.array(2.0 * RANK)
            peft[f"layers.{layer}.proj.{name}.lora_A"] = mx.array(down)
            peft[f"layers.{layer}.proj.{name}.lora_B"] = mx.array(2.0 * up)
            expected[(layer, name)] = 0.5 * 2.0 * up @ down

    _apply(kohya_model, tmp_path / "kohya.safetensors", kohya, scale=0.5)
    _apply(peft_model, tmp_path / "peft.safetensors", peft, scale=0.5)

    for (layer, name), update in expected.items():
        for model in (kohya_model, peft_model):
            adapter = model.layers[layer].proj[name]
            assert isinstance(adapter, LoRALinear)
            np.testing.assert_allclose(_update(adapter), update, rtol=1e-4, atol=1e-4)


class _WideLayer(nn.Module):
    # One MOSS layer path (layers.0.proj.q) with 1024 inputs, about a third of the real width.
    def __init__(self):
        super().__init__()
        mx.random.seed(0)
        linear = nn.Linear(1024, 64, bias=False)
        linear.weight = linear.weight.astype(mx.bfloat16)
        self.layers = [nn.Module()]
        self.layers[0].proj = {"q": linear}


def test_an_update_under_half_a_bfloat16_step_keeps_its_effect_in_a_runtime_layer(tmp_path):
    # The step-distilled MOSS LoRAs: about 90% of the entries of their update are under half a bfloat16 step
    # of the weight they land on, and a merge into the weights rounds those away. The update lines up with the
    # activations, so its effect on the layer output is far above the rounding step of the output. A runtime
    # layer, the MOSS default, applies that effect.
    runtime, merged = _WideLayer(), _WideLayer()
    base = runtime.layers[0].proj["q"]
    rng = np.random.default_rng(1)
    direction = np.sign(rng.standard_normal(1024)).astype(np.float32)
    down = direction[None, :] * 1e-5
    up = rng.standard_normal((64, 1)).astype(np.float32)
    weights = {
        "diffusion_model.layers.0.proj.q.lora_down.weight": mx.array(down),
        "diffusion_model.layers.0.proj.q.lora_up.weight": mx.array(up),
    }
    update = up @ down
    stored = np.abs(np.array(base.weight.astype(mx.float32)))
    half_step = 2.0 ** (np.floor(np.log2(stored)) - 8)
    assert np.mean(np.abs(update) < half_step) > 0.85

    _apply(runtime, tmp_path / "runtime.safetensors", weights, bake_lora=MOSSImage.DEFAULT_BAKE_LORA)
    _apply(merged, tmp_path / "merged.safetensors", weights, bake_lora=True)

    noise = 0.1 * rng.standard_normal((1, 32, 1024)).astype(np.float32)
    x = mx.array(direction[None, None, :] + noise).astype(mx.bfloat16)
    before = np.array(base(x).astype(mx.float32))
    effect = np.array(x.astype(mx.float32)) @ update.T

    def share(layer: nn.Module) -> float:
        moved = np.array(layer(x).astype(mx.float32)) - before
        return float(np.sum(moved * effect) / np.sum(effect * effect))

    assert isinstance(runtime.layers[0].proj["q"], LoRALinear)
    assert share(runtime.layers[0].proj["q"]) > 0.95
    assert share(merged.layers[0].proj["q"]) < 0.5


def test_lora_arguments_come_after_the_existing_positional_ones():
    # MOSSImage(quantize, model_path, model_config) was the signature before LoRA support.
    names = list(inspect.signature(MOSSImage.__init__).parameters)[1:]
    assert names == ["quantize", "model_path", "model_config", "lora_paths", "lora_scales", "bake_lora", "ssd_stream"]


def test_capabilities_report_lora_as_honored_and_unbaked_by_default():
    command = describe_command("mflux-generate-moss-image", moss_image_generate.__name__)
    options = {option["flag"]: option for option in command["options"]}
    assert command["traits"]["lora"] is True
    assert options["--lora"]["status"] == "honored"
    assert options["--lora-style"]["status"] == "ignored"
    assert options["--bake-lora"]["parser_default"] is False
