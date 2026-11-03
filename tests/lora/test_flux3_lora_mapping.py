import inspect

import mlx.core as mx
import numpy as np
import pytest

from mflux.cli.capabilities import describe_command
from mflux.cli.parser.parsers import lora_init_kwargs_from_args
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.flux3.cli import flux3_edit_generate, flux3_generate
from mflux.models.flux3.model.flux3_transformer.flux3_transformer import Flux3Transformer
from mflux.models.flux3.variants import Flux3Klein, Flux3KleinEdit
from mflux.models.flux3.weights.flux3_lora_mapping import Flux3LoRAMapping

pytestmark = pytest.mark.fast

RANK = 2

# One key per layer kind, as ComfyUI / ai-toolkit export them (ausboss/FLUX-3-Image-Klein-Outpaint-LoRA uses the
# block names), with the mflux module each one must reach.
BFL_KEYS = {
    "diffusion_model.content_mode_blocks.image.0.q_proj": "mode_blocks.image.0.attn.to_q",
    "diffusion_model.content_mode_blocks.image_cond.1.k_proj": "mode_blocks.image_cond.1.attn.to_k",
    "diffusion_model.txt_mode_blocks.1.v_proj": "txt_mode_blocks.1.attn.to_v",
    "diffusion_model.single_blocks.1.attn_out": "single_transformer_blocks.1.attn.to_out",
    "diffusion_model.content_mode_blocks.image.1.mlp_in": "mode_blocks.image.1.ff.linear_in",
    "diffusion_model.single_blocks.0.mlp_out": "single_transformer_blocks.0.ff.linear_out",
    "diffusion_model.emb_in.image_cond": "x_embedders.image_cond",
    "diffusion_model.txt_in": "context_embedder",
    "diffusion_model.time_in.in_layer": "timestep_embedder.linear_1",
    "diffusion_model.ref_time_in.out_layer": "target_timestep_embedder.linear_2",
    "diffusion_model.early_stream_modulations.txt.lin": "early_stream_modulations.txt.linear",
    "diffusion_model.single_stream_modulations.image.lin": "single_stream_modulations.image.linear",
    "diffusion_model.final_layer.image.linear": "proj_out",
    "diffusion_model.final_layer.image.adaLN_modulation.1": "norm_out.linear",
}


def _tiny_transformer(flow_map: bool = True) -> Flux3Transformer:
    return Flux3Transformer(
        in_channels=8,
        cond_in_channels=8,
        num_layers=2,
        num_single_layers=2,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=16,
        timestep_channels=16,
        axes_dims_rope=(4, 4, 4, 4),
        flow_map=flow_map,
    )


def _module(transformer: Flux3Transformer, path: str):
    return LoRALoader._get_target_module(transformer, path)


def _lora_pair(key: str, module, seed: int) -> dict[str, mx.array]:
    out_dim, in_dim = module.weight.shape
    rng = np.random.default_rng(seed)
    return {
        f"{key}.lora_A.weight": mx.array(rng.standard_normal((RANK, in_dim)).astype(np.float32)),
        f"{key}.lora_B.weight": mx.array(rng.standard_normal((out_dim, RANK)).astype(np.float32)),
    }


def _apply(transformer, tmp_path, weights, scale=1.0, bake_lora=True, name="lora"):
    path = tmp_path / f"{name}.safetensors"
    mx.save_safetensors(str(path), weights)
    return LoRALoader.load_and_apply_lora(
        lora_mapping=Flux3LoRAMapping.get_mapping(),
        transformer=transformer,
        lora_paths=[str(path)],
        lora_scales=[scale],
        bake_lora=bake_lora,
    )


def _resolve(key: str) -> set[str]:
    targets = set()
    for mapping in LoRALoader._build_pattern_mappings(Flux3LoRAMapping.get_mapping()):
        block = LoRALoader._match_pattern(key, mapping.source_pattern)
        if block is not None:
            targets.add(mapping.target_path.format(block=block))
    return targets


@pytest.mark.parametrize(("key", "target"), BFL_KEYS.items())
def test_bfl_keys_reach_their_module(key, target):
    for prefix in ("", "diffusion_model."):
        name = prefix + key.removeprefix("diffusion_model.")
        assert _resolve(f"{name}.lora_A.weight") == {target}
        assert _resolve(f"{name}.lora_B.weight") == {target}
    assert _resolve(f"lora_unet_{key.removeprefix('diffusion_model.').replace('.', '_')}.lora_up.weight") == {target}


@pytest.mark.parametrize(
    ("key", "target"),
    [
        ("transformer.mode_blocks.image.0.attn.to_out.0", "mode_blocks.image.0.attn.to_out"),
        ("transformer.txt_mode_blocks.0.attn.to_q", "txt_mode_blocks.0.attn.to_q"),
        ("transformer.single_transformer_blocks.1.ff.linear_in", "single_transformer_blocks.1.ff.linear_in"),
        ("base_model.model.x_embedders.image", "x_embedders.image"),
        ("transformer.norm_out.linear", "norm_out.linear"),
    ],
)
def test_diffusers_and_peft_keys_reach_their_module(key, target):
    assert _resolve(f"{key}.lora_A.weight") == {target}
    assert _resolve(f"{key}.lora_B.default.weight") == {target}


def _target_paths(kind: str) -> set[str]:
    paths = set()
    for target in Flux3LoRAMapping.get_mapping():
        if getattr(target, f"possible_{kind}_patterns"):
            blocks = range(2) if "{block}" in target.model_path else [0]
            paths.update(target.model_path.format(block=block) for block in blocks)
    return paths


def test_every_linear_of_the_transformer_is_a_target():
    transformer = _tiny_transformer()
    linears = {path for path, module in transformer.named_modules() if type(module).__name__ == "Linear"}
    assert linears == _target_paths("up")


def test_every_attention_norm_of_the_transformer_takes_a_diff():
    transformer = _tiny_transformer()
    norms = {path for path, module in transformer.named_modules() if type(module).__name__ == "RMSNorm"}
    assert norms == _target_paths("diff")


@pytest.mark.parametrize(
    ("key", "target"),
    [
        ("diffusion_model.single_blocks.1.norm.query_norm.diff", "single_transformer_blocks.1.attn.norm_q"),
        ("diffusion_model.content_mode_blocks.image.0.norm.key_norm.diff", "mode_blocks.image.0.attn.norm_k"),
        (
            "diffusion_model.content_mode_blocks.image_cond.1.norm.query_norm.diff",
            "mode_blocks.image_cond.1.attn.norm_q",
        ),
        ("txt_mode_blocks.0.norm.key_norm.diff", "txt_mode_blocks.0.attn.norm_k"),
        ("transformer.single_transformer_blocks.0.attn.norm_q.diff", "single_transformer_blocks.0.attn.norm_q"),
    ],
)
def test_norm_diff_keys_reach_their_norm(key, target):
    assert _resolve(key) == {target}


def _extracted_lora(transformer: Flux3Transformer) -> dict[str, mx.array]:
    # The layout of the published Base-to-distilled extracts: lora_down/up for the linears, .diff for the norms.
    rng = np.random.default_rng(0)
    shape = transformer.single_transformer_blocks[1].attn.norm_q.weight.shape
    key = "diffusion_model.single_blocks.1.q_proj"
    pair = _lora_pair(key, transformer.single_transformer_blocks[1].attn.to_q, seed=1)
    return {
        "diffusion_model.single_blocks.1.norm.query_norm.diff": mx.array(rng.standard_normal(shape).astype(np.float32)),
        "diffusion_model.content_mode_blocks.image.0.norm.key_norm.diff": mx.array(
            rng.standard_normal(shape).astype(np.float32)
        ),
        f"{key}.lora_down.weight": pair[f"{key}.lora_A.weight"],
        f"{key}.lora_up.weight": pair[f"{key}.lora_B.weight"],
    }


def test_an_extracted_lora_patches_the_norms_and_the_linears(tmp_path):
    transformer = _tiny_transformer(flow_map=False)
    weights = _extracted_lora(transformer)
    norm_q = transformer.single_transformer_blocks[1].attn.norm_q
    norm_k = transformer.mode_blocks["image"][0].attn.norm_k
    to_q = transformer.single_transformer_blocks[1].attn.to_q
    expected_q = np.array(norm_q.weight) + 1.5 * np.array(
        weights["diffusion_model.single_blocks.1.norm.query_norm.diff"]
    )
    expected_k = np.array(norm_k.weight) + 1.5 * np.array(
        weights["diffusion_model.content_mode_blocks.image.0.norm.key_norm.diff"]
    )
    up, down = (np.array(weights[f"diffusion_model.single_blocks.1.q_proj.lora_{x}.weight"]) for x in ("up", "down"))
    expected_linear = np.array(to_q.weight) + 1.5 * up @ down

    _apply(transformer, tmp_path, weights, scale=1.5)

    np.testing.assert_allclose(np.array(norm_q.weight), expected_q, rtol=1e-6)
    np.testing.assert_allclose(np.array(norm_k.weight), expected_k, rtol=1e-6)
    np.testing.assert_allclose(np.array(to_q.weight), expected_linear, rtol=1e-5, atol=1e-5)


def test_an_extracted_lora_without_baking_stops_before_it_changes_the_model(tmp_path):
    transformer = _tiny_transformer(flow_map=False)
    before = np.array(transformer.single_transformer_blocks[1].attn.norm_q.weight)

    with pytest.raises(ValueError, match="require bake_lora=True"):
        _apply(transformer, tmp_path, _extracted_lora(transformer), bake_lora=False)

    np.testing.assert_array_equal(np.array(transformer.single_transformer_blocks[1].attn.norm_q.weight), before)
    assert not isinstance(transformer.single_transformer_blocks[1].attn.to_q, LoRALinear)


def test_bake_adds_the_scaled_update(tmp_path):
    transformer = _tiny_transformer()
    keys = {k: v for k, v in BFL_KEYS.items() if "adaLN" not in k}
    weights, expected = {}, {}
    for seed, (key, target) in enumerate(keys.items()):
        module = _module(transformer, target)
        pair = _lora_pair(key, module, seed)
        weights.update(pair)
        update = np.array(pair[f"{key}.lora_B.weight"]) @ np.array(pair[f"{key}.lora_A.weight"])
        expected[target] = np.array(module.weight) + 1.5 * update

    _apply(transformer, tmp_path, weights, scale=1.5)

    for target, weight in expected.items():
        np.testing.assert_allclose(np.array(_module(transformer, target).weight), weight, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("bake_lora", [True, False])
def test_kohya_alpha_scales_the_update(tmp_path, bake_lora):
    # Alpha 1 at rank 2 halves the update. At alpha = rank (as in the published kohya Klein LoRAs) the scale is 1,
    # and a loader that never read the alpha would still pass.
    transformer = _tiny_transformer()
    key = "diffusion_model.single_blocks.0.q_proj"
    module = transformer.single_transformer_blocks[0].attn.to_q
    pair = _lora_pair(key, module, seed=0)
    down, up = pair[f"{key}.lora_A.weight"], pair[f"{key}.lora_B.weight"]
    expected = np.array(module.weight) + 1.5 * (1.0 / RANK) * (np.array(up) @ np.array(down))
    x = mx.array(np.random.default_rng(1).standard_normal((3, module.weight.shape[1])).astype(np.float32))

    _apply(
        transformer,
        tmp_path,
        {f"{key}.lora_down.weight": down, f"{key}.lora_up.weight": up, f"{key}.alpha": mx.array(1.0)},
        scale=1.5,
        bake_lora=bake_lora,
    )

    np.testing.assert_allclose(
        np.array(transformer.single_transformer_blocks[0].attn.to_q(x)), np.array(x) @ expected.T, rtol=1e-4, atol=1e-4
    )


def test_bfl_final_modulation_is_swapped_to_the_diffusers_order(tmp_path):
    bfl, diffusers = _tiny_transformer(), _tiny_transformer()
    diffusers.update(bfl.parameters())
    pair = _lora_pair("diffusion_model.final_layer.image.adaLN_modulation.1", bfl.norm_out.linear, seed=0)
    up = np.array(pair["diffusion_model.final_layer.image.adaLN_modulation.1.lora_B.weight"])
    half = up.shape[0] // 2
    _apply(bfl, tmp_path, pair, name="bfl")
    _apply(
        diffusers,
        tmp_path,
        {
            "transformer.norm_out.linear.lora_A.weight": pair[
                "diffusion_model.final_layer.image.adaLN_modulation.1.lora_A.weight"
            ],
            "transformer.norm_out.linear.lora_B.weight": mx.array(np.concatenate([up[half:], up[:half]])),
        },
        name="diffusers",
    )
    np.testing.assert_allclose(np.array(bfl.norm_out.linear.weight), np.array(diffusers.norm_out.linear.weight))


def test_unbaked_lora_wraps_the_layer(tmp_path):
    transformer = _tiny_transformer()
    key = "diffusion_model.single_blocks.1.q_proj"
    _, scales = _apply(
        transformer, tmp_path, _lora_pair(key, transformer.single_transformer_blocks[1].attn.to_q, 0), 0.7, False
    )
    assert isinstance(transformer.single_transformer_blocks[1].attn.to_q, LoRALinear)
    assert scales == [pytest.approx(0.7)]


def test_target_time_lora_on_base_stops_instead_of_half_applying(tmp_path):
    base = _tiny_transformer(flow_map=False)
    weights = _lora_pair("diffusion_model.single_blocks.0.q_proj", base.single_transformer_blocks[0].attn.to_q, 0)
    weights |= _lora_pair("diffusion_model.ref_time_in.in_layer", _tiny_transformer().target_timestep_embedder.linear_1, 1)  # fmt: skip
    with pytest.raises(ValueError, match="do not match"):
        _apply(base, tmp_path, weights)


@pytest.mark.parametrize("cli", [flux3_generate, flux3_edit_generate])
def test_flux3_clis_take_lora_arguments(cli, monkeypatch, tmp_path):
    lora = tmp_path / "outpaint.safetensors"
    mx.save_safetensors(str(lora), {"x": mx.zeros((1,))})
    argv = ["prog", "--prompt", "p", "--lora", str(lora), "1.5", "--no-bake-lora"]
    if cli is flux3_edit_generate:
        image = tmp_path / "in.png"
        image.write_bytes(b"")
        argv += ["--image-paths", str(image)]
    monkeypatch.setattr("sys.argv", argv)
    kwargs = lora_init_kwargs_from_args(cli.build_parser().parse_args())
    assert kwargs == {"lora_paths": [str(lora)], "lora_scales": [1.5], "bake_lora": False}


@pytest.mark.parametrize(("name", "cli"), [("mflux-generate-flux3", flux3_generate), ("mflux-generate-flux3-edit", flux3_edit_generate)])  # fmt: skip
def test_flux3_capabilities_report_lora_as_honored(name, cli):
    command = describe_command(name, cli.__name__)
    options = {option["flag"]: option for option in command["options"]}
    assert command["traits"]["lora"] is True
    assert options["--lora"]["status"] == "honored"
    assert options["--lora-style"]["status"] == "ignored"


@pytest.mark.parametrize("model_class", [Flux3Klein, Flux3KleinEdit])
def test_lora_arguments_come_after_the_existing_positional_ones(model_class):
    # Flux3Klein(quantize, model_path, model_config, compute_precision) was the signature before LoRA support.
    names = list(inspect.signature(model_class.__init__).parameters)[1:]
    assert names == ["quantize", "model_path", "model_config", "compute_precision", "lora_paths", "lora_scales", "bake_lora"]  # fmt: skip
