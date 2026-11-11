import mlx.core as mx
import numpy as np
import pytest

from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.sensenova.model.sensenova_transformer.sensenova_transformer import SenseNovaTransformer
from mflux.models.sensenova.weights.sensenova_lora_mapping import SenseNovaLoRAMapping

pytestmark = pytest.mark.fast

RANK = 4
LM = "language_model.model.layers"


def _resolve(key: str) -> set[tuple[str, str]]:
    targets = set()
    for mapping in LoRALoader._build_pattern_mappings(SenseNovaLoRAMapping.get_mapping()):
        block = LoRALoader._match_pattern(key, mapping.source_pattern)
        if block is not None:
            targets.add((mapping.target_path.format(block=block), mapping.matrix_name))
    return targets


@pytest.mark.parametrize(
    ("key", "target"),
    [
        # kohya, as SenseNova-U1.5-8B-MoT-LoRA-8step-V2.safetensors writes it.
        (f"{LM}.0.self_attn.q_proj_mot_gen.lora_down.weight", ("layers.0.self_attn.q_proj", "lora_A")),
        (f"{LM}.41.self_attn.o_proj_mot_gen.lora_up.weight", ("layers.41.self_attn.o_proj", "lora_B")),
        (f"{LM}.7.mlp_mot_gen.gate_proj.lora_down.weight", ("layers.7.mlp.gate_proj", "lora_A")),
        (f"{LM}.7.mlp_mot_gen.down_proj.alpha", ("layers.7.mlp.down_proj", "alpha")),
        # The same names under the ComfyUI and diffusers/PEFT prefixes.
        (f"diffusion_model.{LM}.3.self_attn.k_proj_mot_gen.lora_up.weight", ("layers.3.self_attn.k_proj", "lora_B")),
        (f"transformer.{LM}.5.mlp_mot_gen.up_proj.lora_A.weight", ("layers.5.mlp.up_proj", "lora_A")),
        # PEFT matrices without a .weight suffix, and with the adapter name.
        (f"{LM}.12.self_attn.v_proj_mot_gen.lora_A", ("layers.12.self_attn.v_proj", "lora_A")),
        (
            f"base_model.model.{LM}.2.self_attn.q_proj_mot_gen.lora_B.default.weight",
            ("layers.2.self_attn.q_proj", "lora_B"),
        ),  # fmt: skip
    ],
)
def test_published_key_formats_reach_their_module(key, target):
    assert _resolve(key) == {target}


def test_only_the_gen_stream_linears_are_targets():
    assert _resolve(f"{LM}.0.self_attn.q_proj.lora_down.weight") == set()
    assert _resolve(f"{LM}.0.mlp.gate_proj.lora_down.weight") == set()
    assert _resolve(f"{LM}.0.self_attn.q_norm_mot_gen.lora_down.weight") == set()
    assert len(SenseNovaLoRAMapping.get_mapping()) == 7


def test_the_targets_exist_in_the_transformer():
    transformer = SenseNovaTransformer(num_layers=1)
    modules = dict(transformer.named_modules())

    for target in SenseNovaLoRAMapping.get_mapping():
        assert target.model_path.format(block=0) in modules, target.model_path


def test_an_official_style_file_applies_alpha_over_rank_to_the_gen_stream(tmp_path):
    transformer = SenseNovaTransformer(num_layers=1, hidden_size=64, intermediate_size=64)
    rng = np.random.default_rng(0)
    weights, expected = {}, {}
    for name, path in (("q_proj", "self_attn.q_proj"), ("gate_proj", "mlp.gate_proj")):
        module = dict(transformer.named_modules())[f"layers.0.{path}"]
        out_dim, in_dim = module.weight.shape
        down = rng.standard_normal((RANK, in_dim)).astype(np.float32)
        up = rng.standard_normal((out_dim, RANK)).astype(np.float32)
        source = f"{LM}.0.self_attn.{name}_mot_gen" if name == "q_proj" else f"{LM}.0.mlp_mot_gen.{name}"
        weights[f"{source}.lora_down.weight"] = mx.array(down)
        weights[f"{source}.lora_up.weight"] = mx.array(up)
        # alpha 2 * rank: the update is doubled, as the official file's alpha / rank scale
        weights[f"{source}.alpha"] = mx.array(2.0 * RANK)
        expected[f"layers.0.{path}"] = 2.0 * up @ down
    path = tmp_path / "lora.safetensors"
    mx.save_safetensors(str(path), weights)

    LoRALoader.load_and_apply_lora(
        lora_mapping=SenseNovaLoRAMapping.get_mapping(),
        transformer=transformer,
        lora_paths=[str(path)],
        lora_scales=[1.0],
        bake_lora=False,
    )

    modules = dict(transformer.named_modules())
    for name, update in expected.items():
        layer = modules[name]
        assert isinstance(layer, LoRALinear), name
        np.testing.assert_allclose(
            np.array(layer.scale * (layer.lora_A @ layer.lora_B).T), update, rtol=1e-4, atol=1e-4
        )
    assert not isinstance(modules["layers.0.self_attn.k_proj"], LoRALinear)


@pytest.mark.parametrize("stored_bits", [None, 8])
def test_the_initializer_hands_the_dense_transformer_weights_to_the_loader(monkeypatch, stored_bits):
    # The loader folds a LoRA into dense weights in float32 before -q quantizes them; a stored-quantized
    # checkpoint has no dense weights to fold into.
    from types import SimpleNamespace

    from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
    from mflux.models.sensenova.sensenova_initializer import SenseNovaInitializer

    tree = {"layers": [{"self_attn": {"q_proj": {"weight": mx.zeros((4, 4), dtype=mx.bfloat16)}}}]}
    weights = LoadedWeights(components={"transformer": tree}, meta_data=MetaData(quantization_level=stored_bits))
    seen = {}

    def _record(**kwargs):
        seen.update(kwargs)
        return ["adapter"], [1.0]

    monkeypatch.setattr(LoRALoader, "load_and_apply_lora", staticmethod(_record))
    model = SimpleNamespace(transformer=SenseNovaTransformer(num_layers=1, hidden_size=64, intermediate_size=64))

    SenseNovaInitializer._apply_loras(model, weights, ["adapter"], [1.0], True)

    assert seen["dense_weights"] is (tree if stored_bits is None else None)
    assert seen["bake_lora"] is True
    assert (model.lora_paths, model.lora_scales) == (["adapter"], [1.0])
