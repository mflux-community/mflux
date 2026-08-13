import json
import os
import struct
from pathlib import Path

import pytest
from mlx.utils import tree_flatten

from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.common.weights.mapping.weight_mapper import WeightMapper
from mflux.models.ideogram4.weights import Ideogram4WeightDefinition
from mflux.models.ideogram4.weights.ideogram4_weight_mapping import Ideogram4WeightMapping

MODEL_PATH_ENV = "MFLUX_IDEOGRAM4_MODEL_PATH"
REPO_ID = "ideogram-ai/ideogram-4-fp8"


def _model_path() -> Path:
    value = os.environ.get(MODEL_PATH_ENV)
    if value:
        return Path(value).expanduser()
    cached_path = PathResolution._find_complete_cached_snapshot(
        REPO_ID,
        Ideogram4WeightDefinition.get_download_patterns(),
    )
    if cached_path is None:
        pytest.skip(f"Cache {REPO_ID} or set {MODEL_PATH_ENV} to validate the Ideogram 4 checkpoint layout")
    return cached_path


def _read_header(path: Path) -> dict[str, dict]:
    with path.open("rb") as file:
        header_len = struct.unpack("<Q", file.read(8))[0]
        header = json.loads(file.read(header_len))
    return {key: value for key, value in header.items() if key != "__metadata__" and isinstance(value, dict)}


def _linear_shapes(prefix: str, out_dim: int, in_dim: int, bias: bool = True) -> dict[str, tuple[int, ...]]:
    shapes = {
        f"{prefix}.weight": (out_dim, in_dim),
        f"{prefix}.weight_scale": (out_dim,),
    }
    if bias:
        shapes[f"{prefix}.bias"] = (out_dim,)
    return shapes


def _transformer_shapes(config_path: Path) -> dict[str, tuple[int, ...]]:
    config = json.loads(config_path.read_text())
    num_layers = int(config.get("num_layers", 34))
    num_heads = int(config.get("num_attention_heads", 18))
    head_dim = int(config.get("attention_head_dim", 256))
    emb_dim = num_heads * head_dim
    intermediate_size = int(config.get("intermediate_size", 12288))
    adanln_dim = int(config.get("adaln_dim", 512))
    in_channels = int(config.get("in_channels", 128))
    llm_features_dim = int(config.get("llm_features_dim", 53248))

    shapes = {}
    shapes.update(_linear_shapes("input_proj", emb_dim, in_channels))
    shapes.update(_linear_shapes("llm_cond_proj", emb_dim, llm_features_dim))
    shapes.update(_linear_shapes("t_embedding.mlp_in", emb_dim, emb_dim))
    shapes.update(_linear_shapes("t_embedding.mlp_out", emb_dim, emb_dim))
    shapes.update(_linear_shapes("adaln_proj", adanln_dim, emb_dim))
    shapes.update(_linear_shapes("final_layer.adaln_modulation", emb_dim, adanln_dim))
    shapes.update(_linear_shapes("final_layer.linear", in_channels, emb_dim))
    shapes["llm_cond_norm.weight"] = (llm_features_dim,)
    shapes["embed_image_indicator.weight"] = (2, emb_dim)

    for layer in range(num_layers):
        base = f"layers.{layer}"
        shapes.update(_linear_shapes(f"{base}.attention.qkv", 3 * emb_dim, emb_dim, bias=False))
        shapes.update(_linear_shapes(f"{base}.attention.o", emb_dim, emb_dim, bias=False))
        shapes.update(_linear_shapes(f"{base}.feed_forward.w1", intermediate_size, emb_dim, bias=False))
        shapes.update(_linear_shapes(f"{base}.feed_forward.w2", emb_dim, intermediate_size, bias=False))
        shapes.update(_linear_shapes(f"{base}.feed_forward.w3", intermediate_size, emb_dim, bias=False))
        shapes.update(_linear_shapes(f"{base}.adaln_modulation", 4 * emb_dim, adanln_dim))
        shapes[f"{base}.attention.norm_q.weight"] = (head_dim,)
        shapes[f"{base}.attention.norm_k.weight"] = (head_dim,)
        shapes[f"{base}.attention_norm1.weight"] = (emb_dim,)
        shapes[f"{base}.ffn_norm1.weight"] = (emb_dim,)
        shapes[f"{base}.attention_norm2.weight"] = (emb_dim,)
        shapes[f"{base}.ffn_norm2.weight"] = (emb_dim,)
    return shapes


def _text_encoder_shapes(config_path: Path) -> dict[str, tuple[int, ...]]:
    config = json.loads(config_path.read_text())["text_config"]
    num_layers = int(config.get("num_hidden_layers", 36))
    vocab_size = int(config.get("vocab_size", 151936))
    hidden_size = int(config.get("hidden_size", 4096))
    num_attention_heads = int(config.get("num_attention_heads", 32))
    num_key_value_heads = int(config.get("num_key_value_heads", 8))
    head_dim = int(config.get("head_dim", 128))
    intermediate_size = int(config.get("intermediate_size", 12288))

    shapes = {
        "embed_tokens.weight": (vocab_size, hidden_size),
        "norm.weight": (hidden_size,),
    }
    for layer in range(num_layers):
        base = f"layers.{layer}"
        shapes.update(_linear_shapes(f"{base}.self_attn.q_proj", num_attention_heads * head_dim, hidden_size, False))
        shapes.update(_linear_shapes(f"{base}.self_attn.k_proj", num_key_value_heads * head_dim, hidden_size, False))
        shapes.update(_linear_shapes(f"{base}.self_attn.v_proj", num_key_value_heads * head_dim, hidden_size, False))
        shapes.update(_linear_shapes(f"{base}.self_attn.o_proj", hidden_size, num_attention_heads * head_dim, False))
        shapes.update(_linear_shapes(f"{base}.mlp.gate_proj", intermediate_size, hidden_size, False))
        shapes.update(_linear_shapes(f"{base}.mlp.up_proj", intermediate_size, hidden_size, False))
        shapes.update(_linear_shapes(f"{base}.mlp.down_proj", hidden_size, intermediate_size, False))
        shapes[f"{base}.input_layernorm.weight"] = (hidden_size,)
        shapes[f"{base}.post_attention_layernorm.weight"] = (hidden_size,)
        shapes[f"{base}.self_attn.q_norm.weight"] = (head_dim,)
        shapes[f"{base}.self_attn.k_norm.weight"] = (head_dim,)
    return shapes


def _assert_header_shapes(actual: dict[str, dict], expected: dict[str, tuple[int, ...]]) -> None:
    assert set(actual) == set(expected)
    mismatches = {
        key: (expected[key], tuple(value["shape"]))
        for key, value in actual.items()
        if tuple(value["shape"]) != expected[key]
    }
    assert mismatches == {}


@pytest.mark.fast
def test_ideogram4_local_checkpoint_headers_match_mflux_model_shapes() -> None:
    root = _model_path()

    for subdir in ("transformer", "unconditional_transformer"):
        header = _read_header(root / subdir / "diffusion_pytorch_model.safetensors")
        _assert_header_shapes(header, _transformer_shapes(root / subdir / "config.json"))

    text_header = _read_header(root / "text_encoder" / "model.safetensors")
    filtered_text_header = {
        key[len("language_model.") :]: value
        for key, value in text_header.items()
        if key.startswith("language_model.")
        and key[len("language_model.") :].startswith(("embed_tokens.", "layers.", "norm."))
    }
    _assert_header_shapes(filtered_text_header, _text_encoder_shapes(root / "text_encoder" / "config.json"))


@pytest.mark.fast
def test_ideogram4_transformer_mapping_covers_checkpoint_keys() -> None:
    root = _model_path()
    mapping = Ideogram4WeightMapping.get_transformer_mapping()

    for subdir in ("transformer", "unconditional_transformer"):
        header = _read_header(root / subdir / "diffusion_pytorch_model.safetensors")
        config = json.loads((root / subdir / "config.json").read_text())
        num_layers = int(config.get("num_layers", 34))
        flat_mapping = WeightMapper._build_flat_mapping(mapping, num_blocks=0, num_layers=num_layers)
        unmapped = sorted(key for key in header if key not in flat_mapping)
        assert unmapped == []


@pytest.mark.fast
def test_ideogram4_local_checkpoint_vae_mapping_loads() -> None:
    root = _model_path()
    vae_component = Ideogram4WeightDefinition.get_components()[0]
    weights, quantization_level, version = WeightLoader._load_component(root, vae_component)

    assert quantization_level is None
    assert version is None
    assert len(tree_flatten(weights)) == 250
