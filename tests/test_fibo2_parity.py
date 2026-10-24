import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import torch
from mlx.utils import tree_flatten
from transformers import Qwen3VLConfig, Qwen3VLModel

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common_models.qwen3_vl.qwen3_vl_multimodal import Qwen3VLMultimodal
from mflux.models.fibo2.model.fibo2_text_encoder.fibo2_text_encoder import Fibo2TextEncoder
from mflux.models.fibo2.model.fibo2_transformer.fibo2_perceiver import Fibo2Perceiver
from mflux.models.fibo2.model.fibo2_transformer.fibo2_transformer import Fibo2Transformer

pytestmark = pytest.mark.fast

# A tiny diffusers BriaFibo2Transformer2DModel (4 layers, injection blocks 1 and 3, 4 to 8 gist tokens) with random
# weights, its inputs and its float32 CPU outputs, from diffusers main at 48cdbae by
# scripts/fibo2/export_parity_reference.py
TRANSFORMER_FIXTURE = Path(__file__).parent / "resources" / "fibo2_transformer_parity.npz"
TRANSFORMER_CASES = ("txt2img", "short_text", "long_text", "edit", "multi_edit")

# A tiny Qwen3-VL for the text encoder: 6 layers, the first 3 with deepstack vision features
IMAGE_TOKEN = 99
TEXT_ENCODER_CONFIG = {
    "image_token_id": IMAGE_TOKEN,
    "vision_start_token_id": 98,
    "vision_end_token_id": 97,
    "text_config": {
        "vocab_size": 128,
        "hidden_size": 32,
        "num_hidden_layers": 6,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "intermediate_size": 64,
        "head_dim": 16,
        "max_position_embeddings": 2048,
        "rope_theta": 5000000.0,
        "rms_norm_eps": 1e-6,
        "attention_bias": False,
        "rope_scaling": {"rope_type": "default", "mrope_section": [4, 2, 2], "mrope_interleaved": True},
    },
    "vision_config": {
        "depth": 3,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_heads": 2,
        "patch_size": 2,
        "temporal_patch_size": 2,
        "in_channels": 3,
        "spatial_merge_size": 2,
        "out_hidden_size": 32,
        "num_position_embeddings": 16,
        "deepstack_visual_indexes": [0, 1, 2],
        "hidden_act": "gelu_pytorch_tanh",
    },
}


@pytest.fixture(scope="module")
def reference():
    with np.load(TRANSFORMER_FIXTURE, allow_pickle=False) as archive:
        return dict(archive)


class _TinyTransformer:
    # diffusers indexes these layers inside an nn.Sequential; the Z-Image modules mflux reuses name them
    RENAMED = {
        "t_embedder.mlp.0.": "t_embedder.linear1.",
        "t_embedder.mlp.2.": "t_embedder.linear2.",
        "final_layer.adaLN_modulation.1.": "final_layer.adaLN_modulation.0.",
    }

    @staticmethod
    def build(data: dict[str, np.ndarray], dtype: mx.Dtype, monkeypatch: pytest.MonkeyPatch) -> Fibo2Transformer:
        monkeypatch.setattr(ModelConfig, "precision", dtype)
        config = {key: tuple(value) if isinstance(value, list) else value for key, value in json.loads(str(data["config"])).items()}  # fmt: off
        gist = {key: config.pop(key) for key in ("min_num_gist_tokens", "gist_step", "gist_min_text_len", "gist_max_text_len")}  # fmt: off
        model = Fibo2Transformer(**config)
        # Fibo2Transformer builds the checkpoint's gist range, 256 to 512 tokens; the fixture's is 4 to 8
        model.perceiver = Fibo2Perceiver(
            config["dim"],
            config["n_heads"],
            config["cap_feat_dim"],
            config["perceiver_num_layers"],
            config["norm_eps"],
            min_gist=gist["min_num_gist_tokens"],
            max_gist=config["max_num_gist_tokens"],
            gist_step=gist["gist_step"],
            min_text_len=gist["gist_min_text_len"],
            max_text_len=gist["gist_max_text_len"],
        )
        weights = []
        for key, value in data.items():
            if key.startswith("weight/"):
                name = key.removeprefix("weight/")
                for old, new in _TinyTransformer.RENAMED.items():
                    name = name.replace(old, new)
                weights.append((name, mx.array(value).astype(dtype)))
        model.load_weights(weights, strict=True)
        return model

    @staticmethod
    def run(data: dict[str, np.ndarray], model: Fibo2Transformer, case: str, dtype: mx.Dtype) -> np.ndarray:
        length, num_contexts, sigma = data[f"case/{case}"].tolist()
        contexts = [mx.array(data[f"input/context_{k}"]).astype(dtype) for k in range(int(num_contexts))]
        text = mx.array(data[f"input/text_{int(length)}"]).astype(dtype)
        velocity = model(mx.array(data["input/latents"]), mx.array([sigma]), text, context_latents=contexts or None)
        return np.array(velocity.astype(mx.float32))

    @staticmethod
    def relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
        return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


class _TinyQwen3VL:
    @staticmethod
    def pair(monkeypatch: pytest.MonkeyPatch) -> tuple[Qwen3VLModel, Fibo2TextEncoder]:
        monkeypatch.setattr(Fibo2TextEncoder, "IMAGE_TOKEN_ID", IMAGE_TOKEN)
        torch.manual_seed(0)
        reference = Qwen3VLModel(Qwen3VLConfig(**TEXT_ENCODER_CONFIG)).to(device="cpu", dtype=torch.float32).eval()
        model = Fibo2TextEncoder(TEXT_ENCODER_CONFIG["text_config"], TEXT_ENCODER_CONFIG["vision_config"])
        weights = [
            (key, Qwen3VLMultimodal.vision_weight(key, mx.array(value.detach().float().numpy())))
            for key, value in reference.state_dict().items()
        ]
        model.load_weights(weights, strict=False)
        # Every parameter comes from transformers but the RoPE frequencies, which both compute
        unloaded = {name for name, _ in tree_flatten(model.parameters())} - {name for name, _ in weights}
        assert unloaded == {"language_model.rotary_emb.inv_freq", "visual.rotary_pos_emb.inv_freq"}
        return reference, model


@pytest.mark.parametrize("case", TRANSFORMER_CASES)
def test_the_transformer_matches_diffusers_in_float32_on_the_cpu(reference, monkeypatch, case):
    # On the CPU float32 matmuls run in full precision. What is left is the timestep embedding: torch's and MLX's
    # float32 sin and cos differ by an ulp at its large arguments (sigma * 1000), about 1e-4 at the output. A wrong
    # RoPE plane or modulation gives 2e-2 or more.
    device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        model = _TinyTransformer.build(reference, mx.float32, monkeypatch)
        actual = _TinyTransformer.run(reference, model, case, mx.float32)
    finally:
        mx.set_default_device(device)
    assert _TinyTransformer.relative_error(actual, reference[f"expected/{case}"]) < 1e-3


@pytest.mark.parametrize("case", TRANSFORMER_CASES)
def test_the_transformer_matches_diffusers_to_bfloat16_rounding(reference, monkeypatch, case):
    # diffusers' own bfloat16 run of this random network is 3.8 to 5.8% off its float32 output (CPU and MPS alike)
    actual = _TinyTransformer.run(reference, _TinyTransformer.build(reference, mx.bfloat16, monkeypatch), case, mx.bfloat16)  # fmt: off
    assert np.isfinite(actual).all()
    assert _TinyTransformer.relative_error(actual, reference[f"expected/{case}"]) < 0.08


@pytest.mark.parametrize(
    ("ids", "grids"),
    [
        ([1, 2, 3, 4, 5], None),
        ([1, 98, 99, 99, 99, 99, 97, 5], [[1, 4, 4]]),
        ([1, 98, 99, 99, 99, 99, 97, 98, 99, 99, 97, 5], [[1, 4, 4], [1, 2, 4]]),
    ],
    ids=["text", "one-image", "two-images"],
)
def test_the_text_encoder_matches_transformers_layer_by_layer(monkeypatch, ids, grids):
    # FIBO 2 reads hidden states 5 to 35, numbered as transformers numbers them: 0 is the embeddings with the vision
    # features, and the deepstack features are added to the image tokens of states 1 to 3. transformers records those
    # three before the features are added and mflux after, so they are compared on the text tokens only. The last
    # state is not compared: transformers replaces it with the final norm, which FIBO 2 never reads.
    reference, model = _TinyQwen3VL.pair(monkeypatch)
    input_ids = torch.tensor([ids])
    kwargs, images = {}, ()
    if grids is not None:
        grid = torch.tensor(grids)
        pixels = torch.randn(int(grid.prod(-1).sum()), 24)
        kwargs = {"pixel_values": pixels, "image_grid_thw": grid, "mm_token_type_ids": (input_ids == IMAGE_TOKEN).int()}
        images = (mx.array(pixels.numpy()), mx.array(grid.numpy()))
    with torch.no_grad():
        expected = reference(input_ids=input_ids, use_cache=False, output_hidden_states=True, **kwargs).hidden_states
    device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        actual = model(mx.array(input_ids.numpy()), *images)
    finally:
        mx.set_default_device(device)
    text_tokens = np.array(ids) != IMAGE_TOKEN
    deepstack_layers = len(TEXT_ENCODER_CONFIG["vision_config"]["deepstack_visual_indexes"])
    for layer in range(TEXT_ENCODER_CONFIG["text_config"]["num_hidden_layers"]):
        tokens = text_tokens if 1 <= layer <= deepstack_layers else slice(None)
        np.testing.assert_allclose(
            np.array(actual[layer][0])[tokens],
            expected[layer][0].numpy()[tokens],
            atol=2e-5,
            rtol=1e-4,
            err_msg=f"layer {layer}",
        )
