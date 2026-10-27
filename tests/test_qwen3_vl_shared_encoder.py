import zlib
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_unflatten

from mflux.models.common_models.qwen3_vl.qwen3_vl_multimodal import Qwen3VLMultimodal
from mflux.models.qwen21.model.qwen21_text_encoder.language_model import TextRMSNorm
from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_text_encoder import Qwen21TextEncoder
from mflux.models.qwen21.model.qwen21_text_encoder.text_encoder import QwenImage21TextEncoder

pytestmark = pytest.mark.fast

# Qwen-Image-2.1's Qwen3-VL text and vision encoder at toy size, with weights set from each parameter's name, on
# text alone and on text with two images (vision tower, deepstack, 3D positions). The reference was recorded before
# the Qwen3-VL pieces moved to common_models, so equality shows the move changed nothing for Qwen-Image-2.1.
REFERENCE = Path(__file__).parent / "resources" / "qwen3_vl" / "qwen3_vl_shared_encoder.npz"
IMAGE_TOKEN = 120
CONFIG = {
    "image_token_id": IMAGE_TOKEN,
    "text_config": {
        "vocab_size": 128,
        "hidden_size": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 32,
        "intermediate_size": 96,
        "max_position_embeddings": 4096,
        "rope_theta": 5000000.0,
        "rope_scaling": {"mrope_section": [6, 5, 5]},
        "attention_bias": False,
        "rms_norm_eps": 1e-6,
    },
    "vision_config": {
        "depth": 3,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_heads": 2,
        "out_hidden_size": 64,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "num_position_embeddings": 64,
        "deepstack_visual_indexes": [0, 2],
    },
}
GRIDS = [[1, 4, 6], [1, 2, 2]]


class SharedEncoderOutputs:
    @staticmethod
    def compute() -> dict[str, np.ndarray]:
        # In float32, and in bfloat16 as the model runs, where Qwen3-VL's RMSNorm rounding shows
        outputs = {}
        for dtype, suffix in ((mx.float32, ""), (mx.bfloat16, "_bf16")):
            for name, value in SharedEncoderOutputs._compute(dtype).items():
                outputs[f"{name}{suffix}"] = value
        return outputs

    @staticmethod
    def _compute(dtype: mx.Dtype) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(0)
        input_ids, pixel_values, grids = SharedEncoderOutputs._inputs(rng)
        pixel_values = pixel_values.astype(dtype)
        text_ids = mx.array(rng.integers(0, IMAGE_TOKEN, size=(1, 11)).astype(np.int32))

        encoder = SharedEncoderOutputs._with_named_weights(QwenImage21TextEncoder(CONFIG), dtype)
        text_only = Qwen21TextEncoder(
            vocab_size=128,
            hidden_size=64,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=1,
            intermediate_size=96,
            head_dim=32,
            mrope_section=[6, 5, 5],
        )
        text_only = SharedEncoderOutputs._with_named_weights(text_only, dtype)
        norm = SharedEncoderOutputs._with_named_weights(TextRMSNorm(64, 1e-6), dtype)
        merge = CONFIG["vision_config"]["spatial_merge_size"]
        hidden = mx.array(rng.standard_normal((3, 64)).astype(np.float32)).astype(dtype)
        outputs = {
            "with_images": SharedEncoderOutputs._numpy(encoder(input_ids, pixel_values, grids)),
            "text_alone": SharedEncoderOutputs._numpy(encoder(text_ids)),
            "positions": np.array(Qwen3VLMultimodal.mrope_positions(input_ids, grids, IMAGE_TOKEN, merge)),
            "text_encoder": SharedEncoderOutputs._numpy(text_only(text_ids)),
            "norm": SharedEncoderOutputs._numpy(norm(hidden)),
        }
        # A bfloat16 greedy token can turn on one rounding step between chips; tokens are checked in float32
        if dtype == mx.float32:
            outputs["generated"] = np.array(encoder.generate(input_ids, pixel_values, grids, max_new_tokens=6))
        return outputs

    @staticmethod
    def _numpy(value: mx.array) -> np.ndarray:
        return np.array(value.astype(mx.float32))

    @staticmethod
    def _inputs(rng: np.random.Generator) -> tuple[mx.array, mx.array, mx.array]:
        # Text, the first image's tokens, more text, the second image's tokens, then text
        merge = CONFIG["vision_config"]["spatial_merge_size"]
        counts = [t * h * w // merge**2 for t, h, w in GRIDS]
        ids = [*rng.integers(0, IMAGE_TOKEN, 5), *[IMAGE_TOKEN] * counts[0], *rng.integers(0, IMAGE_TOKEN, 4)]
        ids += [*[IMAGE_TOKEN] * counts[1], *rng.integers(0, IMAGE_TOKEN, 6)]
        patch = 3 * CONFIG["vision_config"]["temporal_patch_size"] * CONFIG["vision_config"]["patch_size"] ** 2
        patches = sum(t * h * w for t, h, w in GRIDS)
        pixel_values = mx.array(rng.standard_normal((patches, patch)), dtype=mx.float32)
        return mx.array(np.array([ids], dtype=np.int32)), pixel_values, mx.array(GRIDS)

    @staticmethod
    def _with_named_weights(module, dtype: mx.Dtype):
        # Each parameter's values come from its own name, so the order modules are built in does not matter. Matrices
        # are scaled by their fan-in and norms sit near 1, so activations have a realistic size
        weights = []
        for name, value in tree_flatten(module.parameters()):
            rng = np.random.default_rng(zlib.crc32(name.encode()))
            noise = rng.standard_normal(value.shape).astype(np.float32)
            if value.ndim >= 2:
                values = noise / np.sqrt(np.prod(value.shape[1:]))
            elif name.endswith("weight"):
                values = 1.0 + 0.1 * noise
            else:
                values = 0.1 * noise
            weights.append((name, mx.array(values.astype(np.float32)).astype(dtype)))
        module.update(tree_unflatten(weights))
        return module


def test_shared_qwen3_vl_pieces_reproduce_the_recorded_outputs():
    reference = np.load(REFERENCE)
    outputs = SharedEncoderOutputs.compute()
    assert set(outputs) == set(reference.files)
    for name, value in outputs.items():
        if np.issubdtype(value.dtype, np.integer):
            np.testing.assert_array_equal(value, reference[name], err_msg=name)
        elif name.endswith("_bf16"):
            # bfloat16 GPU results differ more between chips than float32 ones
            np.testing.assert_allclose(value, reference[name], rtol=2e-2, atol=2e-2, err_msg=name)
        else:
            # Float32 GPU results can differ in the last bits between chips
            np.testing.assert_allclose(value, reference[name], rtol=1e-5, atol=1e-5, err_msg=name)


def test_hidden_states_end_with_the_output_of_the_forward_pass():
    # hidden_states is transformers' output_hidden_states: the embeddings, then every layer, deepstack included
    rng = np.random.default_rng(0)
    model = SharedEncoderOutputs._with_named_weights(QwenImage21TextEncoder(CONFIG), mx.float32).language_model
    hidden = mx.array(rng.standard_normal((1, 9, 64)).astype(np.float32))
    positions = mx.broadcast_to(mx.arange(9)[None, None], (3, 1, 9))
    image_indices = mx.array([2, 3, 4], dtype=mx.int32)
    deepstack = [mx.array(rng.standard_normal((3, 64)).astype(np.float32)) for _ in range(2)]
    for extra in ((), (image_indices, deepstack)):
        states = model.hidden_states(hidden, positions, *extra)
        assert len(states) == len(model.layers) + 1
        np.testing.assert_array_equal(np.array(states[0]), np.array(hidden))
        np.testing.assert_array_equal(np.array(states[-1]), np.array(model(hidden, positions, *extra)))
    plain = model.hidden_states(hidden, positions)
    stacked = model.hidden_states(hidden, positions, image_indices, deepstack)
    assert not np.allclose(np.array(plain[1]), np.array(stacked[1]))
