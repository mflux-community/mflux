import json
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mflux.models.z_image.model.z_image_transformer.transformer import ZImageTransformer
from mflux.models.z_image.weights.z_image_weight_definition import ZImageNoTextEncoderWeightDefinition
from mflux.models.z_image.z_image_initializer import ZImageInitializer

pytestmark = pytest.mark.fast

GROUPS = ("noise_refiner", "context_refiner", "layers")


def test_the_streamed_transformer_gives_the_normal_output_for_two_steps_in_a_row(tmp_path):
    mx.random.seed(0)
    transformer = ZImageTransformer(
        dim=16, n_layers=3, n_heads=2, in_channels=4, cap_feat_dim=8, axes_dims=[2, 2, 4], axes_lens=[64, 32, 32]
    )
    mx.eval(transformer.parameters())
    x = mx.random.normal((4, 1, 8, 8))
    cap = mx.random.normal((5, 8))
    sigmas = mx.array([1.0, 0.5, 0.0])
    expected = [np.array(transformer(x, step, sigmas, cap)) for step in (0, 1)]

    folder = tmp_path / "transformer"
    folder.mkdir()
    tensors = {f"{group}.{i}.{name}": v for group in GROUPS for i, block in enumerate(getattr(transformer, group)) for name, v in tree_flatten(block.parameters())}  # fmt: off
    mx.save_safetensors(str(folder / "0.safetensors"), tensors)
    (folder / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, "0.safetensors")})
    )
    model = SimpleNamespace(small_text_encoder=True, transformer=transformer)
    ZImageInitializer._apply_ssd_stream(model, str(tmp_path), ZImageNoTextEncoderWeightDefinition)

    for step in (0, 1):
        np.testing.assert_array_equal(np.array(transformer(x, step, sigmas, cap)), expected[step])
