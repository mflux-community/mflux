import json

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn

from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.ssd_stream.streamer import Streamer
from mflux.models.moss_image.moss_image_initializer import MOSSImageInitializer
from mflux.models.moss_image.moss_image_streamed_lora import MOSSImageStreamedLoRA
from mflux.models.moss_image.weights.moss_image_lora_mapping import MOSSImageLoRAMapping
from tests.test_ssd_stream import _save

pytestmark = pytest.mark.fast

LAYERS = 3
DIM_IN = 8
DIM_OUT = 6
RANK = 4
PEFT_METADATA = "lora_adapter_metadata"


class _Layer(nn.Module):
    # The projections of a MOSS-Image block, with different input and output sizes so that a transposed factor shows.
    def __init__(self):
        super().__init__()
        self.proj = {name: nn.Linear(DIM_IN, DIM_OUT, bias=False) for name in MOSSImageLoRAMapping.PROJECTIONS}

    def __call__(self, x):
        return mx.concatenate([self.proj[name](x) for name in MOSSImageLoRAMapping.PROJECTIONS], axis=-1)


class _Holder(nn.Module):
    def __init__(self, layers: list[nn.Module]):
        super().__init__()
        self.layers = layers


def _layers() -> list[_Layer]:
    mx.random.seed(0)
    layers = [_Layer() for _ in range(LAYERS)]
    for layer in layers:
        for projection in layer.proj.values():
            projection.weight = projection.weight.astype(mx.bfloat16)
    mx.eval([layer.parameters() for layer in layers])
    return layers


def _copy(layers: list[_Layer]) -> list[_Layer]:
    copies = [_Layer() for _ in layers]
    for copy, layer in zip(copies, layers):
        copy.update(layer.parameters())
    mx.eval([copy.parameters() for copy in copies])
    return copies


def _kohya(rng, alpha: float | None, projections=MOSSImageLoRAMapping.PROJECTIONS, layers=range(LAYERS)) -> dict:
    files = {}
    for layer in layers:
        for name in projections:
            prefix = f"diffusion_model.layers.{layer}.proj.{name}"
            files[f"{prefix}.lora_down.weight"] = mx.array(rng.standard_normal((RANK, DIM_IN)).astype(np.float32))
            files[f"{prefix}.lora_up.weight"] = mx.array(rng.standard_normal((DIM_OUT, RANK)).astype(np.float32))
            if alpha is not None:
                files[f"{prefix}.alpha"] = mx.array(alpha)
    return files


def _peft(rng, projections=MOSSImageLoRAMapping.PROJECTIONS) -> dict:
    files = {}
    for layer in range(LAYERS):
        for name in projections:
            prefix = f"layers.{layer}.proj.{name}"
            files[f"{prefix}.lora_A"] = mx.array(rng.standard_normal((RANK, DIM_IN)).astype(np.float32))
            files[f"{prefix}.lora_B"] = mx.array(rng.standard_normal((DIM_OUT, RANK)).astype(np.float32))
    return files


def _write(path, tensors: dict, metadata: dict | None = None) -> str:
    mx.save_safetensors(str(path), tensors, metadata=metadata or {"format": "pt"})
    return str(path)


def _with_loader(layers: list[_Layer], lora: str, strength: float) -> list[_Layer]:
    copies = _copy(layers)
    LoRALoader.load_and_apply_lora(
        lora_mapping=MOSSImageLoRAMapping.get_mapping(),
        transformer=_Holder(copies),
        lora_paths=[lora],
        lora_scales=[strength],
        bake_lora=False,
    )
    return copies


def _streamed(layers: list[_Layer], folder, lora: str, strength: float) -> list:
    _save(layers, folder)
    modules = _copy(layers)
    reader = MOSSImageStreamedLoRA(lora, strength)
    Streamer(cycle=False).wrap(modules, folder, "layers.", reader)
    reader.check_applied()
    return modules


def _assert_same_as_the_loader(tmp_path, lora: str, strength: float = 0.5) -> None:
    layers = _layers()
    expected = _with_loader(layers, lora, strength)
    streamed = _streamed(layers, tmp_path / "model", lora, strength)
    plain = _copy(layers)
    x = mx.random.normal((2, DIM_IN)).astype(mx.bfloat16)
    for index in range(LAYERS):
        want = np.array(expected[index](x).astype(mx.float32))
        got = np.array(streamed[index](x).astype(mx.float32))
        base = np.array(plain[index](x).astype(mx.float32))
        # The update has to matter, or a wrong scale would pass: it is far above the tolerance below.
        assert np.abs(want - base).max() > 10 * 0.05
        np.testing.assert_allclose(got, want, rtol=3e-2, atol=5e-2)


def test_an_alpha_tensor_gives_the_loaders_output(tmp_path):
    lora = _write(tmp_path / "kohya.safetensors", _kohya(np.random.default_rng(0), alpha=2.0 * RANK))
    _assert_same_as_the_loader(tmp_path, lora)


def test_the_alpha_of_the_file_metadata_gives_the_loaders_output(tmp_path):
    # diffusers and PEFT keep lora_alpha in the file metadata, not in a tensor per layer.
    metadata = {"format": "pt", PEFT_METADATA: json.dumps({"r": RANK, "lora_alpha": 3.0 * RANK})}
    lora = _write(tmp_path / "peft.safetensors", _peft(np.random.default_rng(1)), metadata)
    _assert_same_as_the_loader(tmp_path, lora)


def test_one_alpha_and_rank_for_the_whole_file_give_the_loaders_output(tmp_path):
    metadata = {"format": "pt", "alpha": str(float(RANK) / 2), "rank": str(RANK)}
    lora = _write(tmp_path / "whole.safetensors", _peft(np.random.default_rng(2)), metadata)
    _assert_same_as_the_loader(tmp_path, lora)


def test_a_file_without_an_alpha_gives_the_loaders_output(tmp_path):
    lora = _write(tmp_path / "none.safetensors", _kohya(np.random.default_rng(3), alpha=None))
    _assert_same_as_the_loader(tmp_path, lora)


def test_a_file_that_trains_some_projections_of_some_blocks_gives_the_loaders_output(tmp_path):
    rng = np.random.default_rng(4)
    lora = _write(
        tmp_path / "sparse.safetensors", _kohya(rng, alpha=float(RANK), projections=("q", "ffn_down"), layers=[1])
    )
    layers = _layers()
    expected = _with_loader(layers, lora, 1.0)
    streamed = _streamed(layers, tmp_path / "model", lora, 1.0)
    x = mx.random.normal((2, DIM_IN)).astype(mx.bfloat16)
    for index in range(LAYERS):
        np.testing.assert_allclose(
            np.array(streamed[index](x).astype(mx.float32)),
            np.array(expected[index](x).astype(mx.float32)),
            rtol=3e-2,
            atol=5e-2,
        )


def test_a_dora_file_is_refused_as_the_loader_refuses_it(tmp_path):
    tensors = _kohya(np.random.default_rng(5), alpha=None)
    tensors["diffusion_model.layers.0.proj.q.lora_magnitude_vector"] = mx.ones((DIM_OUT,))
    lora = _write(tmp_path / "dora.safetensors", tensors)
    with pytest.raises(ValueError, match="DoRA"):
        MOSSImageStreamedLoRA(lora, 1.0)
    with pytest.raises(ValueError, match="DoRA"):
        _with_loader(_layers(), lora, 1.0)


def test_a_file_that_names_no_projection_of_the_model_is_refused(tmp_path):
    tensors = {
        f"diffusion_model.layers.{LAYERS + 39}.proj.q.lora_down.weight": mx.zeros((RANK, DIM_IN)),
        f"diffusion_model.layers.{LAYERS + 39}.proj.q.lora_up.weight": mx.zeros((DIM_OUT, RANK)),
    }
    lora = _write(tmp_path / "elsewhere.safetensors", tensors)
    layers = _layers()
    _save(layers, tmp_path / "model")
    reader = MOSSImageStreamedLoRA(lora, 1.0)
    Streamer(cycle=False).wrap(_copy(layers), tmp_path / "model", "layers.", reader)
    with pytest.raises(ValueError, match="No LoRA layers were applied"):
        reader.check_applied()


def test_half_a_pair_is_refused_instead_of_dropped(tmp_path):
    tensors = _kohya(np.random.default_rng(6), alpha=None)
    del tensors["diffusion_model.layers.0.proj.q.lora_up.weight"]
    lora = _write(tmp_path / "half.safetensors", tensors)
    layers = _layers()
    _save(layers, tmp_path / "model")
    with pytest.raises(ValueError, match="only one of the two matrices"):
        Streamer(cycle=False).wrap(_copy(layers), tmp_path / "model", "layers.", MOSSImageStreamedLoRA(lora, 1.0))


def test_the_options_that_stream_a_lora_are_checked_before_any_weight_is_read():
    check = MOSSImageInitializer._check_ssd_stream
    with pytest.raises(ValueError, match="--stream-lora needs --ssd-stream"):
        check(False, True, None, ["a.safetensors"], False)
    with pytest.raises(ValueError, match="one LoRA file, and 2 were given"):
        check(True, True, None, ["a.safetensors", "b.safetensors"], False)
    with pytest.raises(ValueError, match="one LoRA file, and 0 were given"):
        check(True, True, None, None, False)
    with pytest.raises(ValueError, match="--bake-lora"):
        check(True, True, None, ["a.safetensors"], True)
    # A single LoRA streams, and several load as usual when they are not streamed.
    check(True, True, None, ["a.safetensors"], False)
    check(True, False, None, ["a.safetensors", "b.safetensors"], False)
    check(False, False, 8, ["a.safetensors"], True)
