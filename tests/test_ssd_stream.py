import json

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from mlx.utils import tree_flatten

from mflux.models.common.lora.layer.fused_linear_lora_layer import FusedLoRALinear
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.ssd_stream.streamer import StreamedBlock, Streamer
from mflux.models.moss_image.moss_image_initializer import MOSSImageInitializer
from mflux.models.moss_image.variants.txt2img.moss_image import MOSSImage

pytestmark = pytest.mark.fast

DIM = 8
BLOCKS = 4


def _blocks() -> list[nn.Linear]:
    mx.random.seed(0)
    blocks = [nn.Linear(DIM, DIM) for _ in range(BLOCKS)]
    mx.eval([block.parameters() for block in blocks])
    return blocks


def _save(blocks: list[nn.Module], folder, shards: int = 2) -> None:
    # The layout of an mflux-save component: tensors named layers.N.<name> in shards, plus an index file.
    folder.mkdir(parents=True, exist_ok=True)
    weight_map = {}
    per_shard = [{} for _ in range(shards)]
    for index, block in enumerate(blocks):
        for name, value in tree_flatten(block.parameters()):
            per_shard[index % shards][f"layers.{index}.{name}"] = value
    for number, tensors in enumerate(per_shard):
        shard = f"model-{number}.safetensors"
        mx.save_safetensors(str(folder / shard), tensors)
        weight_map.update({key: shard for key in tensors})
    (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


def _streamed(tmp_path, cycle: bool, blocks: list[nn.Module] | None = None):
    blocks = blocks or _blocks()
    _save(blocks, tmp_path)
    expected = [_copy(block) for block in blocks]
    streamer = Streamer(cycle=cycle)
    streamer.wrap(blocks, tmp_path, "layers.")
    return streamer, blocks, expected


def _copy(block: nn.Linear) -> nn.Linear:
    copy = nn.Linear(DIM, DIM)
    copy.update(block.parameters())
    mx.eval(copy.parameters())
    return copy


def _installed(block: StreamedBlock) -> bool:
    return any(value.size > 1 for _, value in tree_flatten(block.module.parameters()))


def test_the_streamed_output_equals_the_normal_output(tmp_path):
    streamer, blocks, expected = _streamed(tmp_path, cycle=False)
    x = mx.random.normal((2, DIM))
    for streamed, normal in zip(blocks, expected):
        np.testing.assert_array_equal(np.array(streamed(x)), np.array(normal(x)))


def test_the_placeholder_returns_after_each_call(tmp_path):
    streamer, blocks, _ = _streamed(tmp_path, cycle=False)
    assert not any(_installed(block) for block in blocks)
    x = mx.random.normal((2, DIM))
    for block in blocks:
        block(x)
        assert not _installed(block)


class _Interrupted(nn.Linear):
    # Runs the real layer, so the weights are installed, then stops as a KeyboardInterrupt would.
    def __call__(self, x):
        super().__call__(x)
        raise KeyboardInterrupt


def test_the_placeholder_returns_when_the_call_raises(tmp_path):
    blocks = [_Interrupted(DIM, DIM) for _ in range(BLOCKS)]
    mx.eval([block.parameters() for block in blocks])
    streamer, blocks, _ = _streamed(tmp_path, cycle=False, blocks=blocks)
    with pytest.raises(KeyboardInterrupt):
        blocks[1](mx.zeros((1, DIM)))
    assert not _installed(blocks[1])


def test_a_run_that_cycles_wraps_to_the_first_block(tmp_path):
    streamer, blocks, expected = _streamed(tmp_path, cycle=True)
    x = mx.random.normal((2, DIM))
    blocks[-1](x)
    assert list(streamer.pending) == [blocks[0]]
    np.testing.assert_array_equal(np.array(blocks[0](x)), np.array(expected[0](x)))


def test_a_run_that_does_not_cycle_reads_nothing_after_the_last_block(tmp_path):
    streamer, blocks, _ = _streamed(tmp_path, cycle=False)
    blocks[-1](mx.random.normal((2, DIM)))
    assert streamer.pending == {}


def test_a_skipped_block_does_not_install_the_weights_of_another_block(tmp_path):
    streamer, blocks, expected = _streamed(tmp_path, cycle=True)
    x = mx.random.normal((2, DIM))
    # A pass that skips block 1, as --uncond-skip-layers does: 0, 2, 3.
    for index in (0, 2, 3):
        np.testing.assert_array_equal(np.array(blocks[index](x)), np.array(expected[index](x)))
        assert not _installed(blocks[1])


def test_blocks_that_the_pass_skips_do_not_pile_up_in_memory(tmp_path):
    streamer, blocks, _ = _streamed(tmp_path, cycle=True, blocks=[nn.Linear(DIM, DIM) for _ in range(12)])
    x = mx.random.normal((2, DIM))
    # The odd blocks are skipped, and each even block reads the odd one after it ahead.
    for index in range(0, 12, 2):
        blocks[index](x)
        assert len(streamer.pending) <= 1


def test_a_folder_without_the_index_file_is_refused_with_the_layout_it_needs(tmp_path):
    with pytest.raises(FileNotFoundError, match="mflux layout"):
        Streamer(cycle=False).wrap(_blocks(), tmp_path, "layers.")


class _Projection(nn.Module):
    def __init__(self, linear: nn.Module):
        super().__init__()
        self.proj = {"q": linear}

    def __call__(self, x):
        return self.proj["q"](x)


def test_the_base_weights_of_a_fused_lora_layer_are_found_and_freed(tmp_path):
    # Two LoRAs on one projection make a FusedLoRALinear, which keeps the base layer as `base_linear`, not `linear`.
    mx.random.seed(1)
    plain = [_Projection(nn.Linear(DIM, DIM, bias=False)) for _ in range(2)]
    mx.eval([block.parameters() for block in plain])
    _save(plain, tmp_path)
    modules = [_Projection(nn.Linear(DIM, DIM, bias=False)) for _ in range(2)]
    for module in modules:
        base = module.proj["q"]
        loras = [LoRALinear.from_linear(base, r=2) for _ in range(2)]
        module.proj["q"] = FusedLoRALinear(base_linear=base, loras=loras)
    streamer = Streamer(cycle=False)
    streamer.wrap(modules, tmp_path, "layers.")
    first = modules[0]
    assert first.module_key("proj.q.weight") == "proj.q.base_linear.weight"
    x = mx.random.normal((1, DIM))
    first(x)
    # Freed after the call: the base weight is the placeholder again, not the resident matrix.
    assert first.module.proj["q"].base_linear.weight.size == 1


def test_a_single_lora_layer_still_maps_to_its_linear_child(tmp_path):
    plain = [_Projection(nn.Linear(DIM, DIM, bias=False)) for _ in range(2)]
    mx.eval([block.parameters() for block in plain])
    _save(plain, tmp_path)
    modules = [_Projection(nn.Linear(DIM, DIM, bias=False)) for _ in range(2)]
    for module in modules:
        module.proj["q"] = LoRALinear.from_linear(module.proj["q"], r=2)
    Streamer(cycle=False).wrap(modules, tmp_path, "layers.")
    assert modules[0].module_key("proj.q.weight") == "proj.q.linear.weight"


@pytest.mark.parametrize("quantize", [4, 8])
def test_quantize_is_refused_before_any_weight_is_read(quantize):
    with pytest.raises(ValueError, match="-q/--quantize"):
        MOSSImageInitializer.init(model=object(), model_config=None, quantize=quantize, ssd_stream=True)


def test_bake_lora_with_a_lora_is_refused_before_any_weight_is_read():
    with pytest.raises(ValueError, match="--bake-lora"):
        MOSSImageInitializer.init(
            model=object(),
            model_config=None,
            quantize=None,
            lora_paths=["a.safetensors"],
            bake_lora=True,
            ssd_stream=True,
        )


def test_a_streamed_model_cannot_be_saved(tmp_path):
    model = MOSSImage.__new__(MOSSImage)
    model.ssd_stream = True
    with pytest.raises(ValueError, match="cannot be saved"):
        model.save_model(str(tmp_path / "out"))
    assert not (tmp_path / "out").exists()


def test_vae_tiling_options_are_reported_as_ignored_with_ssd_stream(monkeypatch):
    from mflux.cli.parser.parsers import CommandLineParser
    from mflux.models.moss_image.cli.moss_image_generate import SSD_STREAM_IGNORED_OPTIONS

    monkeypatch.setattr("sys.argv", ["mflux-generate-moss-image", "--ssd-stream", "--vae-tiling"])
    with pytest.warns(UserWarning, match="--vae-tiling is ignored"):
        CommandLineParser.warn_ignored_options(SSD_STREAM_IGNORED_OPTIONS)
