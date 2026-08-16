import contextlib
from pathlib import Path
from typing import Callable
from unittest import mock

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten

from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.utils.version_util import VersionUtil


class TinyVAEStandIn(nn.Module):
    # The real VAEs are fixed-size (~170M param) architectures with no dimension
    # knobs. ModelSaver/WeightLoader treat every component as an opaque parameter
    # tree, so a small module walks the identical save/load code path.
    def __init__(self) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(3, 64, kernel_size=3)
        self.proj = nn.Linear(64, 64)


class TinyCheckpointRoundtrip:
    # Hermetic twin of the slow save/load tests: quantize tiny real components,
    # save them through ModelSaver (real shard/index/metadata writing), reload
    # through WeightLoader + WeightApplier (real mflux-format detection and
    # stored-quantization restore), and require the parameter trees to match
    # exactly. No downloads, no tokenizers, no image generation.

    # weight_definition is `type` rather than the src WeightDefinitionType alias:
    # that TYPE_CHECKING-only union has drifted (it omits ErnieWeightDefinition,
    # among others), and widening a src type alias is out of scope for tests.
    @staticmethod
    def save_and_reload_expecting_identical_weights(
        weight_definition: type,
        make_components: Callable[[], dict[str, nn.Module]],
        base_path: Path,
        bits: int = 8,
        tensors_per_shard: int | None = None,
    ) -> None:
        saved = make_components()
        # Many layers init deterministically (zeros/ones), which would let a
        # value-corrupting bug hide behind trivially-equal tensors; give every
        # parameter distinct random values so the equality check below has teeth.
        TinyCheckpointRoundtrip._randomize(saved, seed=0)
        TinyCheckpointRoundtrip._quantize(saved, weight_definition, bits)
        # ModelSaver._split_weights splits by size (>= 1 GB), so tiny tensors always
        # land in a single shard and the multi-shard save/load paths (index.json,
        # weight_map, cross-shard ordering) go unexercised. tensors_per_shard forces
        # a shard boundary every N tensors so those paths exist at test size.
        with TinyCheckpointRoundtrip._forced_sharding(tensors_per_shard):
            ModelSaver.save_model(
                model=TinyCheckpointRoundtrip._as_model(saved, weight_definition),
                bits=bits,
                base_path=str(base_path),
                weight_definition=weight_definition,
            )
        if tensors_per_shard is not None:
            shard_files = list(base_path.rglob("*.safetensors"))
            component_count = len(weight_definition.get_components())
            assert len(shard_files) > component_count, (
                f"expected multi-shard components, got {len(shard_files)} shards for {component_count} components"
            )

        loaded = WeightLoader.load(weight_definition=weight_definition, model_path=str(base_path))
        assert loaded.meta_data.quantization_level == bits
        assert loaded.meta_data.mflux_version == VersionUtil.get_mflux_version()

        # A different seed guarantees the fresh modules start from different
        # weights, so equality below proves the values came from disk.
        fresh = make_components()
        TinyCheckpointRoundtrip._randomize(fresh, seed=1)
        applied_bits = WeightApplier.apply_and_quantize(
            weights=loaded,
            models=fresh,
            quantize_arg=None,
            weight_definition=weight_definition,
        )
        assert applied_bits == bits

        for name, saved_module in saved.items():
            saved_weights = dict(tree_flatten(saved_module.parameters()))
            fresh_weights = dict(tree_flatten(fresh[name].parameters()))
            assert saved_weights.keys() == fresh_weights.keys(), f"{name}: key sets differ"
            for key, value in saved_weights.items():
                assert mx.array_equal(value, fresh_weights[key]), f"{name}.{key}: values differ after reload"

    @staticmethod
    def _forced_sharding(tensors_per_shard: int | None):
        if tensors_per_shard is None:
            return contextlib.nullcontext()

        def _split_every_n(weights: dict, max_file_size_gb: int = 2) -> list[dict]:
            items = list(weights.items())
            return [dict(items[i : i + tensors_per_shard]) for i in range(0, len(items), tensors_per_shard)]

        return mock.patch.object(ModelSaver, "_split_weights", staticmethod(_split_every_n))

    @staticmethod
    def _randomize(components: dict[str, nn.Module], seed: int) -> None:
        mx.random.seed(seed)
        for module in components.values():
            randomized = []
            for key, value in tree_flatten(module.parameters()):
                if mx.issubdtype(value.dtype, mx.floating):
                    randomized.append((key, mx.random.normal(value.shape).astype(value.dtype)))
                elif mx.issubdtype(value.dtype, mx.integer):
                    # e.g. Ideogram 4's Fp8Linear stores fp8 weights as uint8
                    randomized.append((key, mx.random.randint(0, 255, value.shape).astype(value.dtype)))
            module.update(tree_unflatten(randomized))

    @staticmethod
    def _quantize(components: dict[str, nn.Module], weight_definition: type, bits: int) -> None:
        group_size = getattr(weight_definition, "quantization_group_size", 64)
        for module in components.values():
            nn.quantize(
                module,
                group_size=group_size,
                bits=bits,
                class_predicate=lambda path, m: weight_definition.quantization_predicate(path, m),
            )

    @staticmethod
    def _as_model(components: dict[str, nn.Module], weight_definition: type):
        # ModelSaver reads components as attributes named by the definition
        # (model_attr, falling back to the component name) — same shape as the
        # real model objects whose save_model() delegates to ModelSaver.
        class _ComponentHolder:
            pass

        model = _ComponentHolder()
        for component in weight_definition.get_components():
            setattr(model, component.model_attr or component.name, components[component.name])
        return model
