import json
import logging
import struct
from functools import partial
from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten

logger = logging.getLogger(__name__)


class Streamer:
    def __init__(self, cycle: bool):
        # cycle: the block after the last one is the first (a transformer run once per step)
        self.cycle = cycle
        self.blocks = []
        self.pending = {}

    def wrap(
        self,
        modules: list[nn.Module],
        component_dir: Path,
        prefix: str | tuple[str, ...],
        lora=None,
        methods: tuple[str, ...] = (),
    ) -> None:
        # prefix: the saved name of the layer list. Of several, the first one that names layer 0 is used.
        # lora: a streamed LoRA that wraps the layers and supplies each block's factors.
        # methods: entry points of a layer other than __call__ that also run it streamed.
        weight_map = Streamer.weight_map(component_dir)
        prefix = Streamer._prefix_in(weight_map, prefix)
        if not any(key.startswith(f"{prefix}0.") for key in weight_map):
            logger.warning(
                f"--ssd-stream found no saved layer that starts with '{prefix}0.' in {component_dir}. "
                "The layers stay in memory and nothing is streamed."
            )
        for index, module in enumerate(modules):
            block = StreamedBlock(
                module, f"{prefix}{index}.", weight_map, component_dir, self, len(self.blocks), index, lora, methods
            )
            modules[index] = block
            self.blocks.append(block)

    @staticmethod
    def weight_map(component_dir: Path) -> dict[str, str]:
        # The shard of each saved name: from the index file of an mflux save, from the index file of a sharded
        # original checkpoint, or from the headers of the shards when there is no index (one-file checkpoints).
        indexes = [component_dir / "model.safetensors.index.json"]
        if not indexes[0].is_file():
            indexes = sorted(component_dir.glob("*.safetensors.index.json"))
        if indexes:
            return json.loads(indexes[0].read_text())["weight_map"]
        shards = sorted(p for p in component_dir.glob("*.safetensors") if not p.name.startswith("._"))
        if not shards:
            raise FileNotFoundError(
                f"--ssd-stream reads each layer by name from the saved shards, and {component_dir} has no "
                "safetensors file. The model must be in the mflux layout (an `mflux-save` folder or a "
                "`mflux-community` repo), or in an original layout whose weight names match the layers."
            )
        return {key: shard.name for shard in shards for key in Streamer._header_keys(shard)}

    def read(self, block: "StreamedBlock") -> list:
        # A fresh mx.load each time: an evaluated array stays in memory while anything references it.
        installed = []
        for shard, keys in block.shard_keys.items():
            loaded = mx.load(shard)
            installed.extend(block.installable(key[len(block.prefix) :], loaded[key]) for key in keys)
        if block.lora is not None:
            installed.extend(block.lora.factors(block.index))
        mx.async_eval([value for _, value in installed])
        return installed

    def prefetch(self, block: "StreamedBlock") -> None:
        ahead = block.position + 1
        if ahead == len(self.blocks):
            if not self.cycle:
                return
            ahead = 0
        following = self.blocks[ahead]
        if following not in self.pending:
            self.pending[following] = self.read(following)

    @staticmethod
    def _prefix_in(weight_map: dict[str, str], prefixes: str | tuple[str, ...]) -> str:
        prefixes = (prefixes,) if isinstance(prefixes, str) else prefixes
        # Without a match the layers keep the weights they loaded, as before: nothing is streamed.
        return next((p for p in prefixes if any(key.startswith(f"{p}0.") for key in weight_map)), prefixes[0])

    @staticmethod
    def _header_keys(shard: Path) -> list[str]:
        with shard.open("rb") as file:
            header = json.loads(file.read(struct.unpack("<Q", file.read(8))[0]))
        return [key for key in header if key != "__metadata__"]


class StreamedBlock:
    # Created on first use, not at import: an import should not allocate an array.
    _placeholder = None

    @classmethod
    def placeholder(cls):
        if cls._placeholder is None:
            cls._placeholder = mx.zeros((1,))
        return cls._placeholder

    def __init__(self, module, prefix, weight_map, component_dir, streamer, position, index, lora=None, methods=()):
        self.lora = lora
        self.index = index
        factor_names = [] if lora is None else lora.wrap(module, index)
        self.module = module
        self.prefix = prefix
        self.streamer = streamer
        self.position = position
        self.methods = methods
        # The dtype each parameter runs in, which a load-time cast (compute precision, a widened component) can
        # make differ from the saved one.
        self.dtypes = {name: value.dtype for name, value in tree_flatten(module.parameters())}
        self.names = set(self.dtypes)
        self.shard_keys = {}
        for key, shard in weight_map.items():
            if key.startswith(prefix):
                self.shard_keys.setdefault(str(component_dir / shard), []).append(key)
        stored = [self.module_key(key[len(prefix) :]) for keys in self.shard_keys.values() for key in keys]
        stored += factor_names
        self.empty = tree_unflatten([(name, self.placeholder()) for name in stored])
        module.update(self.empty, strict=False)

    def module_key(self, key: str) -> str:
        # A layer that a runtime LoRA wraps keeps the original linear layer as its `linear` child (one LoRA) or its
        # `base_linear` child (several LoRAs fused on the layer). Without a match the key is returned as it is.
        if key in self.names:
            return key
        parent, leaf = key.rsplit(".", 1)
        for child in ("linear", "base_linear"):
            if f"{parent}.{child}.{leaf}" in self.names:
                return f"{parent}.{child}.{leaf}"
        return key

    def installable(self, key: str, value: mx.array) -> tuple[str, mx.array]:
        name = self.module_key(key)
        dtype = self.dtypes.get(name, value.dtype)
        return name, value if value.dtype == dtype else value.astype(dtype)

    def __call__(self, *args, **kwargs):
        return self._run(self.module, *args, **kwargs)

    def __getattr__(self, name: str):
        # Only for names that the block itself does not have: the attributes of the layer, and its entry points.
        if name in ("module", "methods"):
            raise AttributeError(name)
        attribute = getattr(self.module, name)
        return partial(self._run, attribute) if name in self.methods else attribute

    def _run(self, function, *args, **kwargs):
        streamer = self.streamer
        installed = streamer.pending.pop(self, None) or streamer.read(self)
        # Whatever else was read ahead belongs to a block that this pass skipped (--uncond-skip-layers reads the
        # block after each one that runs, and the pass jumps over it). Drop it, or the skipped blocks pile up in memory.
        streamer.pending.clear()
        self.module.update(tree_unflatten(installed), strict=False)
        del installed
        try:
            out = function(*args, **kwargs)
            # Compute is dispatched before the next read so the two overlap.
            mx.async_eval(out)
            streamer.prefetch(self)
            mx.eval(out)
        finally:
            # Also when the call is interrupted: the weights must not stay installed on the module.
            self.module.update(self.empty, strict=False)
        return out
