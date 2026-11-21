import json
from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten


class Streamer:
    def __init__(self, cycle: bool):
        # cycle: the block after the last one is the first (a transformer run once per step)
        self.cycle = cycle
        self.blocks = []
        self.pending = {}

    def wrap(self, modules: list[nn.Module], component_dir: Path, prefix: str, lora=None) -> None:
        index_file = component_dir / "model.safetensors.index.json"
        if not index_file.is_file():
            raise FileNotFoundError(
                f"--ssd-stream reads each layer by name from the saved shards, and {index_file} is missing. "
                "The model must be in the mflux layout: an `mflux-save` folder or a `mflux-community` repo."
            )
        weight_map = json.loads(index_file.read_text())["weight_map"]
        for index, module in enumerate(modules):
            block = StreamedBlock(
                module, f"{prefix}{index}.", weight_map, component_dir, self, len(self.blocks), index, lora
            )
            modules[index] = block
            self.blocks.append(block)

    def read(self, block: "StreamedBlock") -> list:
        # A fresh mx.load each time: an evaluated array stays in memory while anything references it.
        installed = []
        for shard, keys in block.shard_keys.items():
            loaded = mx.load(shard)
            installed.extend((block.module_key(key[len(block.prefix) :]), loaded[key]) for key in keys)
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


class StreamedBlock:
    # Created on first use, not at import: an import should not allocate an array.
    _placeholder = None

    @classmethod
    def placeholder(cls):
        if cls._placeholder is None:
            cls._placeholder = mx.zeros((1,))
        return cls._placeholder

    def __init__(self, module, prefix, weight_map, component_dir, streamer, position, index, lora=None):
        self.lora = lora
        self.index = index
        factor_names = [] if lora is None else lora.wrap(module, index)
        self.module = module
        self.prefix = prefix
        self.streamer = streamer
        self.position = position
        self.names = {name for name, _ in tree_flatten(module.parameters())}
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

    def __call__(self, *args, **kwargs):
        streamer = self.streamer
        installed = streamer.pending.pop(self, None) or streamer.read(self)
        # Whatever else was read ahead belongs to a block that this pass skipped (--uncond-skip-layers reads the
        # block after each one that runs, and the pass jumps over it). Drop it, or the skipped blocks pile up in memory.
        streamer.pending.clear()
        self.module.update(tree_unflatten(installed), strict=False)
        del installed
        try:
            out = self.module(*args, **kwargs)
            # Compute is dispatched before the next read so the two overlap.
            mx.async_eval(out)
            streamer.prefetch(self)
            mx.eval(out)
        finally:
            # Also when the call is interrupted: the weights must not stay installed on the module.
            self.module.update(self.empty, strict=False)
        return out
