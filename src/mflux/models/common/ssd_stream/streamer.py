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
        weight_map = json.loads((component_dir / "model.safetensors.index.json").read_text())["weight_map"]
        for index, module in enumerate(modules):
            block = StreamedBlock(module, f"{prefix}{index}.", weight_map, component_dir, self, len(self.blocks), lora)
            modules[index] = block
            self.blocks.append(block)

    def read(self, block: "StreamedBlock") -> list:
        # A fresh mx.load each time: an evaluated array stays in memory while anything references it.
        installed = []
        for shard, keys in block.shard_keys.items():
            loaded = mx.load(shard)
            installed.extend((block.module_key(key[len(block.prefix) :]), loaded[key]) for key in keys)
        if block.lora is not None:
            installed.extend(block.lora.factors(block.prefix))
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
    PLACEHOLDER = mx.zeros((1,))

    def __init__(self, module, prefix, weight_map, component_dir, streamer, position, lora=None):
        self.lora = lora
        factor_names = [] if lora is None else lora.wrap(module, prefix)
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
        self.empty = tree_unflatten([(name, self.PLACEHOLDER) for name in stored])
        module.update(self.empty, strict=False)

    def module_key(self, key: str) -> str:
        # A LoRA layer keeps the original linear layer as its `linear` child.
        if key in self.names:
            return key
        parent, leaf = key.rsplit(".", 1)
        return f"{parent}.linear.{leaf}"

    def __call__(self, *args, **kwargs):
        streamer = self.streamer
        installed = streamer.pending.pop(self, None) or streamer.read(self)
        self.module.update(tree_unflatten(installed), strict=False)
        del installed
        out = self.module(*args, **kwargs)
        # Compute is dispatched before the next read so the two overlap.
        mx.async_eval(out)
        streamer.prefetch(self)
        mx.eval(out)
        self.module.update(self.empty, strict=False)
        return out
