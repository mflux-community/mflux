import mlx.core as mx

from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.lora.mapping.peft_lora_metadata import PeftLoRAMetadata
from mflux.models.moss_image.weights.moss_image_lora_mapping import MOSSImageLoRAMapping


class MOSSImageStreamedLoRA:
    # One LoRA file, kept on the SSD. Each block reads its own part of the file just before it runs and drops it after.
    # Nothing is cached on purpose: a cached tensor stays in memory, and the point is to hold one block of the file.
    # mx.load only reads the header of a safetensors file, so loading it for each block is cheap.

    def __init__(self, path: str, strength: float):
        self.path = path
        self.strength = strength
        tensors, metadata = mx.load(path, return_metadata=True)
        LoRALoader.reject_dora(tensors)
        self.configs = PeftLoRAMetadata.configs(metadata)
        self.applied = 0

    def wrap(self, module, block: int) -> list[str]:
        names = self._keys(block, mx.load(self.path))
        for name in names:
            # r=16 and random factors are only there to build the layer: the placeholder replaces them at once, and
            # the factors of the file are read when the block runs.
            module.proj[name] = LoRALinear.from_linear(module.proj[name])
        self.applied += len(names)
        return [f"proj.{name}.lora_{factor}" for name in names for factor in "AB"]

    def check_applied(self, blocks: int) -> None:
        # Like the loader, refuse a file that names no projection of this model, and a file that names a block the
        # model does not have (the rest of it would be applied, and that block silently dropped). A file that trains
        # only some projections, or some blocks, is fine.
        if self.applied == 0:
            raise ValueError(
                f"No LoRA layers were applied from {self.path}: it names no projection of this model. "
                "--stream-lora reads the same MOSS-Image LoRA key formats as the loader (kohya and PEFT names)."
            )
        mappings = LoRALoader._build_pattern_mappings(MOSSImageLoRAMapping.get_mapping())
        for key in mx.load(self.path):
            for mapping in mappings:
                block = LoRALoader._match_pattern(key, mapping.source_pattern)
                if block is not None and block >= blocks:
                    raise ValueError(
                        f"The LoRA file {self.path} names layer {block} ({key}), and this model has {blocks} layers."
                    )

    def factors(self, block: int) -> list:
        # A fresh load each time: an evaluated array stays in memory while anything references it.
        tensors = mx.load(self.path)
        factors = []
        for name, (down_key, up_key, alpha_key) in self._keys(block, tensors).items():
            rank = tensors[down_key].shape[0]
            alpha = float(tensors[alpha_key]) if alpha_key else PeftLoRAMetadata.alpha(down_key, self.configs, rank)
            scale = self.strength if alpha is None else self.strength * alpha / rank
            factors.append((f"proj.{name}.lora_A", tensors[down_key].T.astype(mx.bfloat16)))
            factors.append((f"proj.{name}.lora_B", (tensors[up_key].T * scale).astype(mx.bfloat16)))
        return factors

    def _keys(self, block: int, tensors: dict) -> dict[str, tuple]:
        # The projections of this block that the file trains, with the keys of their matrices, found by the
        # patterns that mflux's own loader uses for MOSS-Image.
        found = {}
        for target in MOSSImageLoRAMapping.get_mapping():
            down = self._find(tensors, target.possible_down_patterns, block)
            up = self._find(tensors, target.possible_up_patterns, block)
            if (down is None) != (up is None):
                # Like the loader, stop on half a pair: applying the rest would apply part of the adapter.
                raise ValueError(
                    f"The LoRA file {self.path} has only one of the two matrices for {target.model_path.format(block=block)}: "
                    f"{down or up}."
                )
            if down and up:
                found[target.model_path.rsplit(".", 1)[1]] = (
                    down,
                    up,
                    self._find(tensors, target.possible_alpha_patterns, block),
                )
        return found

    @staticmethod
    def _find(tensors: dict, patterns: list[str], block: int) -> str | None:
        found = []
        for pattern in patterns:
            key = pattern.format(block=block)
            for candidate in (key, key.removesuffix(".weight")):
                if candidate in tensors and candidate not in found:
                    found.append(candidate)
        if len(found) > 1:
            # Like the loader: two spellings of one matrix would fill the same slot, and one would be dropped.
            raise ValueError(f"The LoRA file has more than one key for the same matrix of block {block}: {found}")
        return found[0] if found else None
