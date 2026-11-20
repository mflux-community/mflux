import mlx.core as mx

from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.lora.mapping.peft_lora_metadata import PeftLoRAMetadata
from mflux.models.moss_image.weights.moss_image_lora_mapping import MOSSImageLoRAMapping


class MOSSImageStreamedLoRA:
    def __init__(self, path: str, strength: float):
        self.path = path
        self.strength = strength
        self.configs = PeftLoRAMetadata.configs(mx.load(path, return_metadata=True)[1])

    def wrap(self, module, prefix: str) -> list[str]:
        names = self._keys(prefix, mx.load(self.path))
        for name in names:
            module.proj[name] = LoRALinear.from_linear(module.proj[name])
        return [f"proj.{name}.lora_{factor}" for name in names for factor in "AB"]

    def factors(self, prefix: str) -> list:
        # A fresh load each time: an evaluated array stays in memory while anything references it.
        tensors = mx.load(self.path)
        factors = []
        for name, (down_key, up_key, alpha_key) in self._keys(prefix, tensors).items():
            rank = tensors[down_key].shape[0]
            alpha = float(tensors[alpha_key]) if alpha_key else PeftLoRAMetadata.alpha(down_key, self.configs, rank)
            scale = self.strength if alpha is None else self.strength * alpha / rank
            factors.append((f"proj.{name}.lora_A", tensors[down_key].T.astype(mx.bfloat16)))
            factors.append((f"proj.{name}.lora_B", (tensors[up_key].T * scale).astype(mx.bfloat16)))
        return factors

    def _keys(self, prefix: str, tensors: dict) -> dict[str, tuple]:
        # The projections of this block that the file trains, with the keys of their matrices, found by the
        # patterns that mflux's own loader uses for MOSS-Image.
        block = int(prefix.split(".")[1])
        found = {}
        for target in MOSSImageLoRAMapping.get_mapping():
            down = self._find(tensors, target.possible_down_patterns, block)
            up = self._find(tensors, target.possible_up_patterns, block)
            if down and up:
                found[target.model_path.rsplit(".", 1)[1]] = (
                    down,
                    up,
                    self._find(tensors, target.possible_alpha_patterns, block),
                )
        return found

    @staticmethod
    def _find(tensors: dict, patterns: list[str], block: int) -> str | None:
        for pattern in patterns:
            key = pattern.format(block=block)
            for candidate in (key, key.removesuffix(".weight")):
                if candidate in tensors:
                    return candidate
        return None
