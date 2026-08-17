import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from mflux.models.common.lora.mapping.lora_saver import LoRASaver
from mflux.utils.version_util import VersionUtil

if TYPE_CHECKING:
    from mflux.models.common.weights.loading.weight_definition import WeightDefinitionType


class ModelSaver:
    @staticmethod
    def save_model(
        model: Any,
        bits: int,
        base_path: str,
        weight_definition: "WeightDefinitionType",
    ) -> None:
        # Bake and strip any LoRA wrappers (to avoid duplicating shared weights) across
        # every component before writing a single file: baking raises when an adapter does
        # not fit, and that must abort before part of a checkpoint is on disk.
        components = []
        for component_def in weight_definition.get_components():
            attr_name = component_def.model_attr or component_def.name
            component = getattr(model, attr_name, None)
            if component is not None:
                LoRASaver.bake_and_strip_lora(component)
                components.append((component, component_def.hf_subdir))

        # Save tokenizers from model.tokenizers dict
        tokenizer_defs = weight_definition.get_tokenizers()
        for t in tokenizer_defs:
            if hasattr(model, "tokenizers") and t.name in model.tokenizers:
                tokenizer_wrapper = model.tokenizers[t.name]
                if hasattr(tokenizer_wrapper, "tokenizer"):
                    ModelSaver._save_tokenizer(base_path, tokenizer_wrapper.tokenizer, t.hf_subdir)

        # Save model components with progress bar
        for component, subdir in tqdm(components, desc="Saving components", unit="component"):
            ModelSaver._save_weights(base_path, bits, component, subdir)

    @staticmethod
    def _save_tokenizer(base_path: str, tokenizer: PreTrainedTokenizer, subdir: str) -> None:
        path = Path(base_path) / subdir
        path.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(path)

    @staticmethod
    def _save_weights(base_path: str, bits: int, model: nn.Module, subdir: str) -> None:
        path = Path(base_path) / subdir
        path.mkdir(parents=True, exist_ok=True)
        weights = dict(tree_flatten(model.parameters()))
        shards = ModelSaver._split_weights(weights)

        # Build weight_map for index.json (maps each weight key to its shard file)
        weight_map = {}
        shard_iter = tqdm(enumerate(shards), total=len(shards), desc=f"  {subdir}", unit="shard", leave=False)
        for i, shard in shard_iter:
            shard_filename = f"{i}.safetensors"
            mx.save_safetensors(
                str(path / shard_filename),
                shard,
                {
                    "quantization_level": str(bits),
                    "mflux_version": VersionUtil.get_mflux_version(),
                },
            )
            # Record which file each weight belongs to
            for key in shard.keys():
                weight_map[key] = shard_filename

        # Write model.safetensors.index.json for HuggingFace compatibility
        # This ensures the saved model works even if custom metadata is stripped
        index_data = {
            "metadata": {
                "quantization_level": str(bits),
                "mflux_version": VersionUtil.get_mflux_version(),
            },
            "weight_map": weight_map,
        }
        with open(path / "model.safetensors.index.json", "w") as f:
            json.dump(index_data, f, indent=2)

    @staticmethod
    def _split_weights(weights: dict, max_file_size_gb: int = 2) -> list[dict]:
        max_file_size_bytes = max_file_size_gb << 30
        shards: list[dict] = []
        shard: dict = {}
        shard_size = 0
        for k, v in weights.items():
            if shard_size + v.nbytes > max_file_size_bytes:
                shards.append(shard)
                shard, shard_size = {}, 0
            shard[k] = v
            shard_size += v.nbytes
        if shard:  # Don't append empty shard
            shards.append(shard)
        return shards
