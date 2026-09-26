from typing import List

import mlx.nn as nn

from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.seedvr2.weights.seedvr2_weight_mapping import SeedVR2WeightMapping


class SeedVR2WeightDefinition3B:
    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="transformer",
                hf_subdir=".",
                num_blocks=32,
                loading_mode="mlx_native",
                mapping_getter=lambda: SeedVR2WeightMapping.get_transformer_mapping(num_blocks=32),
                weight_files=["seedvr2_ema_3b_fp16.safetensors"],
            ),
            ComponentDefinition(
                name="vae",
                hf_subdir=".",
                num_blocks=4,
                loading_mode="mlx_native",
                mapping_getter=SeedVR2WeightMapping.get_vae_mapping,
                weight_files=["ema_vae_fp16.safetensors"],
            ),
        ]

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return []

    @staticmethod
    def get_download_patterns() -> List[str]:
        return [
            "seedvr2_ema_3b_fp16.safetensors",
            "ema_vae_fp16.safetensors",
        ]

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        return SeedVR2WeightDefinition.quantization_predicate(path, module)


class SeedVR2WeightDefinition7B:
    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="transformer",
                hf_subdir=".",
                num_blocks=36,
                loading_mode="mlx_native",
                mapping_getter=lambda: SeedVR2WeightMapping.get_transformer_mapping(
                    num_blocks=36, use_output_ada=False
                ),
                weight_files=["seedvr2_ema_7b_fp16.safetensors"],
            ),
            ComponentDefinition(
                name="vae",
                hf_subdir=".",
                num_blocks=4,
                loading_mode="mlx_native",
                mapping_getter=SeedVR2WeightMapping.get_vae_mapping,
                weight_files=["ema_vae_fp16.safetensors"],
            ),
        ]

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return []

    @staticmethod
    def get_download_patterns() -> List[str]:
        return [
            "seedvr2_ema_7b_fp16.safetensors",
            "ema_vae_fp16.safetensors",
        ]

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        return SeedVR2WeightDefinition.quantization_predicate(path, module)


class SeedVR2WeightDefinition:
    @staticmethod
    def resolve(model_config) -> type[SeedVR2WeightDefinition3B | SeedVR2WeightDefinition7B]:
        aliases = {a.lower() for a in getattr(model_config, "aliases", [])}
        if "seedvr2-7b" in aliases:
            return SeedVR2WeightDefinition7B
        return SeedVR2WeightDefinition3B

    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return SeedVR2WeightDefinition3B.get_components()

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return []

    @staticmethod
    def get_download_patterns() -> List[str]:
        return SeedVR2WeightDefinition3B.get_download_patterns()

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        if isinstance(module, (nn.Conv2d, nn.Conv3d)):
            return False

        if not hasattr(module, "to_quantized"):
            return False

        if isinstance(module, nn.Linear):
            if hasattr(module, "weight") and module.weight.shape[-1] % 64 != 0:
                return False

        return True
