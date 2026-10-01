import shutil
from pathlib import Path

import pytest
from mlx import nn

from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
from mflux.models.common.weights.mapping.weight_mapping import WeightTarget
from tests.model_saving.tiny_checkpoint_helper import TinyCheckpointRoundtrip


class _TinyComponent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(64, 64)
        self.b = nn.Linear(64, 64)


# The variant's mapping expects the names of a different (diffusers-style) layout, as
# Krea 2's transformer/ variant does; an mflux-saved checkpoint carries none of them.
_DIFFUSERS_MAPPING = [
    WeightTarget(to_pattern="a.weight", from_pattern=["proj_in.weight"]),
    WeightTarget(to_pattern="b.weight", from_pattern=["proj_out.weight"]),
]


def _select_transformer_variant(root_path: Path) -> ComponentDefinition:
    # Krea 2's shape: the static component sits at the repo root (native single file);
    # when transformer/ exists instead, the diffusers variant lives there.
    if (root_path / "transformer").is_dir():
        return ComponentDefinition(
            name="transformer",
            hf_subdir="transformer",
            loading_mode="mlx_native",
            mapping_getter=lambda: _DIFFUSERS_MAPPING,
        )
    return ComponentDefinition(name="transformer", hf_subdir="", loading_mode="mlx_native")


class _VariantDefinition:
    @staticmethod
    def get_components() -> list[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="transformer",
                hf_subdir="",
                loading_mode="mlx_native",
                variant_selector=_select_transformer_variant,
            ),
            ComponentDefinition(name="vae", hf_subdir="vae", loading_mode="mlx_native"),
        ]

    @staticmethod
    def get_tokenizers() -> list:
        return []

    @staticmethod
    def get_download_patterns() -> list[str]:
        return ["**/*.safetensors"]

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        return isinstance(module, nn.Linear) and module.weight.shape[-1] % 64 == 0


def _move_root_shards_under(base_path: Path, subdir: str) -> None:
    # Earlier releases saved Krea 2's transformer into transformer/, and the first
    # published krea-2-turbo-mflux-q8 kept that layout. Rebuild it from a fresh save.
    target = base_path / subdir
    target.mkdir()
    for shard in list(base_path.glob("*.safetensors")) + [base_path / "model.safetensors.index.json"]:
        shutil.move(str(shard), str(target / shard.name))


class TestSavedCheckpointUnderVariantSubdir:
    @pytest.mark.fast
    def test_mflux_shards_under_the_variant_subdir_load(self, tmp_path):
        # RED before the fix: the loader probed only the static save subdir (the root),
        # found no shards there, fell through to the diffusers mapping for transformer/
        # and came back with nothing. Issue #784.
        TinyCheckpointRoundtrip.save_and_reload_expecting_identical_weights(
            weight_definition=_VariantDefinition,
            make_components=lambda: {"transformer": _TinyComponent(), "vae": _TinyComponent()},
            base_path=tmp_path / "variant_subdir_tiny_q8",
            bits=8,
            after_save=lambda base_path: _move_root_shards_under(base_path, "transformer"),
        )
