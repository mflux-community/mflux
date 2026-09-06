import pytest
from mlx import nn

from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
from tests.model_saving.tiny_checkpoint_helper import TinyCheckpointRoundtrip


class _TinyComponent(nn.Module):
    # Two Linears whose last dim is a multiple of 64, so the definition's predicate
    # quantizes them and the save/load path carries real quantized tensors.
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(64, 64)
        self.b = nn.Linear(64, 64)


class _SharedSubdirDefinition:
    # The SeedVR2 layout: two non-tokenizer components that both sit flat at repo root
    # (hf_subdir="."). On the real repo, weight_files tells them apart on load. On save,
    # both would write 0.safetensors and model.safetensors.index.json into the same
    # directory, so the second clobbers the first (issue #621).
    @staticmethod
    def get_components() -> list[ComponentDefinition]:
        return [
            ComponentDefinition(name="transformer", hf_subdir=".", loading_mode="mlx_native"),
            ComponentDefinition(name="vae", hf_subdir=".", loading_mode="mlx_native"),
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


class TestModelSaverSharedSubdir:
    @pytest.mark.fast
    def test_two_components_sharing_a_subdir_roundtrip_without_collision(self, tmp_path):
        # RED without the fix: transformer and vae both save into the same directory
        # (hf_subdir="."), so the second overwrites the first's shards and index, and on
        # reload the transformer comes back with the VAE's weights. GREEN once ModelSaver
        # and WeightLoader give each shared-subdir component its own <subdir>/<name>
        # directory. Issue #621.
        TinyCheckpointRoundtrip.save_and_reload_expecting_identical_weights(
            weight_definition=_SharedSubdirDefinition,
            make_components=lambda: {"transformer": _TinyComponent(), "vae": _TinyComponent()},
            base_path=tmp_path / "shared_subdir_tiny_q8",
            bits=8,
        )
