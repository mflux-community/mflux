import json
from pathlib import Path

import pytest

from mflux.models.common.resolution.lora_resolution import LoraResolution
from mflux.utils.lora_library_util import LoraLibraryUtil


class _Library:
    @staticmethod
    def make(root: Path, *relative_paths: str) -> Path:
        for relative_path in relative_paths:
            (root / relative_path).parent.mkdir(parents=True, exist_ok=True)
            (root / relative_path).write_bytes(b"x" * 3)
        return root


class TestLibraryFiles:
    @pytest.mark.fast
    def test_two_files_with_one_name_are_both_listed(self, tmp_path):
        root = _Library.make(tmp_path, "flux3-klein/style.safetensors", "flux2-klein-4b/style.safetensors")

        files = LoraResolution.library_files([root])

        assert [(str(lora.relative_path), lora.selected) for lora in files] == [
            ("flux2-klein-4b/style.safetensors", True),
            ("flux3-klein/style.safetensors", False),
        ]
        assert LoraResolution.discover_files([root]) == {"style": (root / "flux2-klein-4b/style.safetensors").resolve()}

    @pytest.mark.fast
    def test_the_first_library_path_takes_a_name(self, tmp_path):
        first = _Library.make(tmp_path / "first", "z/style.safetensors")
        second = _Library.make(tmp_path / "second", "a/style.safetensors")

        assert LoraResolution.discover_files([first, second])["style"] == (first / "z/style.safetensors").resolve()
        assert LoraResolution.discover_files([second, first])["style"] == (second / "a/style.safetensors").resolve()

    @pytest.mark.fast
    def test_a_linked_file_keeps_its_path_in_the_library(self, tmp_path):
        target = _Library.make(tmp_path / "cache", "blob.safetensors") / "blob.safetensors"
        root = tmp_path / "loras"
        (root / "flux3-klein").mkdir(parents=True)
        (root / "flux3-klein" / "linked.safetensors").symlink_to(target)

        [lora] = LoraResolution.library_files([root])

        assert lora.relative_path == Path("flux3-klein/linked.safetensors")
        assert lora.path == target.resolve()

    @pytest.mark.fast
    def test_weight_shards_of_a_saved_model_are_not_loras(self, tmp_path):
        root = _Library.make(tmp_path, "model/transformer/0.safetensors", "style.safetensors")

        assert [lora.name for lora in LoraResolution.library_files([root])] == ["style"]

    @pytest.mark.fast
    def test_a_file_in_two_overlapping_library_paths_is_listed_once(self, tmp_path):
        root = _Library.make(tmp_path, "sub/style.safetensors")

        assert len(LoraResolution.library_files([root, root / "sub"])) == 1


class TestLibraryResolution:
    @pytest.mark.fast
    def test_a_name_of_two_files_takes_the_first_and_warns(self, tmp_path, monkeypatch, capsys):
        root = _Library.make(tmp_path, "b/style.safetensors", "a/style.safetensors")
        monkeypatch.setattr(LoraResolution, "_registry", {})
        monkeypatch.setattr(LoraResolution, "_same_name", {})
        monkeypatch.setenv("LORA_LIBRARY_PATH", str(root))
        LoraResolution._initialize_registry()

        result = LoraResolution.resolve("style")

        assert result == str((root / "a/style.safetensors").resolve())
        out = capsys.readouterr().out
        assert "names 2 files" in out and str((root / "b/style.safetensors").resolve()) in out

    @pytest.mark.fast
    def test_a_name_of_one_file_does_not_warn(self, tmp_path, monkeypatch, capsys):
        root = _Library.make(tmp_path, "a/style.safetensors")
        monkeypatch.setattr(LoraResolution, "_registry", {})
        monkeypatch.setattr(LoraResolution, "_same_name", {})
        monkeypatch.setenv("LORA_LIBRARY_PATH", str(root))
        LoraResolution._initialize_registry()

        LoraResolution.resolve("style")

        assert "⚠️" not in capsys.readouterr().out


class TestLibraryList:
    @pytest.mark.fast
    def test_the_list_shows_each_file_and_which_one_a_name_takes(self, tmp_path, capsys):
        root = _Library.make(tmp_path, "flux3-klein/style.safetensors", "flux2-klein-4b/style.safetensors")

        assert LoraLibraryUtil.list_loras(paths=[str(root)]) == 0

        out = capsys.readouterr().out
        assert "  flux2-klein-4b/style.safetensors -> style\n" in out
        assert "  flux3-klein/style.safetensors -> style  (not used: --lora style takes " in out
        assert "Total LoRA files found: 2" in out

    @pytest.mark.fast
    def test_json_gives_one_object_for_each_file(self, tmp_path, capsys):
        root = _Library.make(tmp_path, "flux3-klein/style.safetensors", "flux2-klein-4b/style.safetensors")

        assert LoraLibraryUtil.list_loras(paths=[str(root)], as_json=True) == 0

        assert json.loads(capsys.readouterr().out) == [
            {
                "name": "style",
                "path": str((root / "flux2-klein-4b/style.safetensors").resolve()),
                "root": str(root),
                "relative_path": "flux2-klein-4b/style.safetensors",
                "size_bytes": 3,
                "selected": True,
            },
            {
                "name": "style",
                "path": str((root / "flux3-klein/style.safetensors").resolve()),
                "root": str(root),
                "relative_path": "flux3-klein/style.safetensors",
                "size_bytes": 3,
                "selected": False,
            },
        ]

    @pytest.mark.fast
    def test_json_of_an_empty_library_is_an_empty_list(self, tmp_path, capsys):
        assert LoraLibraryUtil.list_loras(paths=[str(tmp_path)], as_json=True) == 0

        assert json.loads(capsys.readouterr().out) == []
