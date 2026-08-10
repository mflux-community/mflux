from unittest.mock import patch

import pytest
from huggingface_hub.utils import LocalEntryNotFoundError

from mflux.models.common.resolution.lora_resolution import LoraResolution


class TestLoraResolutionLocal:
    @pytest.mark.fast
    def test_existing_local_file(self, tmp_path):
        lora_file = tmp_path / "my-lora.safetensors"
        lora_file.touch()

        result = LoraResolution.resolve(path=str(lora_file))

        assert result == str(lora_file)


class TestLoraResolutionRegistry:
    @pytest.mark.fast
    def test_registry_lookup(self, tmp_path):
        lora_file = tmp_path / "registered-lora.safetensors"
        lora_file.touch()

        original_registry = LoraResolution._registry.copy()
        try:
            LoraResolution._registry["my-alias"] = lora_file

            result = LoraResolution.resolve(path="my-alias")

            assert result == str(lora_file)
        finally:
            LoraResolution._registry = original_registry


class TestLoraResolutionHuggingFace:
    @pytest.mark.fast
    @patch("mflux.models.common.resolution.lora_resolution.snapshot_download")
    def test_huggingface_repo_downloads_when_not_cached(self, mock_download, tmp_path):
        lora_file = tmp_path / "lora.safetensors"
        lora_file.touch()
        # First call (cache check) raises, second call (download) succeeds
        mock_download.side_effect = [
            LocalEntryNotFoundError("Not cached"),
            str(tmp_path),
        ]

        result = LoraResolution.resolve(path="org/lora-repo")

        assert mock_download.call_count == 2
        # First call should have local_files_only=True
        first_call = mock_download.call_args_list[0]
        assert first_call[1].get("local_files_only") is True
        assert result == str(lora_file)

    @pytest.mark.fast
    @patch("mflux.models.common.resolution.lora_resolution.snapshot_download")
    def test_huggingface_repo_uses_cache_when_available(self, mock_download, tmp_path):
        lora_file = tmp_path / "lora.safetensors"
        lora_file.touch()
        # Cache check succeeds - no network download needed
        mock_download.return_value = str(tmp_path)

        result = LoraResolution.resolve(path="org/lora-repo")

        # Called twice with local_files_only=True (once in check, once in execute)
        assert mock_download.call_count == 2
        for call in mock_download.call_args_list:
            assert call[1].get("local_files_only") is True
        assert result == str(lora_file)

    @pytest.mark.fast
    def test_collection_format_is_detected(self):
        # Collection format: "repo:filename" with a "/" in repo
        assert LoraResolution._is_collection_format("org/repo:file.safetensors")
        assert not LoraResolution._is_collection_format("org/repo")
        assert not LoraResolution._is_collection_format("local:file")

    @pytest.mark.fast
    @patch("mflux.models.common.resolution.lora_resolution.snapshot_download")
    def test_multiple_safetensors_in_repo_raises_error(self, mock_download, tmp_path):
        # Create multiple .safetensors files in the directory
        (tmp_path / "lora_v1.safetensors").touch()
        (tmp_path / "lora_v2.safetensors").touch()

        # First call (cache check) raises, second call (download) succeeds
        mock_download.side_effect = [
            LocalEntryNotFoundError("Not cached"),
            str(tmp_path),
        ]

        with pytest.raises(ValueError) as exc_info:
            LoraResolution.resolve(path="org/multi-lora-repo")

        error_msg = str(exc_info.value)
        assert "Multiple .safetensors files found" in error_msg
        assert "org/multi-lora-repo:lora_v1.safetensors" in error_msg
        assert "org/multi-lora-repo:lora_v2.safetensors" in error_msg

    @pytest.mark.fast
    @patch("mflux.models.common.resolution.lora_resolution.snapshot_download")
    def test_multiple_safetensors_cached_raises_error(self, mock_download, tmp_path):
        # Create multiple .safetensors files in the directory
        (tmp_path / "lora_a.safetensors").touch()
        (tmp_path / "lora_b.safetensors").touch()

        # Cache check succeeds (both calls return same path)
        mock_download.return_value = str(tmp_path)

        with pytest.raises(ValueError) as exc_info:
            LoraResolution.resolve(path="org/cached-multi-lora")

        assert "Multiple .safetensors files found" in str(exc_info.value)

    @pytest.mark.fast
    @patch("mflux.models.common.resolution.lora_resolution.snapshot_download")
    def test_single_safetensor_in_repo_succeeds(self, mock_download, tmp_path):
        # Create only one .safetensors file
        lora_file = tmp_path / "single-lora.safetensors"
        lora_file.touch()

        mock_download.side_effect = [
            LocalEntryNotFoundError("Not cached"),
            str(tmp_path),
        ]

        result = LoraResolution.resolve(path="org/single-lora-repo")

        assert result == str(lora_file)

    @pytest.mark.fast
    def test_collection_downloads_when_sibling_file_is_cached(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "loras"
        cache_dir.mkdir()
        monkeypatch.setattr(
            "mflux.models.common.resolution.lora_resolution.MFLUX_LORA_CACHE_DIR",
            cache_dir,
        )

        snapshot_dir = tmp_path / "snapshot"
        snapshot_dir.mkdir()
        requested = snapshot_dir / "style-b.safetensors"
        requested.touch()
        (snapshot_dir / "style-a.safetensors").touch()

        with patch("mflux.models.common.resolution.lora_resolution.snapshot_download") as mock_download:
            mock_download.side_effect = [
                LocalEntryNotFoundError("sibling cached, requested file missing"),
                str(snapshot_dir),
            ]

            result = LoraResolution.resolve(path="org/styles:style-b.safetensors")

        assert result == str(cache_dir / "style-b.safetensors")
        assert mock_download.call_count == 2
        assert mock_download.call_args_list[0][1]["allow_patterns"] == ["style-b.safetensors"]
        assert mock_download.call_args_list[0][1]["local_files_only"] is True
        assert mock_download.call_args_list[1][1]["local_files_only"] is False

    @pytest.mark.fast
    def test_collection_uses_mflux_cache_symlink(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "loras"
        cache_dir.mkdir()
        monkeypatch.setattr(
            "mflux.models.common.resolution.lora_resolution.MFLUX_LORA_CACHE_DIR",
            cache_dir,
        )

        cached = cache_dir / "cached-style.safetensors"
        cached.write_bytes(b"x")

        with patch("mflux.models.common.resolution.lora_resolution.snapshot_download") as mock_download:
            result = LoraResolution.resolve(path="org/styles:cached-style.safetensors")

        assert result == str(cached)
        mock_download.assert_not_called()


class TestLoraResolutionError:
    @pytest.mark.fast
    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError) as exc_info:
            LoraResolution.resolve(path="nonexistent-lora")

        assert "LoRA file not found" in str(exc_info.value)


class TestLoraResolutionPaths:
    @pytest.mark.fast
    def test_resolve_paths_empty(self):
        result = LoraResolution.resolve_paths(paths=None)
        assert result == []

    @pytest.mark.fast
    def test_resolve_paths_filters_invalid(self, tmp_path):
        valid_lora = tmp_path / "valid.safetensors"
        valid_lora.touch()

        result = LoraResolution.resolve_paths(paths=[str(valid_lora), "invalid-path"])

        assert len(result) == 1
        assert result[0] == str(valid_lora)


class TestLoraResolutionScales:
    @pytest.mark.fast
    def test_resolve_scales_none(self):
        result = LoraResolution.resolve_scales(scales=None, num_paths=3)
        assert result == [1.0, 1.0, 1.0]

    @pytest.mark.fast
    def test_resolve_scales_provided(self):
        result = LoraResolution.resolve_scales(scales=[0.5, 0.8], num_paths=2)
        assert result == [0.5, 0.8]

    @pytest.mark.fast
    def test_resolve_scales_too_few_pads_with_default(self, capsys):
        # 2 scales provided but 3 paths - should pad with 1.0
        result = LoraResolution.resolve_scales(scales=[0.5, 0.8], num_paths=3)

        assert result == [0.5, 0.8, 1.0]
        captured = capsys.readouterr()
        assert "doesn't match" in captured.out

    @pytest.mark.fast
    def test_resolve_scales_too_many_truncates(self, capsys):
        # 3 scales provided but only 2 paths - should truncate
        result = LoraResolution.resolve_scales(scales=[0.5, 0.8, 0.9], num_paths=2)

        assert result == [0.5, 0.8]
        captured = capsys.readouterr()
        assert "doesn't match" in captured.out

    @pytest.mark.fast
    def test_resolve_scales_matching_no_warning(self, capsys):
        # Matching counts should not warn
        result = LoraResolution.resolve_scales(scales=[0.5, 0.8], num_paths=2)

        assert result == [0.5, 0.8]
        captured = capsys.readouterr()
        assert "doesn't match" not in captured.out


class TestLoraResolutionRules:
    @pytest.mark.fast
    def test_local_takes_priority_over_registry(self, tmp_path):
        lora_file = tmp_path / "lora.safetensors"
        lora_file.touch()

        original_registry = LoraResolution._registry.copy()
        try:
            LoraResolution._registry[str(lora_file)] = tmp_path / "other.safetensors"

            result = LoraResolution.resolve(path=str(lora_file))

            assert result == str(lora_file)
        finally:
            LoraResolution._registry = original_registry


class TestLoraResolutionTildeExpansion:
    @pytest.mark.fast
    def test_expands_home_directory(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        lora_dir = tmp_path / "loras"
        lora_dir.mkdir()
        lora_file = lora_dir / "my-style.safetensors"
        lora_file.touch()

        result = LoraResolution.resolve(path="~/loras/my-style.safetensors")

        assert result == str(lora_file)

    @pytest.mark.fast
    def test_tilde_path_not_treated_as_huggingface(self, tmp_path, monkeypatch):
        # ~/loras/file has exactly one slash after tilde expansion path check
        # It should NOT be treated as HuggingFace format
        monkeypatch.setenv("HOME", str(tmp_path))
        lora_file = tmp_path / "org" / "lora.safetensors"
        lora_file.parent.mkdir(parents=True)
        lora_file.touch()

        # This path looks like "~/org/lora.safetensors" which could be confused with HF format
        result = LoraResolution.resolve(path="~/org/lora.safetensors")

        assert result == str(lora_file)


class TestLoraResolutionRelativePaths:
    @pytest.mark.fast
    def test_relative_path_not_treated_as_huggingface(self):
        # ./org/model should NOT be treated as HuggingFace even if it doesn't exist
        # It should fail as a local path, not try to download from HF
        assert not LoraResolution._is_hf_format("./org/model")
        assert not LoraResolution._is_hf_format("../org/model")

    @pytest.mark.fast
    def test_dotslash_single_segment_not_huggingface(self):
        # ./lora has exactly one slash - should NOT match HuggingFace format
        assert not LoraResolution._is_hf_format("./lora")
        assert not LoraResolution._is_hf_format("../lora")

    @pytest.mark.fast
    def test_relative_path_existing_file_resolves_locally(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        lora_dir = tmp_path / "subdir"
        lora_dir.mkdir()
        lora_file = lora_dir / "style.safetensors"
        lora_file.touch()

        result = LoraResolution.resolve(path="./subdir/style.safetensors")

        assert "style.safetensors" in result
