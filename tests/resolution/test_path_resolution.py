from unittest.mock import patch

import pytest

from mflux.models.common.resolution.path_resolution import PathResolution


class TestPathResolutionNone:
    @pytest.mark.fast
    def test_none_returns_none(self):
        result = PathResolution.resolve(path=None)
        assert result is None


class TestPathResolutionLocal:
    @pytest.mark.fast
    def test_existing_local_path(self, tmp_path):
        model_dir = tmp_path / "my-model"
        model_dir.mkdir()

        result = PathResolution.resolve(path=str(model_dir))

        assert result == model_dir

    @pytest.mark.fast
    def test_expands_home_directory(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        model_dir = tmp_path / "models" / "test"
        model_dir.mkdir(parents=True)

        result = PathResolution.resolve(path="~/models/test")

        assert result == model_dir

    @pytest.mark.fast
    def test_local_path_with_slash_preferred_over_huggingface(self, tmp_path):
        # Create a local path that looks like org/model
        org_dir = tmp_path / "org"
        org_dir.mkdir()
        model_dir = org_dir / "model"
        model_dir.mkdir()

        result = PathResolution.resolve(path=str(model_dir))

        assert result == model_dir

    @pytest.mark.fast
    def test_relative_path_not_treated_as_huggingface(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        model_dir = tmp_path / "org" / "model"
        model_dir.mkdir(parents=True)

        result = PathResolution.resolve(path="./org/model")

        assert result.name == "model"

    @pytest.mark.fast
    def test_tilde_path_not_treated_as_huggingface(self):
        # ~/org/model should NOT be treated as HuggingFace even if it doesn't exist
        # It should fail as a local path, not try to download from HF
        assert not PathResolution._is_hf_format("~/org/model")
        assert not PathResolution._is_hf_format("~/models/custom")

    @pytest.mark.fast
    def test_nonexistent_tilde_path_raises_local_error(self):
        # A tilde path that doesn't exist should raise FileNotFoundError
        # NOT try to download from HuggingFace
        with pytest.raises(FileNotFoundError) as exc_info:
            PathResolution.resolve(path="~/nonexistent/model")

        # Error should indicate it's a local path issue, not HF
        assert "Model not found" in str(exc_info.value)


class TestPathResolutionHuggingFace:
    @pytest.mark.fast
    @patch("mflux.models.common.resolution.path_resolution.snapshot_download")
    def test_huggingface_format_downloads_when_not_cached(self, mock_download, tmp_path):
        # No cache exists, so snapshot_download is called once to download
        mock_download.return_value = str(tmp_path / "cached")

        result = PathResolution.resolve(path="org/model")

        # Called once (download only - cache check uses _find_complete_cached_snapshot)
        assert mock_download.call_count == 1
        call_kwargs = mock_download.call_args[1]
        assert "local_files_only" not in call_kwargs
        assert result == tmp_path / "cached"

    @pytest.mark.fast
    def test_huggingface_uses_cache_when_available(self, tmp_path):
        # Create a fake cached snapshot structure
        repo_cache = tmp_path / "models--org--model" / "snapshots" / "abc123"
        repo_cache.mkdir(parents=True)
        (repo_cache / "model.safetensors").touch()

        with patch("mflux.models.common.resolution.path_resolution.HF_HUB_CACHE", str(tmp_path)):
            result = PathResolution.resolve(path="org/model")

        # Should return the cached path without calling snapshot_download
        assert result == repo_cache

    @pytest.mark.fast
    @patch("mflux.models.common.resolution.path_resolution.snapshot_download")
    def test_huggingface_passes_patterns(self, mock_download, tmp_path):
        mock_download.return_value = str(tmp_path / "cached")

        PathResolution.resolve(path="org/model", patterns=["*.bin", "*.json"])

        call_kwargs = mock_download.call_args[1]
        assert call_kwargs["allow_patterns"] == ["*.bin", "*.json"]


class TestPathResolutionError:
    @pytest.mark.fast
    def test_nonexistent_local_path_raises(self):
        with pytest.raises(FileNotFoundError) as exc_info:
            PathResolution.resolve(path="/nonexistent/path/to/model")

        assert "Model not found" in str(exc_info.value)
        assert "/nonexistent/path/to/model" in str(exc_info.value)

    @pytest.mark.fast
    def test_invalid_format_raises(self):
        with pytest.raises(FileNotFoundError) as exc_info:
            PathResolution.resolve(path="not-a-valid-path")

        assert "Model not found" in str(exc_info.value)

    @pytest.mark.fast
    def test_error_message_is_helpful(self):
        with pytest.raises(FileNotFoundError) as exc_info:
            PathResolution.resolve(path="/bad/path")

        error_msg = str(exc_info.value)
        assert "local path" in error_msg.lower()
        assert "org/model" in error_msg


class TestPathResolutionRules:
    @pytest.mark.fast
    def test_rules_are_checked_in_order(self, tmp_path):
        # Create a local path that also matches HuggingFace format
        model_dir = tmp_path / "org" / "model"
        model_dir.mkdir(parents=True)

        # Local should win because it's checked first
        result = PathResolution.resolve(path=str(model_dir))

        assert result == model_dir

    @pytest.mark.fast
    def test_relative_paths_are_local_not_huggingface(self):
        # ./org/model should NOT be treated as HuggingFace
        # It should fail because the local path doesn't exist
        with pytest.raises(FileNotFoundError):
            PathResolution.resolve(path="./org/model")

    @pytest.mark.fast
    def test_parent_relative_paths_are_local_not_huggingface(self):
        # ../org/model should NOT be treated as HuggingFace
        with pytest.raises(FileNotFoundError):
            PathResolution.resolve(path="../org/model")


class TestPathResolutionEmptyDirectory:
    @pytest.mark.fast
    def test_empty_directory_prints_warning(self, tmp_path, capsys):
        # Create an empty directory
        empty_model_dir = tmp_path / "empty-model"
        empty_model_dir.mkdir()

        result = PathResolution.resolve(path=str(empty_model_dir))

        # Should resolve (directory exists) but warn about missing files
        assert result == empty_model_dir
        captured = capsys.readouterr()
        assert "contains no files matching" in captured.out

    @pytest.mark.fast
    def test_directory_with_matching_files_no_warning(self, tmp_path, capsys):
        # Create a directory with matching files
        model_dir = tmp_path / "model-with-weights"
        model_dir.mkdir()
        (model_dir / "weights.safetensors").touch()

        result = PathResolution.resolve(path=str(model_dir))

        # Should resolve without warning
        assert result == model_dir
        captured = capsys.readouterr()
        assert "contains no files matching" not in captured.out

    @pytest.mark.fast
    def test_saved_checkpoint_in_component_subdirs_no_warning(self, tmp_path, capsys):
        # mflux-save writes SeedVR2's two parts to transformer/ and vae/, while the patterns
        # name the repo's flat files (#823).
        model_dir = tmp_path / "seedvr2-saved"
        for subdir in ("transformer", "vae"):
            (model_dir / subdir).mkdir(parents=True)
            (model_dir / subdir / "0.safetensors").touch()
        patterns = ["seedvr2_ema_3b_fp16.safetensors", "ema_vae_fp16.safetensors"]

        result = PathResolution.resolve(path=str(model_dir), patterns=patterns, saved_subdirs=["transformer", "vae"])

        assert result == model_dir
        assert "contains no files matching" not in capsys.readouterr().out

        # Without the save subdirs the same directory still warns.
        PathResolution.resolve(path=str(model_dir), patterns=patterns)
        assert "contains no files matching" in capsys.readouterr().out

    @pytest.mark.fast
    def test_half_copied_saved_checkpoint_still_warns(self, tmp_path, capsys):
        # The loader needs every part, so a save missing one of them is worth the warning.
        model_dir = tmp_path / "seedvr2-half"
        (model_dir / "transformer").mkdir(parents=True)
        (model_dir / "transformer" / "0.safetensors").touch()
        (model_dir / "vae").mkdir()

        PathResolution.resolve(
            path=str(model_dir),
            patterns=["seedvr2_ema_3b_fp16.safetensors", "ema_vae_fp16.safetensors"],
            saved_subdirs=["transformer", "vae"],
        )

        assert "contains no files matching" in capsys.readouterr().out

    @pytest.mark.fast
    def test_empty_directory_with_custom_patterns(self, tmp_path, capsys):
        # Create a directory with .bin file but looking for .json
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "weights.bin").touch()

        result = PathResolution.resolve(path=str(model_dir), patterns=["*.json"])

        # Should warn because no .json files found
        assert result == model_dir
        captured = capsys.readouterr()
        assert "contains no files matching" in captured.out
        assert "*.json" in captured.out
