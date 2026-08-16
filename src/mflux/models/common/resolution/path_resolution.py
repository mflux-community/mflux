import json
import logging
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE

from mflux.models.common.resolution.actions import PathAction, Rule

logger = logging.getLogger(__name__)


class PathResolution:
    RULES = frozenset(
        {
            Rule(priority=0, name="none", check="is_none", action=PathAction.LOCAL),
            Rule(priority=1, name="local", check="exists_locally", action=PathAction.LOCAL),
            Rule(priority=2, name="hf_cached", check="is_hf_cached", action=PathAction.HUGGINGFACE_CACHED),
            Rule(priority=3, name="hf_download", check="is_hf_format", action=PathAction.HUGGINGFACE),
            Rule(priority=4, name="error", check="always", action=PathAction.ERROR),
        }
    )

    @staticmethod
    def resolve(path: str | None, patterns: list[str] | None = None) -> Path | None:
        if patterns is None:
            patterns = ["*.safetensors"]

        for rule in sorted(PathResolution.RULES, key=lambda r: r.priority):
            if PathResolution._check(rule.check, path, patterns):
                logger.debug(f"Path resolution: '{path}' → rule '{rule.name}' ({rule.action.value})")
                return PathResolution._execute(rule.action, path, patterns)

        raise ValueError(f"No rule matched for path: {path}")

    @staticmethod
    def _is_hf_format(path: str | None) -> bool:
        return path is not None and "/" in path and path.count("/") == 1 and not path.startswith(("./", "../", "~/"))

    @staticmethod
    def _check(check: str, path: str | None, patterns: list[str]) -> bool:
        if check == "is_none":
            return path is None
        if check == "exists_locally":
            if path is None:
                return False
            local_path = Path(path).expanduser()
            if not local_path.exists():
                return False
            # Warn if directory exists but contains no matching files
            if local_path.is_dir():
                has_matching_files = any(list(local_path.glob(p)) for p in patterns)
                if not has_matching_files:
                    print(
                        f"⚠️  Directory '{path}' exists but contains no files matching {patterns}. "
                        f"Model loading may fail."
                    )
            return True
        if check == "is_hf_cached":
            if not PathResolution._is_hf_format(path):
                return False
            # Check if we have a complete cached snapshot
            return PathResolution._find_complete_cached_snapshot(path, patterns) is not None
        if check == "is_hf_format":
            return PathResolution._is_hf_format(path)
        if check == "always":
            return True
        return False

    @staticmethod
    def _execute(action: PathAction, path: str | None, patterns: list[str]) -> Path | None:
        if action == PathAction.LOCAL:
            return Path(path).expanduser() if path else None
        if action == PathAction.HUGGINGFACE_CACHED:
            # Find the best complete cached snapshot
            cached_path = PathResolution._find_complete_cached_snapshot(path, patterns)
            if cached_path:
                return cached_path
            # Fallback to standard snapshot_download (shouldn't happen if _check passed)
            return Path(snapshot_download(repo_id=path, allow_patterns=patterns, local_files_only=True))
        if action == PathAction.HUGGINGFACE:
            print(f"Downloading model from HuggingFace: {path}...")
            return Path(snapshot_download(repo_id=path, allow_patterns=patterns))
        if action == PathAction.ERROR:
            raise FileNotFoundError(
                f"Model not found: '{path}'. "
                f"If local path, make sure it exists. "
                f"If HuggingFace repo, use 'org/model' format."
            )
        return None

    @staticmethod
    def _find_complete_cached_snapshot(repo_id: str | None, patterns: list[str]) -> Path | None:
        if repo_id is None:
            return None

        # Build the cache directory path for this repo
        # HuggingFace cache structure: {cache_dir}/models--{org}--{model}/snapshots/{revision}/
        repo_cache_name = f"models--{repo_id.replace('/', '--')}"
        repo_cache_dir = Path(HF_HUB_CACHE) / repo_cache_name / "snapshots"

        if not repo_cache_dir.exists():
            return None

        # Extract subdirectories that need safetensors files (e.g., "vae/*.safetensors" → "vae")
        required_subdirs = PathResolution._get_required_subdirs_with_safetensors(patterns)

        # Check each snapshot for completeness, prefer more recent ones
        snapshots = sorted(repo_cache_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)

        for snapshot_path in snapshots:
            if not snapshot_path.is_dir():
                continue
            if PathResolution._is_snapshot_complete(snapshot_path, required_subdirs, patterns):
                logger.debug(f"Found complete cached snapshot: {snapshot_path}")
                return snapshot_path

        return None

    @staticmethod
    def _get_required_subdirs_with_safetensors(patterns: list[str]) -> set[str]:
        subdirs = set()
        for pattern in patterns:
            # Only care about safetensors patterns
            if "*.safetensors" not in pattern:
                continue
            # Handle patterns like "vae/*.safetensors"
            if "/" in pattern:
                subdir = pattern.split("/")[0]
                # Only add if it's a real subdir name (not a glob pattern itself)
                if "*" not in subdir:
                    subdirs.add(subdir)
        return subdirs

    @staticmethod
    def _is_snapshot_complete(
        snapshot_path: Path, required_subdirs: set[str], patterns: list[str] | None = None
    ) -> bool:
        if not required_subdirs:
            # No specific subdirs required - check that all patterns are satisfied
            if patterns:
                if not PathResolution._patterns_satisfied(snapshot_path, patterns):
                    return False
                return PathResolution._indexed_shards_present(snapshot_path)
            else:
                # Fallback: just check for any safetensors
                if not any(snapshot_path.glob("**/*.safetensors")):
                    return False
                return PathResolution._indexed_shards_present(snapshot_path)

        # required_subdirs only collects patterns naming safetensors inside a fixed
        # subdirectory, so a weight file at the snapshot root is never checked below.
        # Krea 2 Turbo is that case: its transformer is the root-level turbo.safetensors.
        # Only safetensors count here. Download patterns are permissive filters, not a
        # manifest: FLUX.2 lists added_tokens.json and chat_template.jinja at the root
        # and ships both under tokenizer/.
        root_weight_patterns = [
            pattern for pattern in patterns or [] if "/" not in pattern and pattern.endswith(".safetensors")
        ]
        if root_weight_patterns and not PathResolution._patterns_satisfied(snapshot_path, root_weight_patterns):
            return False

        for subdir in required_subdirs:
            subdir_path = snapshot_path / subdir
            if not subdir_path.exists():
                return False
            # Check if subdir has at least one safetensors file (following symlinks)
            has_safetensors = False
            for f in subdir_path.iterdir():
                if f.name.endswith(".safetensors"):
                    # Verify the symlink target exists (handles broken symlinks)
                    if f.is_symlink():
                        if os.path.exists(f):
                            has_safetensors = True
                            break
                    else:
                        has_safetensors = True
                        break
            if not has_safetensors:
                return False
            if not PathResolution._indexed_shards_present(subdir_path):
                return False

        return True

    @staticmethod
    def _patterns_satisfied(snapshot_path: Path, patterns: list[str]) -> bool:
        # Every pattern has to match at least one file that is really on disk.
        for pattern in patterns:
            # Check if this specific pattern has any matches
            matches = list(snapshot_path.glob(pattern))
            if not matches:
                return False
            # Verify at least one match actually exists (handles broken symlinks)
            has_valid_match = False
            for match in matches:
                if match.is_symlink():
                    if os.path.exists(match):
                        has_valid_match = True
                        break
                else:
                    has_valid_match = True
                    break
            if not has_valid_match:
                return False
        return True

    @staticmethod
    def _indexed_shards_present(directory: Path) -> bool:
        """Whether every shard named by an index in `directory` is on disk.

        Indexed checkpoints are only complete when every referenced shard exists.
        One cached shard must not make a partial snapshot look complete and
        suppress Hugging Face's repair download. Directories without an index are
        complete by this measure, which is what unsharded checkpoints are.
        """
        for index_path in directory.glob("*.safetensors.index.json"):
            try:
                with index_path.open(encoding="utf-8") as index_file:
                    index = json.load(index_file)
                weight_map = index.get("weight_map") if isinstance(index, dict) else None
                if not isinstance(weight_map, dict) or not weight_map:
                    return False
                referenced_shards = set(weight_map.values())
                if not all(
                    isinstance(filename, str) and Path(filename).name == filename and (directory / filename).is_file()
                    for filename in referenced_shards
                ):
                    return False
            except (OSError, TypeError, ValueError, json.JSONDecodeError):  # noqa: PERF203
                return False
        return True
