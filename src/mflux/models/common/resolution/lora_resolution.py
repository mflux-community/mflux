import logging
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError

from mflux.cli.defaults.defaults import MFLUX_LORA_CACHE_DIR
from mflux.models.common.resolution.actions import LoraAction, Rule

logger = logging.getLogger(__name__)


class LoraResolution:
    RULES = frozenset(
        {
            Rule(priority=0, name="local", check="exists_locally", action=LoraAction.LOCAL),
            Rule(priority=1, name="registry", check="in_registry", action=LoraAction.REGISTRY),
            Rule(priority=2, name="collection_cached", check="is_collection_cached", action=LoraAction.HUGGINGFACE_COLLECTION_CACHED),
            Rule(priority=3, name="collection_download", check="is_collection_format", action=LoraAction.HUGGINGFACE_COLLECTION),
            Rule(priority=4, name="repo_cached", check="is_repo_cached", action=LoraAction.HUGGINGFACE_REPO_CACHED),
            Rule(priority=5, name="repo_download", check="is_hf_format", action=LoraAction.HUGGINGFACE_REPO),
            Rule(priority=6, name="error", check="always", action=LoraAction.ERROR),
        }
    )  # fmt: off

    _registry: dict[str, Path] = {}

    @staticmethod
    def resolve(path: str) -> str:
        for rule in sorted(LoraResolution.RULES, key=lambda r: r.priority):
            if LoraResolution._check(rule.check, path):
                logger.debug(f"LoRA resolution: '{path}' → rule '{rule.name}' ({rule.action.value})")
                return LoraResolution._execute(rule.action, path)

        raise ValueError(f"No rule matched for LoRA path: {path}")

    @staticmethod
    def resolve_paths(paths: list[str] | None) -> list[str]:
        if not paths:
            return []
        return [r for path in paths if (r := LoraResolution._try_resolve(path)) is not None]

    @staticmethod
    def _try_resolve(path: str) -> str | None:
        try:
            return LoraResolution.resolve(path)
        except FileNotFoundError as e:
            print(f"⚠️  {e}")
            return None

    @staticmethod
    def resolve_scales(scales: list[float] | None, num_paths: int) -> list[float]:
        if not scales:
            return [1.0] * num_paths
        if len(scales) != num_paths:
            print(
                f"⚠️  Number of LoRA scales ({len(scales)}) doesn't match number of LoRA paths ({num_paths}). "
                f"Using provided scales and defaulting remaining to 1.0."
            )
            # Pad with 1.0 if too few scales, truncate if too many
            if len(scales) < num_paths:
                return list(scales) + [1.0] * (num_paths - len(scales))
            return list(scales[:num_paths])
        return scales

    @staticmethod
    def _is_collection_format(path: str) -> bool:
        # Supported forms:
        # - HuggingFace collection-like: "org/repo:filename.safetensors"
        # - ai-toolkit style (explicit): "hf:org/repo/filename.safetensors"
        if path.startswith("hf:"):
            remainder = path[3:]
            if ":" in remainder and "/" in remainder:
                return True
            if remainder.count("/") == 2 and not remainder.startswith(("./", "../", "~/")):
                return True
            return False
        if ":" in path and "/" in path:
            return True
        return False

    @staticmethod
    def _split_collection_path(path: str) -> tuple[str, str]:
        if path.startswith("hf:"):
            path = path[3:]
        if ":" in path:
            repo_id, filename = path.split(":", 1)
            return repo_id, filename
        # ai-toolkit style: org/repo/filename
        repo_id, filename = path.rsplit("/", 1)
        return repo_id, filename

    @staticmethod
    def _is_hf_format(path: str) -> bool:
        return "/" in path and path.count("/") == 1 and not path.startswith(("./", "../", "~/"))

    @staticmethod
    def _check(check: str, path: str) -> bool:
        if check == "exists_locally":
            return Path(path).expanduser().exists()
        if check == "in_registry":
            return path in LoraResolution._registry
        if check == "is_collection_cached":
            if not LoraResolution._is_collection_format(path):
                return False
            repo_id, filename = LoraResolution._split_collection_path(path)
            return LoraResolution._is_collection_in_cache(repo_id, filename)
        if check == "is_collection_format":
            return LoraResolution._is_collection_format(path)
        if check == "is_repo_cached":
            if not LoraResolution._is_hf_format(path):
                return False
            return LoraResolution._is_repo_in_cache(path)
        if check == "is_hf_format":
            return LoraResolution._is_hf_format(path)
        if check == "always":
            return True
        return False

    @staticmethod
    def _is_collection_in_cache(repo_id: str, filename: str) -> bool:
        return LoraResolution._find_collection_file(repo_id, filename, local_files_only=True) is not None

    @staticmethod
    def _linked_collection_path(filename: str) -> str | None:
        cache_path = MFLUX_LORA_CACHE_DIR
        cached_file_path = cache_path / filename
        if not cached_file_path.is_file():
            return None
        try:
            with open(cached_file_path, "rb") as f:
                f.read(1)
            return str(cached_file_path)
        except OSError:
            return None

    @staticmethod
    def _find_collection_file(repo_id: str, filename: str, *, local_files_only: bool) -> str | None:
        cache_path = MFLUX_LORA_CACHE_DIR
        cache_path.mkdir(parents=True, exist_ok=True)

        linked = LoraResolution._linked_collection_path(filename)
        if linked is not None:
            return linked

        try:
            download_path = Path(
                snapshot_download(
                    repo_id=repo_id,
                    allow_patterns=[filename],
                    cache_dir=str(cache_path),
                    local_files_only=local_files_only,
                )
            )
            return LoraResolution._find_and_link_file(download_path, filename, cache_path)
        except (LocalEntryNotFoundError, FileNotFoundError):
            return None

    @staticmethod
    def _is_repo_in_cache(repo_id: str) -> bool:
        cache_path = MFLUX_LORA_CACHE_DIR
        try:
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=["*.safetensors"],
                cache_dir=str(cache_path),
                local_files_only=True,
            )
            return True
        except LocalEntryNotFoundError:
            return False

    @staticmethod
    def _execute(action: LoraAction, path: str) -> str:
        if action == LoraAction.LOCAL:
            return str(Path(path).expanduser())
        if action == LoraAction.REGISTRY:
            return str(LoraResolution._registry[path])
        if action == LoraAction.HUGGINGFACE_COLLECTION_CACHED:
            repo_id, filename = LoraResolution._split_collection_path(path)
            return LoraResolution._load_collection_from_cache(repo_id, filename)
        if action == LoraAction.HUGGINGFACE_COLLECTION:
            repo_id, filename = LoraResolution._split_collection_path(path)
            return LoraResolution._download_collection(repo_id, filename)
        if action == LoraAction.HUGGINGFACE_REPO_CACHED:
            return LoraResolution._load_repo_from_cache(path)
        if action == LoraAction.HUGGINGFACE_REPO:
            return LoraResolution._download_repo(path)
        if action == LoraAction.ERROR:
            raise FileNotFoundError(
                f"LoRA file not found: '{path}'. File does not exist and is not in the LoRA library."
            )
        raise ValueError(f"Unknown action: {action}")

    @staticmethod
    def _load_repo_from_cache(repo_id: str) -> str:
        cache_path = MFLUX_LORA_CACHE_DIR
        cache_path.mkdir(parents=True, exist_ok=True)

        download_path = Path(
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=["*.safetensors"],
                cache_dir=str(cache_path),
                local_files_only=True,
            )
        )

        safetensor_files = list(download_path.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"No .safetensors file found in cached repo: {repo_id}")

        if len(safetensor_files) > 1:
            file_names = sorted([f.name for f in safetensor_files])
            file_list = "\n".join(f"  - {repo_id}:{name}" for name in file_names)
            raise ValueError(
                f"Multiple .safetensors files found in '{repo_id}'. Please specify which file to use:\n{file_list}"
            )

        return str(safetensor_files[0])

    @staticmethod
    def _download_repo(repo_id: str) -> str:
        cache_path = MFLUX_LORA_CACHE_DIR
        cache_path.mkdir(parents=True, exist_ok=True)

        print(f"Downloading LoRA from HuggingFace: {repo_id}...")
        download_path = Path(
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=["*.safetensors"],
                cache_dir=str(cache_path),
            )
        )

        safetensor_files = list(download_path.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"No .safetensors file found in HuggingFace repo: {repo_id}")

        if len(safetensor_files) > 1:
            file_names = sorted([f.name for f in safetensor_files])
            file_list = "\n".join(f"  - {repo_id}:{name}" for name in file_names)
            raise ValueError(
                f"Multiple .safetensors files found in '{repo_id}'. Please specify which file to use:\n{file_list}"
            )

        return str(safetensor_files[0])

    @staticmethod
    def _load_collection_from_cache(repo_id: str, filename: str) -> str:
        cached = LoraResolution._find_collection_file(repo_id, filename, local_files_only=True)
        if cached is not None:
            return cached
        return LoraResolution._download_collection(repo_id, filename)

    @staticmethod
    def _download_collection(repo_id: str, filename: str) -> str:
        cache_path = MFLUX_LORA_CACHE_DIR
        cache_path.mkdir(parents=True, exist_ok=True)

        print(f"Downloading LoRA '{filename}' from {repo_id}...")
        resolved = LoraResolution._find_collection_file(repo_id, filename, local_files_only=False)
        if resolved is not None:
            return resolved

        raise FileNotFoundError(f"Could not find LoRA file '{filename}' in HuggingFace repo: {repo_id}")

    @staticmethod
    def _find_and_link_file(download_path: Path, filename: str, cache_path: Path) -> str:
        found_files = list(download_path.glob(f"**/*{filename}*"))
        for file in found_files:
            if file.is_file() and file.suffix in [".safetensors", ".bin"]:
                if not filename.endswith(file.suffix):
                    target_name = f"{filename}{file.suffix}"
                else:
                    target_name = filename

                target_path = cache_path / target_name
                if not target_path.exists():
                    try:
                        target_path.symlink_to(file)
                    except (OSError, AttributeError):
                        shutil.copy2(file, target_path)

                return str(target_path)

        raise FileNotFoundError(f"Could not find LoRA file '{filename}' in downloaded files")

    @staticmethod
    def get_registry() -> dict[str, Path]:
        return LoraResolution._registry.copy()

    @staticmethod
    def discover_files(library_paths: list[Path]) -> dict[str, Path]:
        lora_files = {}
        for library_path in reversed(library_paths):
            if not library_path.exists() or not library_path.is_dir():
                continue
            for safetensor_path in library_path.rglob("*.safetensors"):
                basename = safetensor_path.stem
                if basename.isdigit() and safetensor_path.parent.name == "transformer":
                    continue
                lora_files[basename] = safetensor_path.resolve()
        return lora_files

    @staticmethod
    def _initialize_registry() -> None:
        library_path_env = os.environ.get("LORA_LIBRARY_PATH")
        if library_path_env:
            library_paths = [Path(p.strip()) for p in library_path_env.split(":") if p.strip()]
            LoraResolution._registry = LoraResolution.discover_files(library_paths)


LoraResolution._initialize_registry()
