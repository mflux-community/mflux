import json
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
import torch
from mlx.utils import tree_unflatten
from safetensors.torch import load_file as torch_load_file

from mflux.cli.defaults.defaults import MFLUX_CACHE_DIR
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
from mflux.models.common.weights.loading.safetensors_reader import SafetensorsReader
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
from mflux.models.common.weights.mapping.weight_mapper import WeightMapper

if TYPE_CHECKING:
    from mflux.models.common.weights.loading.weight_definition import WeightDefinitionType

logger = logging.getLogger(__name__)


class WeightLoader:
    @staticmethod
    def load_single(
        component: ComponentDefinition,
        repo_id: str,
        file_pattern: str = "*.safetensors",
    ) -> LoadedWeights:
        # file_pattern is both the download filter and the test PathResolution applies to a
        # cached snapshot, so it must name safetensors and nothing else. Adding "config.json"
        # would mark every cached copy of a repo that keeps none at its root incomplete
        # (FLUX.2-klein-4B, the Lens VAE, keeps its under vae/), and a pattern loose enough to
        # match a config file would pass a half-downloaded snapshot as complete.
        root_path = PathResolution.resolve(path=repo_id, patterns=[file_pattern])
        if root_path is None:
            raise ValueError(f"No weights location for component '{component.name}': resolved nothing from {repo_id!r}.")  # fmt: off
        return WeightLoader.load_single_local(component=component, root_path=root_path)

    @staticmethod
    def load_single_local(
        component: ComponentDefinition,
        root_path: Path,
    ) -> LoadedWeights:
        weights, q_level, version = WeightLoader._load_component(root_path, component)
        return LoadedWeights(
            components={component.name: weights},
            meta_data=MetaData(quantization_level=q_level, mflux_version=version),
        )

    @staticmethod
    def load(
        weight_definition: "WeightDefinitionType",
        model_path: str | None = None,
        download_patterns: list[str] | None = None,
    ) -> LoadedWeights:
        # download_patterns lets a caller supply variant-aware HF allow_patterns (e.g. Krea 2
        # Turbo vs Raw need different transformer layouts); default to the definition's list.
        root_path = PathResolution.resolve(
            path=model_path,
            patterns=download_patterns if download_patterns is not None else weight_definition.get_download_patterns(),
        )

        # 2. Load each component (with caching for shared sources)
        # A missing root path is a whole-model condition: without it, the first component
        # that lacks a direct download URL used to raise a per-component error ("no
        # download_url for component: vae") that sends the reader auditing their cache and
        # that component, when nothing was ever resolved to load from. Say so once, up
        # front. Definitions whose every component ships a URL (DepthPro) still work
        # without a root path.
        if root_path is None and any(c.download_url is None for c in weight_definition.get_components()):
            raise ValueError(
                f"No weights location for {getattr(weight_definition, '__name__', weight_definition)}: "
                f"model_path was not given and the model config resolved no model_name, so there is "
                f"no directory or repository to load from."
            )

        components = {}
        quantization_level = None
        mflux_version = None
        raw_weights_cache: dict[tuple, dict] = {}  # Cache by (path, loading_mode, weight_files)

        for component in weight_definition.get_components():
            weights, q_level, version = WeightLoader._load_component(root_path, component, raw_weights_cache)
            components[component.name] = weights

            # Track metadata from first component that has it
            if quantization_level is None and q_level is not None:
                quantization_level = q_level
                mflux_version = version

        return LoadedWeights(
            components=components,
            meta_data=MetaData(
                quantization_level=quantization_level,
                mflux_version=mflux_version,
            ),
        )

    @staticmethod
    def _load_component(
        root_path: Path | None,
        component: ComponentDefinition,
        raw_weights_cache: dict[tuple, dict] | None = None,
    ) -> tuple[dict, int | None, str | None]:
        # Some components are distributed in more than one on-disk layout (e.g. a native
        # single-file checkpoint vs a diffusers sharded directory with different keys).
        # Let the component pick the concrete definition based on what is present on disk.
        if component.variant_selector is not None and root_path is not None:
            component = component.variant_selector(root_path)

        # Handle direct URL downloads (e.g., Apple CDN for DepthPro)
        if component.download_url is not None:
            file_path = WeightLoader._download_from_url(component.download_url, component.name)
            raw_weights = WeightLoader._load_weights_file(file_path, component.loading_mode)
        else:
            if root_path is None:
                raise ValueError(f"No root_path and no download_url for component: {component.name}")
            component_path = root_path / component.hf_subdir

            # Try mflux saved format first (including FP8 components reloaded after mflux-save).
            weights, q_level, version = WeightLoader._try_load_mflux_format(component_path)
            if weights is not None:
                return weights, q_level, version

            # Check cache for shared loading (e.g., FIBO VLM decoder + visual from same source)
            cache_key = (str(component_path), component.loading_mode, tuple(component.weight_files or []))
            if raw_weights_cache is not None and cache_key in raw_weights_cache:
                raw_weights = raw_weights_cache[cache_key]
            else:
                # Fall back to HuggingFace format with mapping
                raw_weights = WeightLoader._load_safetensors(
                    component_path, component.loading_mode, component.weight_files
                )
                # Cache for potential reuse by other components
                if raw_weights_cache is not None:
                    raw_weights_cache[cache_key] = raw_weights

        # Apply prefix filtering if specified (e.g., filter "model.language_model" vs "model.visual")
        if component.weight_prefix_filters is not None:
            raw_weights = {
                k: v
                for k, v in raw_weights.items()
                if any(k.startswith(prefix) for prefix in component.weight_prefix_filters)
            }

        if component.key_transform is not None:
            transformed_weights = {}
            for key, value in raw_weights.items():
                transformed_key = component.key_transform(key)
                if transformed_key is not None:
                    transformed_weights[transformed_key] = value
            raw_weights = transformed_weights

        if component.weight_transform is not None:
            raw_weights = {k: component.weight_transform(k, v) for k, v in raw_weights.items()}

        # Apply precision conversion if specified
        if component.precision is not None:
            raw_weights = WeightLoader._convert_precision(raw_weights, component.precision)

        # Passthrough mode: apply bulk transform and unflatten (no key mapping)
        if component.mapping_getter is None:
            if component.bulk_transform is not None:
                raw_weights = {k: component.bulk_transform(v) for k, v in raw_weights.items()}
            return tree_unflatten(list(raw_weights.items())), None, None

        # Standard mode: apply declarative weight mapping
        mapped_weights = WeightMapper.apply_mapping(
            hf_weights=raw_weights,
            mapping=component.mapping_getter(),
            num_blocks=component.num_blocks,
            num_layers=component.num_layers,
        )
        return mapped_weights, None, None

    @staticmethod
    def _indexed_shard_names(path: Path) -> list[str] | None:
        # Returns None when there is no usable index, and the caller then falls back to
        # reading the directory. A damaged index is no evidence that the weights are gone:
        # ModelSaver writes every shard and the index last, so an interrupted save can
        # leave correct weights beside a truncated json, and those have to keep loading.
        index_path = path / "model.safetensors.index.json"
        if not index_path.is_file():
            return None
        try:
            with index_path.open(encoding="utf-8") as index_file:
                index = json.load(index_file)
        except (OSError, ValueError) as exc:
            logger.warning(f"Ignoring unreadable weight index at {index_path}: {exc}")
            return None

        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            logger.warning(f"Ignoring weight index at {index_path}: it names no shards")
            return None

        # Validate before the set: a weight_map value can be any json type, and an
        # unhashable one would raise out of here instead of falling back to the directory.
        if not all(isinstance(name, str) and name and Path(name).name == name for name in weight_map.values()):
            logger.warning(f"Ignoring weight index at {index_path}: it names something other than a local file")
            return None
        return sorted(set(weight_map.values()))

    @staticmethod
    def _try_load_mflux_format(path: Path) -> tuple[dict | None, int | None, str | None]:
        if not path.exists():
            return None, None, None

        shard_files = sorted(f for f in path.glob("*.safetensors") if not f.name.startswith("._"))
        if not shard_files:
            return None, None, None

        # ModelSaver writes model.safetensors.index.json naming every shard, so when it is
        # there it decides what this checkpoint is made of. Reading the directory instead
        # gets both failures: a half-copied checkpoint loads whatever survived and comes up
        # missing weights, and a q4 save over an older q8 one in the same directory picks up
        # the q8 tail that mkdir(exist_ok=True) never cleared, whose tensors then overwrite
        # the ones that were just written. The metadata below is read off an indexed shard
        # for the same reason: whatever sorts first in the directory need not be ours.
        indexed = WeightLoader._indexed_shard_names(path)
        present = [path / name for name in indexed if (path / name).is_file()] if indexed else []

        # Check metadata on first file
        data = mx.load(str(present[0] if present else shard_files[0]), return_metadata=True)
        if len(data) <= 1:
            return None, None, None

        quantization_level_str = data[1].get("quantization_level")
        mflux_version = data[1].get("mflux_version")

        # If no mflux metadata, this isn't our format
        if quantization_level_str is None and mflux_version is None:
            return None, None, None

        # Convert quantization level from string to int
        if quantization_level_str in (None, "None", "null", ""):
            quantization_level = None
        else:
            quantization_level = int(quantization_level_str)

        if indexed is not None:
            missing = [name for name in indexed if not (path / name).is_file()]
            if missing:
                raise FileNotFoundError(
                    f"Incomplete saved model at {path}: the weight index names {len(indexed)} shard(s), "
                    f"and {missing} are not on disk. Copy or save the model again."
                )
            shard_files = present

        # Load all shards
        all_weights: dict[str, mx.array] = {}
        for shard in shard_files:
            shard_data = mx.load(str(shard), return_metadata=True)
            all_weights.update(dict(shard_data[0].items()))

        unflattened = tree_unflatten(list(all_weights.items()))
        return unflattened, quantization_level, mflux_version

    @staticmethod
    def _download_from_url(url: str, component_name: str) -> Path:
        cache_dir = MFLUX_CACHE_DIR / component_name
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Extract filename from URL
        filename = url.split("/")[-1]
        file_path = cache_dir / filename

        if not file_path.exists():
            logger.info(f"Downloading {component_name} weights from {url}...")
            try:
                urllib.request.urlretrieve(url, file_path)
                logger.info(f"Downloaded to {file_path}")
            except (urllib.error.URLError, urllib.error.HTTPError) as e:
                logger.error(f"Failed to download: {e}")
                logger.info(f"Please manually download from: {url}")
                raise FileNotFoundError(f"Model file not found at {file_path}") from e

        return file_path

    @staticmethod
    def _load_weights_file(file_path: Path, loading_mode: str) -> dict[str, mx.array]:
        if loading_mode == "torch_checkpoint":
            return WeightLoader._load_torch_checkpoint(file_path)
        elif loading_mode in ("mlx_native", "single"):
            data = mx.load(str(file_path), return_metadata=True)
            return dict(data[0].items())
        else:
            raise ValueError(f"Unsupported loading mode for single file: {loading_mode}")

    @staticmethod
    def _load_torch_checkpoint(file_path: Path) -> dict[str, mx.array]:
        pt_weights = torch.load(file_path, map_location="cpu", weights_only=False)
        return {k: mx.array(v.numpy()) for k, v in pt_weights.items() if isinstance(v, torch.Tensor)}

    @staticmethod
    def _load_safetensors(path: Path, loading_mode: str, weight_files: list[str] | None = None) -> dict[str, mx.array]:
        if loading_mode == "mlx_native":
            return WeightLoader._load_mlx_native(path, weight_files)
        elif loading_mode == "torch_convert":
            return WeightLoader._load_torch_convert(path, weight_files)
        elif loading_mode == "multi_json":
            return WeightLoader._load_multi_json(path)
        elif loading_mode == "torch_bfloat16":
            return WeightLoader._load_torch_bfloat16(path)
        elif loading_mode == "single":
            return WeightLoader._load_single(path)
        elif loading_mode == "multi_glob":
            return WeightLoader._load_multi_glob(path)
        elif loading_mode == "fp8_safetensors":
            return WeightLoader._load_fp8_safetensors(path)
        else:
            raise ValueError(f"Unknown loading mode: {loading_mode}")

    @staticmethod
    def _load_mlx_native(path: Path, weight_files: list[str] | None = None) -> dict[str, mx.array]:
        if weight_files:
            # Load only specified files
            missing = [f for f in weight_files if not (path / f).exists()]
            if missing:
                raise FileNotFoundError(f"Missing specified weight files in {path}: {missing}")
            shard_files = [path / f for f in weight_files]
        else:
            # Fall back to loading all safetensors files
            shard_files = sorted(f for f in path.glob("*.safetensors") if not f.name.startswith("._"))
            if not shard_files:
                raise FileNotFoundError(f"No safetensors files found in {path}")

        all_weights: dict[str, mx.array] = {}
        for shard in shard_files:
            weights = mx.load(str(shard))
            all_weights.update(weights)

        return all_weights

    @staticmethod
    def _load_torch_convert(path: Path, weight_files: list[str] | None = None) -> dict[str, mx.array]:
        if weight_files:
            # Load only specified files
            missing = [f for f in weight_files if not (path / f).exists()]
            if missing:
                raise FileNotFoundError(f"Missing specified weight files in {path}: {missing}")
            shard_files = [path / f for f in weight_files]
        else:
            # Fall back to loading all safetensors files
            shard_files = sorted(f for f in path.glob("*.safetensors") if not f.name.startswith("._"))
            if not shard_files:
                raise FileNotFoundError(f"No safetensors files found in {path}")

        all_weights: dict[str, mx.array] = {}
        for shard in shard_files:
            torch_weights = torch_load_file(str(shard))
            for key, tensor in torch_weights.items():
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float16)
                all_weights[key] = mx.array(tensor.numpy())

        return all_weights

    @staticmethod
    def _load_multi_json(path: Path) -> dict[str, mx.array]:
        index_path = path / "model.safetensors.index.json"
        with open(index_path) as f:
            index = json.load(f)

        # Group weights by file
        files_to_load: dict[str, list[str]] = {}
        for param_name, file_name in index["weight_map"].items():
            if file_name not in files_to_load:
                files_to_load[file_name] = []
            files_to_load[file_name].append(param_name)

        all_weights: dict[str, mx.array] = {}
        for file_name, param_names in files_to_load.items():
            file_path = path / file_name

            # Use mx.load which handles bfloat16 natively
            file_weights = mx.load(str(file_path))

            for param_name in param_names:
                if param_name in file_weights:
                    all_weights[param_name] = file_weights[param_name]

        return all_weights

    @staticmethod
    def _load_torch_bfloat16(path: Path) -> dict[str, mx.array]:
        index_path = path / "model.safetensors.index.json"
        with open(index_path) as f:
            index = json.load(f)

        weight_files = sorted(set(index["weight_map"].values()))

        all_weights: dict[str, mx.array] = {}
        for wf in weight_files:
            file_path = path / wf
            data = torch_load_file(str(file_path))
            for k, v in data.items():
                if v.dtype == torch.bfloat16:
                    v = v.to(torch.float16)
                np_arr = v.detach().cpu().numpy()
                all_weights[k] = mx.array(np_arr)

        return all_weights

    @staticmethod
    def _load_single(path: Path) -> dict[str, mx.array]:
        safetensors_files = [f for f in path.glob("*.safetensors") if not f.name.startswith("._")]
        if not safetensors_files:
            raise FileNotFoundError(f"No safetensors files found in {path}")

        weights_file = safetensors_files[0]
        data = mx.load(str(weights_file), return_metadata=True)
        return dict(data[0].items())

    @staticmethod
    def _load_multi_glob(path: Path) -> dict[str, mx.array]:
        shard_files = sorted(f for f in path.glob("*.safetensors") if not f.name.startswith("._"))
        if not shard_files:
            raise FileNotFoundError(f"No safetensors files found in {path}")

        all_weights: dict[str, mx.array] = {}
        for shard in shard_files:
            data, _ = mx.load(str(shard), return_metadata=True)
            all_weights.update(dict(data.items()))

        return all_weights

    @staticmethod
    def _load_fp8_safetensors(path: Path) -> dict[str, mx.array]:
        return SafetensorsReader.read_directory(path)

    @staticmethod
    def _convert_precision(weights: dict[str, mx.array], precision: mx.Dtype) -> dict[str, mx.array]:
        return {k: v if v.dtype == precision else v.astype(precision) for k, v in weights.items()}
