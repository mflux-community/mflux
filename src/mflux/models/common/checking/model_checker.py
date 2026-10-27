import hashlib
import json
import math
import os
import re
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub.constants import HF_HUB_CACHE

from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.common.resolution.model_revisions import ModelRevisions
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
from mflux.models.common.weights.mapping.weight_mapper import WeightMapper
from mflux.utils.exceptions import ModelConfigError


@dataclass(frozen=True)
class ModelCheckResult:
    check: str
    format: str | None
    size_bytes: int
    model: str | None
    problems: tuple[str, ...]


class ModelChecker:
    # Bits per element for each safetensors dtype.
    _DTYPE_BITS = {
        "BOOL": 8,
        "U8": 8,
        "I8": 8,
        "F8_E5M2": 8,
        "F8_E4M3": 8,
        "F8_E8M0": 8,
        "I16": 16,
        "U16": 16,
        "F16": 16,
        "BF16": 16,
        "I32": 32,
        "U32": 32,
        "F32": 32,
        "I64": 64,
        "U64": 64,
        "F64": 64,
        "F4": 4,
        "F6_E2M3": 6,
        "F6_E3M2": 6,
    }

    @staticmethod
    def check(reference: str, base_model: str | None = None) -> ModelCheckResult:
        root, resolution_problem = ModelChecker._resolve(reference)
        if root is None:
            return ModelCheckResult("fail", None, 0, None, (resolution_problem,))

        files = sorted(path for path in root.rglob("*.safetensors") if path.is_file())
        if not files:
            problems: list[str] = []
            if ModelChecker._validate_indexes(root, problems):
                return ModelCheckResult(
                    "warn", "mflux", 0, ModelChecker._resolve_model_key(reference, base_model), tuple(problems)
                )
            return ModelCheckResult("fail", None, 0, None, (f"No safetensors files exist under {root}.",))

        headers: dict[Path, dict[str, Any]] = {}
        problems: list[str] = []
        for path in files:
            header, problem = ModelChecker._read_header_or_problem(path, root)
            if header is not None:
                headers[path] = header
            if problem is not None:
                problems.append(problem)

        index_metadata = ModelChecker._index_metadata(root)
        for label, metadata in [
            *((str(path.relative_to(root)), header.get("__metadata__")) for path, header in headers.items()),
            *index_metadata,
        ]:
            problem = ModelChecker._metadata_problem(metadata, label)
            if problem is not None:
                problems.append(problem)

        is_mflux = any(ModelChecker._is_mflux_header(header) for header in headers.values()) or any(
            ModelChecker._is_mflux_metadata(metadata) for _, metadata in index_metadata
        )
        format_name = "mflux" if is_mflux else "safetensors"
        key = ModelChecker._resolve_model_key(reference, base_model)
        if key is None:
            if format_name == "safetensors":
                ModelChecker._validate_indexes(root, problems)
                return ModelCheckResult(
                    "fail" if problems else "pass", format_name, ModelChecker._size(files), None, tuple(problems)
                )
            problems.append("Cannot identify a model that an MFlux CLI supports.")
            return ModelCheckResult("fail", format_name, ModelChecker._size(files), None, tuple(problems))

        definition = ModelChecker._definition_for(key)
        if definition is None:
            problems.append(f"MFlux has no checker definition for {key}.")
            return ModelCheckResult("fail", format_name, ModelChecker._size(files), key, tuple(problems))

        missing_parts = ModelChecker._validate_components(root, definition, format_name, headers, problems)
        missing_parts = ModelChecker._validate_indexes(root, problems) or missing_parts

        if format_name == "mflux" and missing_parts:
            return ModelCheckResult("warn", format_name, ModelChecker._size(files), key, tuple(problems))
        return ModelCheckResult(
            "fail" if problems else "pass", format_name, ModelChecker._size(files), key, tuple(problems)
        )

    @staticmethod
    def format_result(result: ModelCheckResult, reference: str, fix: bool = False, repair: str | None = None) -> str:
        lines = ["#################", "MFlux Check Model:", f" check: {result.check}"]
        if result.format is not None:
            lines.append(f" format: {result.format}")
        lines.append(f" size: {result.size_bytes / 1_000_000_000:.2f} GB")
        if result.model is not None and result.check != "pass":
            lines.append(f" model: {result.model}")
        lines.extend(f" problem: {problem}" for problem in result.problems)
        if result.check == "warn":
            lines.append(f" fix: mflux-check model --fix {reference}")
            if fix:
                lines.append(
                    f" repair: {repair or 'No local replacement is available. The check did not download files.'}"
                )
        return "\n".join(lines)

    @staticmethod
    def repair(reference: str) -> str:
        # A Hugging Face snapshot links each file to blobs/<sha256>. A dangling link
        # gives the exact content hash of the missing shard, so repair accepts a
        # replacement only when its SHA-256 matches. Without that hash, matching tensor
        # names cannot prove that two revisions hold the same weights.
        root, problem = ModelChecker._resolve(reference)
        if root is None:
            return problem
        snapshots = root.parent
        if snapshots.name != "snapshots":
            return "No local replacement is available. The check did not download files."
        blobs = (snapshots.parent / "blobs").resolve()

        repaired = 0
        for index_path in root.rglob("*.safetensors.index.json"):
            try:
                weight_map = ModelChecker._read_weight_map(index_path)
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
            relative_dir = index_path.parent.relative_to(root)
            for name in set(weight_map.values()):
                target = index_path.parent / name
                if target.is_file():
                    continue
                blob = ModelChecker._missing_blob(target, blobs)
                if blob is None:
                    continue
                expected_tensors = {tensor for tensor, shard_name in weight_map.items() if shard_name == name}
                source = next(
                    (
                        candidate
                        for snapshot in snapshots.iterdir()
                        if snapshot != root
                        and (candidate := snapshot / relative_dir / name).is_file()
                        and ModelChecker._shard_contains_tensors(candidate, expected_tensors)
                        and ModelChecker._sha256(candidate) == blob.name
                    ),
                    None,
                )
                if source is not None:
                    partial = blob.with_name(f"{blob.name}.mflux-repair")
                    shutil.copyfile(source, partial)
                    os.replace(partial, blob)
                    repaired += 1
        if repaired:
            return f"Restored {repaired} shard(s) from another local cache snapshot."
        return "No local replacement is available. The check did not download files."

    @staticmethod
    def _missing_blob(target: Path, blobs: Path) -> Path | None:
        # Accept only a dangling link to blobs/<sha256> in the same cache repository.
        # Repair never writes through a link to any other location.
        if not target.is_symlink():
            return None
        blob = (target.parent / os.readlink(target)).resolve()
        if blob.parent != blobs or not re.fullmatch(r"[0-9a-f]{64}", blob.name) or blob.exists():
            return None
        return blob

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1 << 24), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _resolve(reference: str) -> tuple[Path | None, str]:
        local = Path(reference).expanduser()
        if local.exists():
            return local, ""
        if not PathResolution._is_hf_format(reference):
            return None, f"Model path does not exist: {reference}."

        snapshots = Path(HF_HUB_CACHE) / f"models--{reference.replace('/', '--')}" / "snapshots"
        if not snapshots.is_dir():
            return None, f"Model is not in the Hugging Face cache: {reference}."
        pinned = ModelRevisions.get(reference)
        if pinned is not None:
            # MFlux loads only the pinned commit of this repo, so check that snapshot.
            if not (snapshots / pinned).is_dir():
                return None, f"Model is not in the Hugging Face cache at the pinned revision {pinned}: {reference}."
            return snapshots / pinned, ""
        candidates = sorted(
            (path for path in snapshots.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime
        )
        if not candidates:
            return None, f"Model has no cached snapshots: {reference}."
        return candidates[-1], ""

    @staticmethod
    def _resolve_model_key(reference: str, base_model: str | None = None) -> str | None:
        try:
            return ConfigResolution.resolve_key(reference, base_model=base_model)
        except ModelConfigError:
            return None

    @staticmethod
    def _definition_for(key: str) -> type | None:
        if key in {"dev-redux"}:
            from mflux.models.flux.weights.flux_weight_definition import FluxReduxSaveDefinition

            return FluxReduxSaveDefinition
        if key in {"dev-controlnet-canny", "dev-controlnet-upscaler", "schnell-controlnet-canny"}:
            from mflux.models.flux.weights.flux_weight_definition import FluxControlnetWeightDefinition

            return FluxControlnetWeightDefinition
        if key in {"dev", "schnell", "dev-depth", "dev-fill", "dev-fill-catvton", "dev-kontext", "krea-dev"}:
            from mflux.models.flux.weights.flux_weight_definition import FluxWeightDefinition

            return FluxWeightDefinition
        if key.startswith("flux2-"):
            from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition

            return Flux2KleinWeightDefinition
        if key in {"qwen-image", "qwen-image-edit", "qwen-image-edit-2511"}:
            from mflux.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition

            return QwenWeightDefinition
        if key in {"qwen-image-2.1-controlnet-union", "qwen-image-2.1-turbo-controlnet-union"}:
            from mflux.models.qwen21.weights.qwen_image21_controlnet_weight_definition import (
                QwenImage21ControlnetWeightDefinition,
            )

            return QwenImage21ControlnetWeightDefinition
        if key == "qwen-image-2.1":
            from mflux.models.qwen21.weights.qwen21_weight_definition import Qwen21WeightDefinition

            return Qwen21WeightDefinition
        if key in {"z-image", "z-image-turbo"}:
            from mflux.models.z_image.weights.z_image_weight_definition import ZImageWeightDefinition

            return ZImageWeightDefinition
        if key == "z-image-turbo-controlnet-union-2.1":
            from mflux.models.z_image.weights.z_image_controlnet_weight_definition import (
                ZImageControlnetWeightDefinition,
            )

            return ZImageControlnetWeightDefinition
        if key in {"seedvr2-3b", "seedvr2-7b"}:
            from mflux.models.seedvr2.weights.seedvr2_weight_definition import (
                SeedVR2WeightDefinition3B,
                SeedVR2WeightDefinition7B,
            )

            return SeedVR2WeightDefinition3B if key == "seedvr2-3b" else SeedVR2WeightDefinition7B
        if key in {"fibo", "fibo-lite", "fibo-edit", "fibo-edit-rmbg"}:
            from mflux.models.fibo.weights.fibo_weight_definition import FIBOWeightDefinition

            return FIBOWeightDefinition
        if key in {"fibo2", "fibo2-edit-turbo"}:
            from mflux.models.fibo2.weights.fibo2_weight_definition import Fibo2WeightDefinition

            return Fibo2WeightDefinition
        if key in {"ernie-image", "ernie-image-turbo"}:
            from mflux.models.ernie_image.weights.ernie_weight_definition import ErnieWeightDefinition

            return ErnieWeightDefinition
        if key == "boogu-image-turbo":
            from mflux.models.boogu.weights.boogu_weight_definition import BooguWeightDefinition

            return BooguWeightDefinition
        if key == "ideogram-4-fp8":
            from mflux.models.ideogram4.weights.ideogram4_weight_definition import Ideogram4WeightDefinition

            return Ideogram4WeightDefinition
        if key in {"flux3-image-klein", "flux3-image-klein-base"}:
            from mflux.models.flux3.weights.flux3_weight_definition import (
                Flux3KleinBaseWeightDefinition,
                Flux3KleinWeightDefinition,
            )

            return Flux3KleinWeightDefinition if key == "flux3-image-klein" else Flux3KleinBaseWeightDefinition
        if key == "lens-turbo":
            # The DiT repo only. The VAE and the text encoder load from other repos.
            from mflux.models.lens.weights.lens_weight_definition import LensWeightDefinition

            return LensWeightDefinition
        if key == "qwen-image-2.1-turbo":
            from mflux.models.qwen21.weights.qwen_image21_weight_definition import QwenImage21WeightDefinition

            return QwenImage21WeightDefinition
        if key in {"krea-2", "krea-2-raw", "krea-2-controlnet-depth"}:
            from mflux.models.krea2.weights.krea2_weight_definition import Krea2WeightDefinition

            return Krea2WeightDefinition
        if key == "ming-image-design":
            from mflux.models.ming_image.weights.ming_image_weight_definition import MingImageWeightDefinition

            return MingImageWeightDefinition
        if key in {"moss-image-sft", "moss-image-base"}:
            from mflux.models.moss_image.weights.moss_image_weight_definition import MOSSImageWeightDefinition

            return MOSSImageWeightDefinition
        if key in {"anima-turbo", "anima-aesthetic", "anima-base"}:
            from mflux.models.anima.weights.anima_weight_definition import AnimaWeightDefinition

            return AnimaWeightDefinition
        if key in {"iris-3b", "iris-3b-base"}:
            from mflux.models.iris.weights.iris_weight_definition import (
                IrisTurboWeightDefinition,
                IrisWeightDefinition,
            )

            return IrisTurboWeightDefinition if key == "iris-3b" else IrisWeightDefinition
        return None

    @staticmethod
    def _validate_components(
        root: Path,
        definition: type,
        format_name: str,
        headers: dict[Path, dict[str, Any]],
        problems: list[str],
    ) -> bool:
        components = definition.get_components()
        save_subdirs = ComponentDefinition.save_subdirs(components)
        missing_parts = False
        for component in components:
            save_subdir = save_subdirs[component.name]
            # Use the on-disk layout that the loader selects. For example, Krea 2 Raw keeps
            # its transformer under transformer/, not at the repository root.
            if component.variant_selector is not None:
                # A selector can raise when no known layout is on disk (Krea 2 without a
                # transformer). Report it as a missing part, not as a crash.
                try:
                    component = component.variant_selector(root)
                except FileNotFoundError as exc:
                    problems.append(str(exc))
                    missing_parts = True
                    continue
            if format_name == "mflux":
                # The loader reads the save subdir first, then the hf_subdir of the variant.
                candidates = dict.fromkeys([save_subdir, component.hf_subdir])
            else:
                candidates = {component.hf_subdir: None}
            relative = next(iter(candidates))
            component_files: list[Path] = []
            for candidate in candidates:
                directory = root / candidate
                component_files = [path for path in headers if path.parent == directory]
                if component_files:
                    relative = candidate
                    break
            if not component_files:
                problems.append(f"Missing component: {relative or component.name}.")
                missing_parts = True
                continue
            if format_name == "safetensors" and component.mapping_getter is not None:
                keys = ModelChecker._component_keys(component, component_files, headers)
                missing = WeightMapper.missing_required_names(
                    keys, component.mapping_getter(), component.num_blocks, component.num_layers
                )
                if missing:
                    problems.append(f"Component {component.name} misses required weights: {', '.join(missing[:3])}.")
        return missing_parts

    @staticmethod
    def _component_keys(
        component: ComponentDefinition,
        files: list[Path],
        headers: dict[Path, dict[str, Any]],
    ) -> dict[str, None]:
        names = {}
        for path in files:
            for name in headers[path]:
                if name == "__metadata__":
                    continue
                if component.weight_prefix_filters is not None and not any(
                    name.startswith(prefix) for prefix in component.weight_prefix_filters
                ):
                    continue
                if component.key_transform is not None:
                    name = component.key_transform(name)
                    if name is None:
                        continue
                names[name] = None
        return names

    @staticmethod
    def _validate_indexes(root: Path, problems: list[str]) -> bool:
        missing_parts = False
        for index_path in root.rglob("*.safetensors.index.json"):
            missing, problem = ModelChecker._index_missing_parts(index_path, root)
            if problem is not None:
                problems.append(problem)
            if missing:
                missing_parts = True
        return missing_parts

    @staticmethod
    def _read_header(path: Path) -> dict[str, Any]:
        with path.open("rb") as file:
            raw_length = file.read(8)
            if len(raw_length) != 8:
                raise ValueError("file has no safetensors header")
            length = struct.unpack("<Q", raw_length)[0]
            file_size = path.stat().st_size
            if length > file_size - 8:
                raise ValueError("header length exceeds file size")
            header = json.loads(file.read(length))
        if not isinstance(header, dict):
            raise ValueError("header is not an object")
        payload_size = file_size - 8 - length
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            offsets = tensor.get("data_offsets") if isinstance(tensor, dict) else None
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(not isinstance(offset, int) for offset in offsets)
                or offsets[0] < 0
                or offsets[0] > offsets[1]
                or offsets[1] > payload_size
            ):
                raise ValueError(f"tensor {name!r} has invalid data_offsets")
            bits = ModelChecker._DTYPE_BITS.get(tensor.get("dtype"))
            if bits is None:
                raise ValueError(f"tensor {name!r} has an unknown dtype")
            shape = tensor.get("shape")
            if not isinstance(shape, list) or any(
                not isinstance(size, int) or isinstance(size, bool) or size < 0 for size in shape
            ):
                raise ValueError(f"tensor {name!r} has an invalid shape")
            if math.prod(shape) * bits != (offsets[1] - offsets[0]) * 8:
                raise ValueError(f"tensor {name!r} has a shape that does not match its data_offsets")
        return header

    @staticmethod
    def _read_header_or_problem(path: Path, root: Path) -> tuple[dict[str, Any] | None, str | None]:
        try:
            return ModelChecker._read_header(path), None
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return None, f"Cannot read safetensors header: {path.relative_to(root)} ({exc})."

    @staticmethod
    def _index_missing_parts(index_path: Path, root: Path) -> tuple[bool, str | None]:
        try:
            weight_map = ModelChecker._read_weight_map(index_path)
            names = set(weight_map.values())
            missing = [name for name in sorted(names) if not (index_path.parent / name).is_file()]
            if missing:
                return True, f"Index {index_path.relative_to(root)} misses shards: {', '.join(missing)}."
            invalid = [
                f"{tensor} in {name}"
                for tensor, name in weight_map.items()
                if not ModelChecker._shard_contains_tensors(index_path.parent / name, {tensor})
            ]
            if invalid:
                return (
                    False,
                    f"Index {index_path.relative_to(root)} assigns tensors absent from shards: {', '.join(invalid)}.",
                )
            return False, None
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return False, f"Cannot read index {index_path.relative_to(root)} ({exc})."

    @staticmethod
    def _read_weight_map(index_path: Path) -> dict[str, str]:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("weight_map is missing")
        if not all(isinstance(tensor, str) and tensor for tensor in weight_map):
            raise ValueError("weight_map has an invalid tensor name")
        if not all(ModelChecker._is_local_shard_name(name) for name in weight_map.values()):
            raise ValueError("weight_map names a non-local shard")
        return weight_map

    @staticmethod
    def _is_local_shard_name(name: object) -> bool:
        return isinstance(name, str) and bool(name) and "/" not in name and "\\" not in name and Path(name).name == name

    @staticmethod
    def _shard_contains_tensors(path: Path, tensors: set[str]) -> bool:
        try:
            return tensors.issubset(ModelChecker._read_header(path))
        except (OSError, ValueError, json.JSONDecodeError):
            return False

    @staticmethod
    def _is_mflux_header(header: dict[str, Any]) -> bool:
        return ModelChecker._is_mflux_metadata(header.get("__metadata__"))

    @staticmethod
    def _is_mflux_metadata(metadata: object) -> bool:
        return isinstance(metadata, dict) and ("mflux_version" in metadata or "quantization_level" in metadata)

    @staticmethod
    def _index_metadata(root: Path) -> list[tuple[str, object]]:
        entries = []
        for index_path in root.rglob("*.safetensors.index.json"):
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(index, dict):
                entries.append((str(index_path.relative_to(root)), index.get("metadata")))
        return entries

    @staticmethod
    def _metadata_problem(metadata: object, label: str) -> str | None:
        # Accept the same quantization_level values that WeightLoader accepts.
        if not isinstance(metadata, dict) or not ModelChecker._is_mflux_metadata(metadata):
            return None
        level = metadata.get("quantization_level")
        if level in (None, "None", "null", ""):
            return None
        try:
            int(level)
        except (TypeError, ValueError):
            return f"Invalid quantization_level {level!r} in {label}."
        return None

    @staticmethod
    def _size(files: list[Path]) -> int:
        seen = set()
        total = 0
        for path in files:
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if identity not in seen:
                seen.add(identity)
                total += stat.st_size
        return total
