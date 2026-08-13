import datetime
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from mflux.models.common.training.state.zip_util import ZipUtil

TRAINING_FILE_NAME_RUN_MANIFEST = "run.json"


@dataclass
class DataSpec:
    image: Path
    input_image: Path | None
    prompt: str

    @classmethod
    def create(cls, param: Mapping[str, Any], absolute_or_relative_path: str, base_path: Path | None) -> "DataSpec":
        image_path = DataSpec._resolve_relative_to_data_path(
            Path(str(param["image"])),
            absolute_or_relative_path=absolute_or_relative_path,
            base_path=base_path,
        )
        input_image = None
        if param.get("input_image", None) is not None:
            input_image = DataSpec._resolve_relative_to_data_path(
                Path(str(param["input_image"])),
                absolute_or_relative_path=absolute_or_relative_path,
                base_path=base_path,
            )

        has_prompt = "prompt" in param and param["prompt"] is not None
        has_prompt_file = "prompt_file" in param and param["prompt_file"] is not None
        if has_prompt and has_prompt_file:
            raise ValueError("Each data item must provide either 'prompt' or 'prompt_file' (not both).")
        if has_prompt:
            prompt = str(param["prompt"])
        elif has_prompt_file:
            prompt_path = DataSpec._resolve_relative_to_data_path(
                Path(str(param["prompt_file"])),
                absolute_or_relative_path=absolute_or_relative_path,
                base_path=base_path,
            )
            try:
                prompt = prompt_path.read_text(encoding="utf-8", errors="replace").strip()
            except FileNotFoundError as e:
                raise ValueError(f"Prompt file not found: {prompt_path}") from e
        else:
            raise ValueError("Each data item must provide either 'prompt' or 'prompt_file'.")

        return cls(image=image_path, input_image=input_image, prompt=prompt)

    @staticmethod
    def _resolve_relative_to_data_path(path: Path, *, absolute_or_relative_path: str, base_path: Path | None) -> Path:
        if path.is_absolute():
            return path

        data_root = Path(absolute_or_relative_path)
        if data_root.is_absolute():
            return data_root / path

        if base_path is None:
            raise ValueError(
                "Config uses a relative data path, but no config_path is available to resolve it. "
                "Use an absolute data path when resuming from a checkpoint."
            )

        return Path(base_path).parent / data_root / path


@dataclass
class TrainingLoopSpec:
    num_epochs: int
    batch_size: int
    timestep_low: int = 0
    timestep_high: int | None = None
    iterator_state_path: str | None = None
    # Timestep-index sampling distribution: None/"uniform" (flat), "sigmoid" (mid-concentrated,
    # ai-toolkit's default — best for subject identity), "content" (cubic, favors low-noise /
    # fine detail), "style" (favors high-noise / coarse style). Applies to adapters that sample
    # sigma from the index grid (flux/z-image/ernie); Ideogram has its own sample_sigma override.
    timestep_type: str | None = None


@dataclass
class OptimizerSpec:
    name: str
    learning_rate: float
    state_path: str | None = None
    # LR schedule: None = constant; "cosine" = cosine decay (needs lr_total_steps). A linear
    # warmup of lr_warmup_steps is prepended either way (0 = none). Resumes correctly: the
    # schedule is rebuilt from these fields and applied at the optimizer's restored step.
    lr_schedule: str | None = None
    lr_warmup_steps: int = 0
    lr_total_steps: int | None = None
    # Extra keyword arguments forwarded to the MLX optimizer constructor
    # (e.g. weight_decay, betas, eps). Without this, any optimizer hyperparameter
    # other than the learning rate silently falls back to the MLX default.
    optimizer_params: dict = field(default_factory=dict)
    # Global-norm gradient clipping; None = off (preserves prior behavior).
    max_grad_norm: float | None = None
    # Gradient accumulation: average grads over this many micro-batches before one optimizer
    # step, for an effective batch of batch_size * gradient_accumulation_steps without the memory
    # of a bigger batch. 1 = off (steps every micro-batch). Frequencies (save/plot/preview) and
    # the step count stay in micro-batch units.
    gradient_accumulation_steps: int = 1


@dataclass
class CheckpointSpec:
    save_frequency: int
    output_path: str


@dataclass
class StatisticsSpec:
    state_path: str | None = None


@dataclass
class BlockRange:
    start: int | None = None
    end: int | None = None
    indices: list[int] | None = None

    def get_blocks(self) -> list[int]:
        if self.indices is not None:
            return list(self.indices)
        if self.start is not None and self.end is not None:
            return list(range(self.start, self.end))
        raise ValueError("Either 'start' and 'end' or 'indices' must be provided.")


@dataclass
class LoraTargetSpec:
    module_path: str
    rank: int
    blocks: BlockRange | None = None
    # LoRA alpha; effective scale = alpha / rank (published-recipe convention).
    # None keeps the prior behavior (scale = 1.0).
    alpha: float | None = None


@dataclass
class LoraLayersSpec:
    targets: list[LoraTargetSpec]
    state_path: str | None = None


@dataclass
class MonitoringSpec:
    plot_frequency: int
    generate_image_frequency: int
    preview_prompts: list[str]
    preview_prompt_names: list[str]
    preview_width: int = 1024
    preview_height: int = 1024
    preview_images: list[Path] | None = None
    smooth_loss: bool = False
    smooth_loss_window: int = 5

    @staticmethod
    def create(
        param: Mapping[str, Any],
        *,
        preview_prompts: list[str],
        preview_prompt_names: list[str],
    ) -> "MonitoringSpec":
        has_validation_prompt = "validation_prompt" in param and param["validation_prompt"] is not None
        if has_validation_prompt:
            raise ValueError("Monitoring no longer accepts 'validation_prompt'. Use data/preview*.txt.")

        has_validation_prompt_file = "validation_prompt_file" in param and param["validation_prompt_file"] is not None
        if has_validation_prompt_file:
            raise ValueError("Monitoring no longer accepts 'validation_prompt_file'. Use data/preview*.txt.")

        has_validation_image = "validation_image" in param and param["validation_image"] is not None
        if has_validation_image:
            raise ValueError("Monitoring no longer accepts 'validation_image'. Use data/preview.* instead.")

        has_preview_prompt = "preview_prompt" in param and param["preview_prompt"] is not None
        if has_preview_prompt:
            raise ValueError("Monitoring no longer accepts 'preview_prompt'. Use data/preview*.txt.")

        has_preview_prompt_file = "preview_prompt_file" in param and param["preview_prompt_file"] is not None
        if has_preview_prompt_file:
            raise ValueError("Monitoring no longer accepts 'preview_prompt_file'. Use data/preview*.txt.")

        has_preview_image = "preview_image" in param and param["preview_image"] is not None
        if has_preview_image:
            raise ValueError("Monitoring no longer accepts 'preview_image'. Use data/preview.* instead.")

        has_validation_width = "validation_width" in param and param["validation_width"] is not None
        has_validation_height = "validation_height" in param and param["validation_height"] is not None
        if has_validation_width or has_validation_height:
            raise ValueError("Monitoring uses 'preview_width'/'preview_height'.")

        if not preview_prompts:
            raise ValueError("Monitoring requires at least one data/preview*.txt.")

        preview_width = int(param.get("preview_width", 1024))
        preview_height = int(param.get("preview_height", 1024))
        if preview_width <= 0 or preview_height <= 0:
            raise ValueError("Monitoring preview_width/preview_height must be > 0")

        plot_frequency = int(param["plot_frequency"])
        generate_image_frequency = int(param["generate_image_frequency"])
        if plot_frequency <= 0:
            raise ValueError("Monitoring plot_frequency must be > 0")
        if generate_image_frequency <= 0:
            raise ValueError("Monitoring generate_image_frequency must be > 0")

        smooth_loss = bool(param.get("smooth_loss", False))
        smooth_loss_window = int(param.get("smooth_loss_window", 5))
        if smooth_loss and smooth_loss_window < 2:
            raise ValueError("Monitoring smooth_loss_window must be >= 2 when smooth_loss is enabled.")

        return MonitoringSpec(
            plot_frequency=plot_frequency,
            generate_image_frequency=generate_image_frequency,
            preview_prompts=preview_prompts,
            preview_prompt_names=preview_prompt_names,
            preview_width=preview_width,
            preview_height=preview_height,
            preview_images=None,
            smooth_loss=smooth_loss,
            smooth_loss_window=smooth_loss_window,
        )


@dataclass
class TrainingSpec:
    model: str
    model_path: str | None
    seed: int
    steps: int
    guidance: float
    quantize: int | None
    training_loop: TrainingLoopSpec
    optimizer: OptimizerSpec
    checkpoint: CheckpointSpec
    monitoring: MonitoringSpec | None
    lora_layers: LoraLayersSpec
    statistics: StatisticsSpec | None
    data: list[DataSpec]
    is_edit: bool = False
    # Optional cap on the largest side when inferring per-image dimensions.
    max_resolution: int | None = None
    # Absolute path to data root directory (resolved from data path in config/checkpoint).
    data_root: str | None = None
    config_path: str | None = None
    checkpoint_path: str | None = None
    # If true, training will use a disk-backed cache for encoded data to reduce RAM usage.
    low_ram: bool = False
    # Cap (in GB) on MLX's buffer-cache pool (mx.set_cache_limit). The pool otherwise grows
    # to hold the largest transient allocations seen; bounding it returns freed buffers to
    # the OS at ~no speed cost (unlike low_ram's clear-every-step). None = MLX default.
    cache_limit_gb: float | None = None

    @staticmethod
    def resolve(
        config_path: str | None,
        resume_path: str | None,
        *,
        create_output_dir: bool = True,
    ) -> "TrainingSpec":
        if bool(config_path) == bool(resume_path):
            raise ValueError("Provide exactly one of 'config_path' or 'resume_path'.")
        if resume_path:
            return TrainingSpec._from_checkpoint(resume_path, new_folder=False, create_output_dir=create_output_dir)
        return TrainingSpec._from_config(config_path, new_folder=True, create_output_dir=create_output_dir)

    @staticmethod
    def _from_config(path: str, new_folder: bool = True, *, create_output_dir: bool = True) -> "TrainingSpec":
        with open(Path(path), "r") as f:
            data = json.load(f)
        return TrainingSpec.from_conf(data, path, new_folder, create_output_dir=create_output_dir)

    @staticmethod
    def from_conf(
        config: dict, config_path: str | None, new_folder: bool = True, *, create_output_dir: bool = True
    ) -> "TrainingSpec":
        absolute_config_path = None if config_path is None else Path(config_path).absolute()

        steps = int(config["steps"])
        guidance_raw = config.get("guidance", 0.0)
        guidance = 0.0 if guidance_raw is None else float(guidance_raw)
        quantize_raw = config.get("quantize", None)
        quantize = None if quantize_raw is None else int(quantize_raw)

        max_resolution_raw = config.get("max_resolution", None)
        max_resolution = None if max_resolution_raw is None else int(max_resolution_raw)
        if max_resolution is not None and max_resolution <= 0:
            raise ValueError("'max_resolution' must be > 0")

        data_conf = config["data"]
        if isinstance(data_conf, str):
            absolute_or_relative_path = data_conf
        elif isinstance(data_conf, dict):
            if config_path is not None:
                raise ValueError("Config 'data' must be a string path.")
            if "path" not in data_conf:
                raise ValueError("Config 'data' must be a string path.")
            absolute_or_relative_path = str(data_conf["path"])
        else:
            raise ValueError("Config 'data' must be a string path.")

        data_root_dir = TrainingSpec._resolve_data_root_dir(
            data_path=absolute_or_relative_path,
            base_path=absolute_config_path,
        )

        low_ram = bool(config.get("low_ram", False))
        if low_ram and data_root_dir is None:
            raise ValueError("'low_ram' requires a valid data path.")

        cache_limit_raw = config.get("cache_limit_gb", None)
        cache_limit_gb = None if cache_limit_raw is None else float(cache_limit_raw)
        if cache_limit_gb is not None and cache_limit_gb <= 0:
            raise ValueError("'cache_limit_gb' must be > 0")

        monitoring_conf = config.get("monitoring", None)
        preview_prompt_paths: list[Path] = []
        preview_prompts: list[str] = []
        preview_prompt_names: list[str] = []
        preview_images_by_stem: dict[str, Path] = {}
        preview_image_paths: list[Path] = []
        if monitoring_conf is not None:
            preview_prompt_paths = TrainingSpec._find_preview_prompt_files(data_root_dir)
            preview_prompts = TrainingSpec._load_preview_prompts(preview_prompt_paths)
            preview_prompt_names = [path.stem for path in preview_prompt_paths]
            preview_images_by_stem = TrainingSpec._find_preview_images_in_data(data_root_dir)
            preview_image_paths = list(preview_images_by_stem.values())

        # Auto-discovery: all images in data path with matching .txt prompt files
        data = TrainingSpec._discover_data(
            data_path=absolute_or_relative_path,
            base_path=absolute_config_path,
            exclude_paths=None if not preview_image_paths else set(preview_image_paths),
        )
        is_edit = any(item.input_image is not None for item in data)

        training_loop_conf = config["training_loop"].copy()

        timestep_low = int(training_loop_conf.get("timestep_low", 0))
        timestep_high_raw = training_loop_conf.get("timestep_high", None)
        timestep_high = None if timestep_high_raw is None else int(timestep_high_raw)
        resolved_timestep_high = steps if timestep_high is None else timestep_high

        if timestep_low < 0:
            raise ValueError("'timestep_low' must be >= 0")
        if resolved_timestep_high <= 0:
            raise ValueError("'timestep_high' must be > 0")
        if timestep_low >= resolved_timestep_high:
            raise ValueError("'timestep_low' must be < 'timestep_high'")
        if resolved_timestep_high > steps:
            raise ValueError("'timestep_high' must be <= 'steps'")

        training_loop = TrainingLoopSpec(**training_loop_conf)
        optimizer = OptimizerSpec(**config["optimizer"])

        checkpoint_conf = config["checkpoint"]
        save_frequency = int(checkpoint_conf["save_frequency"])
        if save_frequency <= 0:
            raise ValueError("Checkpoint save_frequency must be > 0")
        checkpoint = CheckpointSpec(
            save_frequency=save_frequency,
            output_path=TrainingSpec._resolve_output_path(
                checkpoint_conf["output_path"], new_folder, create_output_dir=create_output_dir
            ),
        )

        # Track whether we're using fallback prompts (for edit training preview image logic)
        using_fallback_prompts = False
        if monitoring_conf is not None and not preview_prompts:
            # Fallback to the first data prompt if no preview*.txt is provided (works for both txt2img and edit).
            preview_prompts = [data[0].prompt]
            preview_prompt_names = [data[0].image.stem]
            using_fallback_prompts = True

        monitoring = (
            None
            if monitoring_conf is None
            else MonitoringSpec.create(
                monitoring_conf,
                preview_prompts=preview_prompts,
                preview_prompt_names=preview_prompt_names,
            )
        )
        # Statistics are always tracked (no config needed)
        statistics = StatisticsSpec()

        # Parse LoRA configuration
        targets: list[LoraTargetSpec] = []

        lora_conf = config.get("lora_layers", None) or {}
        if "targets" not in lora_conf:
            raise ValueError("Config must include lora_layers.targets[]")
        for t in lora_conf["targets"]:
            blocks = None
            if t.get("blocks") is not None:
                blocks = BlockRange(**t["blocks"])
            targets.append(
                LoraTargetSpec(
                    module_path=t["module_path"],
                    rank=t["rank"],
                    blocks=blocks,
                    alpha=t.get("alpha"),
                )
            )

        lora_layers = LoraLayersSpec(targets=targets)

        if is_edit and any(item.input_image is None for item in data):
            raise ValueError("Edit training requires input_image for every data item.")
        if is_edit and monitoring is not None:
            # For edit training, if preview prompts were explicitly provided (not fallback), require matching preview images.
            # If using fallback, use the first input image as the preview conditioning image.
            if using_fallback_prompts:
                first_input_image = data[0].input_image
                if first_input_image is None:
                    raise ValueError("Edit training fallback preview requires input_image in discovered data.")
                monitoring.preview_images = [first_input_image]
            else:
                # Require preview images for each explicit preview prompt
                if not preview_image_paths:
                    raise ValueError(
                        "Edit training with explicit preview prompts requires matching preview images. "
                        "Add data/preview*.{png,jpg,jpeg,webp} files for each preview*.txt."
                    )
                preview_images: list[Path] = []
                for name in preview_prompt_names:
                    image_path = preview_images_by_stem.get(name)
                    if image_path is None:
                        raise ValueError(f"Preview image not found for {name}.txt. Expected {name}.* in data.")
                    preview_images.append(image_path)
                monitoring.preview_images = preview_images
        if not is_edit and monitoring is not None and preview_image_paths:
            raise ValueError("data/preview.* is only supported for edit training.")

        return TrainingSpec(
            model=config["model"],
            model_path=config.get("model_path"),
            seed=config["seed"],
            steps=steps,
            guidance=guidance,
            quantize=quantize,
            max_resolution=max_resolution,
            training_loop=training_loop,
            optimizer=optimizer,
            checkpoint=checkpoint,
            monitoring=monitoring,
            lora_layers=lora_layers,
            statistics=statistics,
            data=data,
            is_edit=is_edit,
            data_root=str(data_root_dir.resolve()),
            config_path=None if absolute_config_path is None else str(absolute_config_path),
            low_ram=low_ram,
            cache_limit_gb=cache_limit_gb,
        )

    @staticmethod
    def _resolve_data_root_dir(*, data_path: str, base_path: Path | None) -> Path:
        root = Path(data_path)
        if root.is_absolute():
            return root
        if base_path is None:
            raise ValueError(
                "Config uses a relative data path, but no config_path is available to resolve it. "
                "Use an absolute data path when resuming from a checkpoint."
            )
        return Path(base_path).parent / root

    @staticmethod
    def _find_preview_prompt_files(root_dir: Path) -> list[Path]:
        prompt_paths = [p for p in root_dir.glob("preview*.txt") if p.is_file()]
        return sorted(prompt_paths, key=lambda p: p.name)

    @staticmethod
    def _load_preview_prompts(prompt_paths: list[Path]) -> list[str]:
        return [path.read_text(encoding="utf-8", errors="replace").strip() for path in prompt_paths]

    @staticmethod
    def _find_preview_images_in_data(root_dir: Path) -> dict[str, Path]:
        image_exts = {".jpg", ".jpeg", ".png", ".webp"}
        previews: dict[str, Path] = {}
        for path in root_dir.iterdir():
            if not path.is_file():
                continue
            if path.suffix.lower() not in image_exts:
                continue
            if not path.stem.startswith("preview"):
                continue
            if path.stem in previews:
                raise ValueError(f"Multiple preview images found for {path.stem}.*. Keep only one extension.")
            previews[path.stem] = path.resolve()
        return previews

    @staticmethod
    def _discover_data(
        *, data_path: str, base_path: Path | None, exclude_paths: set[Path] | None = None
    ) -> list[DataSpec]:
        root_dir = TrainingSpec._resolve_data_root_dir(data_path=data_path, base_path=base_path)
        if not root_dir.exists() or not root_dir.is_dir():
            raise ValueError(f"data path must be an existing directory when using auto-discovery: {root_dir}")

        image_exts = {".jpg", ".jpeg", ".png", ".webp"}
        excluded = set()
        if exclude_paths:
            excluded = {p.resolve() for p in exclude_paths}
        image_files = sorted(
            [
                p
                for p in root_dir.iterdir()
                if p.is_file() and p.suffix.lower() in image_exts and p.resolve() not in excluded
            ]
        )
        if not image_files:
            raise ValueError(f"No image files found in data directory: {root_dir}")

        data_items: list[DataSpec] = []
        out_files = [p for p in image_files if p.stem.endswith("_out")]
        if out_files:
            out_bases = {p.stem[: -len("_out")] for p in out_files}
            for image_path in image_files:
                if image_path.stem.endswith("_out"):
                    continue
                if image_path.stem.endswith("_in"):
                    base = image_path.stem[: -len("_in")]
                    if base not in out_bases:
                        raise ValueError(
                            f"Found input image without matching output: {image_path.name}. "
                            "Remove the stray *_in.* or add the corresponding *_out.*."
                        )
                    continue
                raise ValueError(
                    "Data folder mixes edit-style and txt2img images. Use only *_out/_in pairs or only standard images."
                )
            # Edit-style auto-discovery: match *_out.* to *_in.* and use *_in.txt for prompt.
            images_by_stem = {p.stem: p for p in image_files}
            for out_path in sorted(out_files):
                base = out_path.stem[: -len("_out")]
                input_path = images_by_stem.get(f"{base}_in")
                if input_path is None:
                    raise ValueError(f"Missing input image for '{out_path.name}'. Expected '{base}_in.*' in {root_dir}")
                prompt_file = root_dir / f"{base}_in.txt"
                if not prompt_file.exists():
                    raise ValueError(
                        f"Missing prompt file for image '{input_path.name}'. Expected '{prompt_file.name}' in {root_dir}"
                    )
                data_items.append(
                    DataSpec.create(
                        {
                            "image": out_path.name,
                            "input_image": input_path.name,
                            "prompt_file": prompt_file.name,
                        },
                        data_path,
                        base_path,
                    )
                )
            return data_items

        for image_path in image_files:
            prompt_file = image_path.with_suffix(".txt")
            if not prompt_file.exists():
                raise ValueError(
                    f"Missing prompt file for image '{image_path.name}'. Expected '{prompt_file.name}' in {root_dir}"
                )
            data_items.append(
                DataSpec.create(
                    {
                        "image": image_path.name,
                        "prompt_file": prompt_file.name,
                    },
                    data_path,
                    base_path,
                )
            )
        return data_items

    def to_json(self) -> str:
        spec_dict = asdict(self)
        serialized_dict = TrainingSpec._custom_serializer(spec_dict)
        return json.dumps(serialized_dict, indent=4)

    @staticmethod
    def _custom_serializer(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, datetime.datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(obj, list):
            return [TrainingSpec._custom_serializer(item) for item in obj]
        if isinstance(obj, dict):
            return {key: TrainingSpec._custom_serializer(value) for key, value in obj.items()}
        return obj

    @staticmethod
    def _resolve_output_path(path: str, new_folder: bool, *, create_output_dir: bool = True) -> str:
        requested_path = os.path.expanduser(path)
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if new_folder and os.path.exists(requested_path):
            requested_path = f"{requested_path}_{now_str}"
        if create_output_dir:
            os.makedirs(requested_path, exist_ok=True)
        return requested_path

    @staticmethod
    def _from_checkpoint(path: str, new_folder: bool, *, create_output_dir: bool = True) -> "TrainingSpec":
        checkpoint = ZipUtil.unzip(path, "checkpoint.json", lambda x: json.load(open(x, "r")))
        config = ZipUtil.unzip(path, checkpoint["files"]["config"], lambda x: json.load(open(x, "r")))
        run_manifest = TrainingSpec._load_run_manifest(path)
        resolved_data_root = TrainingSpec._resolve_data_root_from_manifest(
            run_manifest=run_manifest,
            checkpoint_path=path,
        )
        config["data"] = resolved_data_root
        spec = TrainingSpec.from_conf(config, None, new_folder, create_output_dir=create_output_dir)
        TrainingSpec._validate_data_fingerprint(run_manifest=run_manifest, training_spec=spec)
        spec.optimizer.state_path = checkpoint["files"]["optimizer"]
        spec.lora_layers.state_path = checkpoint["files"]["lora_adapter"]
        spec.training_loop.iterator_state_path = checkpoint["files"]["iterator"]
        if spec.statistics is None:
            spec.statistics = StatisticsSpec()
        spec.statistics.state_path = checkpoint["files"]["loss"]
        spec.checkpoint_path = path
        spec.config_path = None
        return spec

    @staticmethod
    def _load_run_manifest(path: str) -> dict:
        try:
            return ZipUtil.unzip(path, TRAINING_FILE_NAME_RUN_MANIFEST, lambda x: json.load(open(x, "r")))
        except FileNotFoundError as exc:
            raise ValueError(
                "Resume requires a checkpoint zip containing run.json. "
                "Recreate the checkpoint with the new training flow."
            ) from exc

    @staticmethod
    def _resolve_data_root_from_manifest(*, run_manifest: dict, checkpoint_path: str) -> str:
        data_root = run_manifest.get("data_root")
        if data_root and Path(data_root).exists():
            return str(Path(data_root))

        relative_root = run_manifest.get("data_root_relative")
        if relative_root:
            candidate = Path(checkpoint_path).parent / relative_root
            if candidate.exists():
                return str(candidate.resolve())

        raise ValueError(
            "Data folder not found when resuming. Place the data directory in the original "
            "location or next to the checkpoint zip."
        )

    @staticmethod
    def _validate_data_fingerprint(*, run_manifest: dict, training_spec: "TrainingSpec") -> None:
        expected_model = run_manifest.get("model")
        if expected_model is not None and expected_model != training_spec.model:
            raise ValueError(
                f"Checkpoint model does not match config. Expected '{expected_model}', got '{training_spec.model}'."
            )

        expected_is_edit = run_manifest.get("is_edit")
        if expected_is_edit is not None and bool(expected_is_edit) != training_spec.is_edit:
            raise ValueError(
                "Checkpoint edit mode does not match config. "
                f"Expected is_edit={bool(expected_is_edit)}, got is_edit={training_spec.is_edit}."
            )

        fingerprint = run_manifest.get("data_fingerprint") or {}
        expected_is_edit = fingerprint.get("is_edit")
        if expected_is_edit is not None and bool(expected_is_edit) != training_spec.is_edit:
            raise ValueError(
                "Data folder does not match the checkpoint. "
                f"Expected is_edit={bool(expected_is_edit)}, got is_edit={training_spec.is_edit}."
            )
        expected_count = fingerprint.get("count")
        if expected_count is not None and expected_count != len(training_spec.data):
            raise ValueError(
                "Data folder does not match the checkpoint. Expected "
                f"{expected_count} items, found {len(training_spec.data)}."
            )

        expected_images = fingerprint.get("images") or []
        if expected_images:
            resolved = []
            root = training_spec.data_root
            if root:
                root_path = Path(root).resolve()
                for item in training_spec.data:
                    resolved_path = item.image.resolve()
                    if resolved_path.is_relative_to(root_path):
                        resolved.append(str(resolved_path.relative_to(root_path)))
                    else:
                        resolved.append(resolved_path.name)
            if sorted(expected_images) != sorted(resolved):
                raise ValueError("Data folder does not match the checkpoint. The image list differs.")

        expected_input_images = fingerprint.get("input_images") or []
        if expected_input_images:
            resolved_inputs = []
            root = training_spec.data_root
            if root:
                root_path = Path(root).resolve()
                for item in training_spec.data:
                    if item.input_image is None:
                        continue
                    resolved_path = item.input_image.resolve()
                    if resolved_path.is_relative_to(root_path):
                        resolved_inputs.append(str(resolved_path.relative_to(root_path)))
                    else:
                        resolved_inputs.append(resolved_path.name)
            if sorted(expected_input_images) != sorted(resolved_inputs):
                raise ValueError("Data folder does not match the checkpoint. The input image list differs.")
