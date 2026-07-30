import json
import logging
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import PIL.Image

from mflux.models.common.config import ModelConfig
from mflux.models.flux.variants.concept_attention.attention_data import ConceptHeatmap
from mflux.utils.version_util import VersionUtil

log = logging.getLogger(__name__)


class GeneratedImage:
    def __init__(
        self,
        image: PIL.Image.Image,
        model_config: ModelConfig,
        seed: int,
        prompt: str,
        steps: int,
        guidance: float | None,
        precision: mx.Dtype,
        quantization: int,
        generation_time: float,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        height: int | None = None,
        width: int | None = None,
        controlnet_image_path: str | Path | None = None,
        controlnet_strength: float | None = None,
        image_path: str | Path | None = None,
        image_paths: list[str] | list[Path] | None = None,
        image_strength: float | None = None,
        masked_image_path: str | Path | None = None,
        depth_image_path: str | Path | None = None,
        redux_image_paths: list[str] | list[Path] | None = None,
        redux_image_strengths: list[float] | None = None,
        concept_heatmap: ConceptHeatmap | None = None,
        negative_prompt: str | None = None,
        init_metadata: dict | None = None,
        pid_decode: bool = False,
        pid_degrade_sigma: float = 0.0,
    ):
        self.image = image
        self.model_config = model_config
        self.seed = seed
        self.prompt = prompt
        self.steps = steps
        self.guidance = guidance
        self.precision = precision
        self.quantization = quantization
        self.generation_time = generation_time
        self.lora_paths = lora_paths
        self.lora_scales = lora_scales
        self.height = height
        self.width = width
        self.controlnet_image_path = controlnet_image_path
        self.controlnet_strength = controlnet_strength
        self.image_path = image_path
        self.image_paths = image_paths
        self.image_strength = image_strength
        self.masked_image_path = masked_image_path
        self.depth_image_path = depth_image_path
        self.redux_image_paths = redux_image_paths
        self.redux_image_strengths = redux_image_strengths
        self.concept_heatmap = concept_heatmap
        self.negative_prompt = negative_prompt
        self.init_metadata = init_metadata
        self.pid_decode = pid_decode
        self.pid_degrade_sigma = pid_degrade_sigma

    def get_right_half(self) -> "GeneratedImage":
        # Calculate the coordinates for the right half
        width, height = self.image.size
        right_half = self.image.crop((width // 2, 0, width, height))

        # Create a new GeneratedImage with the right half and the same metadata
        return GeneratedImage(
            image=right_half,
            model_config=self.model_config,
            seed=self.seed,
            prompt=self.prompt,
            steps=self.steps,
            guidance=self.guidance,
            precision=self.precision,
            quantization=self.quantization,
            generation_time=self.generation_time,
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            height=self.height,
            width=self.width,
            controlnet_image_path=self.controlnet_image_path,
            controlnet_strength=self.controlnet_strength,
            image_path=self.image_path,
            image_strength=self.image_strength,
            masked_image_path=self.masked_image_path,
            depth_image_path=self.depth_image_path,
            concept_heatmap=self.concept_heatmap,
            init_metadata=self.init_metadata,
            pid_decode=self.pid_decode,
            pid_degrade_sigma=self.pid_degrade_sigma,
        )

    def save(
        self,
        path: str | Path,
        export_json_metadata: bool = False,
        overwrite: bool = False,
    ) -> None:
        from mflux.utils.image_util import ImageUtil

        final_path = ImageUtil.resolve_output_path(path=path, overwrite=overwrite)

        if self._should_save_fibo_prompt_sidecar():
            self._save_prompt_file(final_path, overwrite=True)

        ImageUtil.save_image(self.image, final_path, self._get_metadata(), export_json_metadata, overwrite=True)

    def save_with_image(
        self,
        path: str | Path,
        pixel_image: PIL.Image.Image,
        export_json_metadata: bool = False,
        overwrite: bool = False,
    ) -> None:
        from mflux.utils.image_util import ImageUtil

        final_path = ImageUtil.resolve_output_path(path=path, overwrite=overwrite)

        if self._should_save_fibo_prompt_sidecar():
            self._save_prompt_file(final_path, overwrite=True)

        ImageUtil.save_image(pixel_image, final_path, self._get_metadata(), export_json_metadata, overwrite=True)

    def save_with_heatmap(
        self,
        path: str | Path,
        export_json_metadata: bool = False,
        overwrite: bool = False,
    ) -> None:
        # Save the main image
        self.save(path=path, export_json_metadata=export_json_metadata, overwrite=overwrite)

        # Save the concept heatmap if available
        if self.concept_heatmap:
            file_path = Path(path)
            heatmap_path = file_path.with_stem(file_path.stem + "_heatmap")
            self.save_concept_heatmap(path=heatmap_path, export_json_metadata=export_json_metadata, overwrite=overwrite)

    def save_concept_heatmap(
        self,
        path: str | Path,
        export_json_metadata: bool = False,
        overwrite: bool = False,
    ) -> None:
        if self.concept_heatmap:
            from mflux.utils.image_util import ImageUtil

            ImageUtil.save_image(
                image=self.concept_heatmap.image,
                path=path,
                metadata=self.concept_heatmap.get_metadata(),
                export_json_metadata=export_json_metadata,
                overwrite=overwrite,
            )
        else:
            raise ValueError("No concept heatmap available to save")

    def _format_redux_strengths(self) -> list[float] | None:
        if not self.redux_image_strengths:
            return None
        return [round(scale, 2) for scale in self.redux_image_strengths]

    def _should_save_fibo_prompt_sidecar(self) -> bool:
        name = self.model_config.model_name
        if name == "briaai/Fibo-Edit-RMBG":
            return False
        return (
            name == "briaai/FIBO"
            or name == "briaai/Fibo-lite"
            or name == "briaai/Fibo-Edit"
            or str(self.model_config.base_model) == "fibo"
        )

    def _save_prompt_file(self, image_path: str | Path, overwrite: bool) -> None:
        file_path = Path(image_path)
        # For FIBO models, use .json instead of .prompt.json
        prompt_path = file_path.with_suffix(".json")

        # Handle overwrite logic similar to image saving
        if not overwrite:
            counter = 1
            while prompt_path.exists():
                new_name = f"{file_path.stem}_{counter}.json"
                prompt_path = file_path.parent / new_name
                counter += 1

        try:
            # Parse and pretty-print the JSON prompt
            try:
                prompt_json = json.loads(self.prompt)
                with open(prompt_path, "w") as f:
                    json.dump(prompt_json, f, indent=2, ensure_ascii=False)
            except (json.JSONDecodeError, ValueError):
                # If prompt is not valid JSON, save as-is (shouldn't happen for FIBO)
                with open(prompt_path, "w") as f:
                    f.write(self.prompt)

            log.info(f"Prompt file saved successfully at: {prompt_path}")
        except Exception as e:  # noqa: BLE001
            log.error(f"Error saving prompt file: {e}")

    def _get_metadata(self) -> dict:
        metadata = {
            "mflux_version": VersionUtil.get_mflux_version(),
            "model": self.model_config.model_name,
            "base_model": str(self.model_config.base_model),
            "seed": self.seed,
            "steps": self.steps,
            "guidance": self.guidance if self.model_config.supports_guidance else None,
            "height": self.height,
            "width": self.width,
            "precision": str(self.precision),
            "quantize": self.quantization,
            "generation_time_seconds": round(self.generation_time, 2),
            "created_at": datetime.now().isoformat(),
            "lora_paths": [str(p) for p in self.lora_paths] if self.lora_paths else None,
            "lora_scales": [round(scale, 2) for scale in self.lora_scales] if self.lora_scales else None,
            "image_path": str(self.image_path) if self.image_path else None,
            "image_paths": [str(p) for p in self.image_paths] if self.image_paths else None,
            "image_strength": self.image_strength if (self.image_path or self.image_paths) else None,
            "controlnet_image_path": str(self.controlnet_image_path) if self.controlnet_image_path else None,
            "controlnet_strength": round(self.controlnet_strength, 2) if self.controlnet_strength else None,
            "masked_image_path": str(self.masked_image_path) if self.masked_image_path else None,
            "depth_image_path": str(self.depth_image_path) if self.depth_image_path else None,
            "redux_image_paths": str(self.redux_image_paths) if self.redux_image_paths else None,
            "redux_image_strengths": self._format_redux_strengths(),
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt if self.negative_prompt else None,
            "pid_decode": self.pid_decode,
            "pid_degrade_sigma": self.pid_degrade_sigma if self.pid_decode else None,
        }

        # If we have initial metadata from a source image, merge it
        if self.init_metadata and (old_exif := self.init_metadata.get("exif")):
            # 1. If we don't have a prompt (e.g. SeedVR2), use the original prompt
            if not metadata.get("prompt") and old_exif.get("prompt"):
                metadata["prompt"] = old_exif.get("prompt")

            # 2. If we don't have a negative prompt, use the original one
            if not metadata.get("negative_prompt") and old_exif.get("negative_prompt"):
                metadata["negative_prompt"] = old_exif.get("negative_prompt")

            # 3. Store original dimensions
            if old_exif.get("width") and old_exif.get("width") != self.width:
                metadata["original_width"] = old_exif.get("width")
            if old_exif.get("height") and old_exif.get("height") != self.height:
                metadata["original_height"] = old_exif.get("height")

            # 4. Carry over other metadata that doesn't conflict and is useful
            fields_to_carry = [
                "seed",
                "steps",
                "guidance",
                "model",
                "lora_paths",
                "lora_scales",
                "quantize",
            ]
            for field in fields_to_carry:
                if old_val := old_exif.get(field):
                    metadata[f"original_{field}"] = old_val

        return metadata
