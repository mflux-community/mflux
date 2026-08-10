import argparse
import json
import random
import sys
import time
import typing as t
from pathlib import Path

from mflux.cli.defaults import defaults as ui_defaults
from mflux.models.common.resolution.lora_resolution import LoraResolution
from mflux.models.flux.variants.in_context.utils.in_context_loras import LORA_NAME_MAP
from mflux.utils import box_values, scale_factor


class ModelSpecAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)


def int_or_special_value(value) -> int | scale_factor.ScaleFactor:
    if value.lower() == "auto":
        return scale_factor.ScaleFactor(value=1)

    # Try to parse as integer first
    try:
        return int(value)
    except ValueError:
        pass

    # If not an integer, try to parse as scale factor
    try:
        return scale_factor.ScaleFactor.parse(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"'{value}' is not a valid integer or 'auto' or a scale factor like '2x' or '3.5x'"
        )


def lora_init_kwargs_from_args(args: argparse.Namespace) -> dict[str, t.Any]:
    return {
        "lora_paths": args.lora_paths,
        "lora_scales": args.lora_scales,
        "bake_lora": args.bake_lora,
    }


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid number")
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"'{value}' must be > 0")
    return parsed


# fmt: off
class CommandLineParser(argparse.ArgumentParser):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supports_metadata_config = False
        self.supports_image_generation = False
        self.supports_controlnet = False
        self.supports_dimension_scale_factor = False
        self.supports_image_to_image = False
        self.supports_image_outpaint = False
        self.supports_lora = False
        self.require_model_arg = True
        self.require_init_image = False

    def add_general_arguments(self) -> None:
        self.add_argument("--battery-percentage-stop-limit", "-B", type=lambda v: max(min(int(v), 99), 1), default=ui_defaults.BATTERY_PERCENTAGE_STOP_LIMIT, help=f"On Macs powered by battery, stop image generation when battery reaches this percentage. Default: {ui_defaults.BATTERY_PERCENTAGE_STOP_LIMIT}")
        self.add_argument("--low-ram", action="store_true", help="Enable low-RAM mode to reduce memory usage (may impact performance).")
        self.add_argument("--mlx-cache-limit-gb", type=positive_float, default=None, help="Limit MLX cache size in GB without enabling full low-RAM mode (e.g. 8 or 16).")

    def add_seedvr2_upscale_arguments(self) -> None:
        self.supports_image_generation = True
        self.require_prompt = False
        seedvr2_group = self.add_argument_group("SeedVR2 upscale configuration")
        seedvr2_group.add_argument(
            "--image-path",
            "-i",
            type=Path,
            required=True,
            nargs="+",
            help="Path to the input image(s) or directories to upscale.",
        )
        seedvr2_group.add_argument("--seed", "-s", type=int, default=[42], nargs="+", help="Random seed(s) for reproducibility.")
        seedvr2_group.add_argument("--resolution", "-r", type=int_or_special_value, default=384, help="Target resolution for the shortest edge (pixels) or scale factor (e.g., '2x').")
        seedvr2_group.add_argument("--softness", type=float, default=0.0, help="Value between 0.0 (off, factor 1) and 1.0 (max, factor 8). Default: 0.0.")

    def add_model_arguments(self, path_type: t.Literal["load", "save"] = "load", require_model_arg: bool = True) -> None:
        self.require_model_arg = require_model_arg
        self.add_argument("--model", "-m", type=str, required=require_model_arg, action=ModelSpecAction, help=f"The model to use ({' or '.join(ui_defaults.MODEL_CHOICES)}, a HuggingFace repo org/model, or a local path).")
        if path_type == "save":
            self.add_argument("--path", type=str, required=True, help="Local path for saving a model to disk.")
        self.add_argument("--base-model", type=str, required=False, choices=ui_defaults.MODEL_CHOICES, help="When using a third-party huggingface model, explicitly specify whether the base model is dev or schnell")
        self.add_argument("--quantize",  "-q", type=int, choices=ui_defaults.QUANTIZE_CHOICES, default=None, help=f"Quantize the model ({' or '.join(map(str, ui_defaults.QUANTIZE_CHOICES))}, Default is None)")

    def add_lora_arguments(self) -> None:
        self.supports_lora = True
        lora_group = self.add_argument_group("LoRA configuration")
        lora_group.add_argument("--lora-style", type=str, choices=sorted(LORA_NAME_MAP.keys()), help="Style of the LoRA to use (e.g., 'storyboard' for film storyboard style)")
        lora_group.add_argument("--lora", dest="lora", action="append", nargs="+", default=None, metavar=("PATH", "SCALE"), help="Add a LoRA as an atomic PATH with optional SCALE (default 1.0). Repeatable: --lora A.safetensors 0.7 --lora B.safetensors. PATH accepts local files, HuggingFace repos (org/model), or collection format (repo:filename.safetensors). Preferred over --lora-paths/--lora-scales.")
        self.add_argument("--lora-paths", type=str, nargs="*", default=None, help="[DEPRECATED: use --lora] LoRA paths: local files, HuggingFace repos (org/model), or collection format (repo:filename.safetensors)")
        self.add_argument("--lora-scales", type=float, nargs="*", default=None, help="[DEPRECATED: use --lora] Scaling factor to adjust the impact of LoRA weights on the model. A value of 1.0 applies the LoRA weights as they are.")
        lora_group.add_argument(
            "--bake-lora",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Merge LoRA/LoKr deltas into base weights after load (default: on). Use --no-bake-lora to keep runtime adapters.",
        )

    def _add_image_generator_common_arguments(self, supports_dimension_scale_factor=False) -> None:
        self.supports_image_generation = True
        if supports_dimension_scale_factor:
            self.supports_dimension_scale_factor = True
            self.add_argument("--height", type=int_or_special_value, default="auto", help="Image height (Default is source image height)")
            self.add_argument("--width", type=int_or_special_value, default="auto", help="Image width (Default is source image width)")
        else:
            self.add_argument("--height", type=int, default=ui_defaults.HEIGHT, help=f"Image height (Default is {ui_defaults.HEIGHT})")
            self.add_argument("--width", type=int, default=ui_defaults.WIDTH, help=f"Image width (Default is {ui_defaults.HEIGHT})")

        self.add_argument("--steps", type=int, default=None, help="Inference Steps")
        self.add_argument("--guidance", type=float, default=None, help=f"Guidance Scale (Default varies by tool: {ui_defaults.GUIDANCE_SCALE} for most, {ui_defaults.DEFAULT_DEV_FILL_GUIDANCE} for fill tools, {ui_defaults.DEFAULT_DEPTH_GUIDANCE} for depth)")

    def add_image_generator_arguments(self, supports_metadata_config=False, require_prompt=True, supports_dimension_scale_factor=False) -> None:
        prompt_group = self.add_mutually_exclusive_group(required=(require_prompt and not supports_metadata_config))
        prompt_group.add_argument("--prompt", type=str, help="The textual description of the image to generate.")
        prompt_group.add_argument("--prompt-file", type=Path, help="Path to a file containing the prompt text. The file will be re-read before each generation, allowing you to edit the prompt between iterations when using multiple seeds without restarting the program.")
        self.add_argument("--negative-prompt", type=str, default="", help="The negative prompt to guide what the model should not generate.")
        self.add_argument("--seed", type=int, default=None, nargs='+', help="Specify 1+ Entropy Seeds (Default is 1 time-based random-seed)")
        self.add_argument("--auto-seeds", type=int, default=-1, help="Auto generate N Entropy Seeds (random ints between 0 and 1 billion")
        self.add_argument("--scheduler", type=str, default="linear", help="Choose from implemented schedulers (linear only for now). Or bring your own: 'your_package.some_module.FooScheduler'")
        self._add_image_generator_common_arguments(supports_dimension_scale_factor=supports_dimension_scale_factor)
        if supports_metadata_config:
            self.add_metadata_config()
        self.require_prompt = require_prompt

    def add_image_to_image_arguments(self, required=False) -> None:
        self.supports_image_to_image = True
        # The requirement is enforced after normalization (see parse_args) so it can be
        # satisfied by either the new --image flag or the legacy --image-path flag.
        self.require_init_image = required
        self.add_argument("--image", dest="image", nargs="+", default=None, metavar=("PATH", "STRENGTH"), help=f"Init image as an atomic PATH with optional STRENGTH (default {ui_defaults.IMAGE_STRENGTH}): --image photo.jpg 0.6. Preferred over --image-path/--image-strength.")
        self.add_argument("--image-path", type=Path, required=False, default=None, help="[DEPRECATED: use --image] Local path to init image")
        self.add_argument("--image-strength", type=float, required=False, default=ui_defaults.IMAGE_STRENGTH, help=f"[DEPRECATED: use --image] Controls how strongly the init image influences the output image. A value of 0.0 means no influence. (Default is {ui_defaults.IMAGE_STRENGTH})")

    def add_batch_image_generator_arguments(self) -> None:
        self.add_argument("--batch-prompts-file", type=Path, required=True, default=argparse.SUPPRESS, help="Local path for a file that holds a batch of prompts.")
        self.add_argument("--global-seed", type=int, default=argparse.SUPPRESS, help="Entropy Seed (used for all prompts in the batch)")
        self._add_image_generator_common_arguments()

    def add_fill_arguments(self) -> None:
        self.add_argument("--image-path", type=Path, required=True, help="Local path to the source image")
        self.add_argument("--masked-image-path", type=Path, required=True, help="Local path to the mask image")

    def add_catvton_arguments(self) -> None:
        self.add_argument("--person-image", type=str, required=True, help="Path to person image")
        self.add_argument("--person-mask", type=str, required=True, help="Path to person mask image")
        self.add_argument("--garment-image", type=str, required=True, help="Garment Image")

    def add_in_context_edit_arguments(self) -> None:
        self.supports_in_context_edit = True
        self.add_argument("--reference-image", type=str, required=True, help="Path to reference image")
        self.add_argument("--instruction", type=str, help="User instruction to be wrapped in diptych template (e.g., 'make the hair black'). This will be automatically formatted as 'A diptych with two side-by-side images of the same scene. On the right, the scene is exactly the same as on the left but {instruction}'. Either --instruction or --prompt is required.")  # fmt:off

    def add_in_context_arguments(self) -> None:
        self.add_argument("--save-full-image", action="store_true", default=False, help="Additionally, save the full image containing the reference image. Useful for verifying the in-context usage of the reference image.")

    def add_in_context_dev_arguments(self) -> None:
        self.add_argument("--reference-image", type=Path, required=True, dest="image_path", help="Path to reference image")

    def add_depth_arguments(self) -> None:
        self.add_argument("--image-path", type=Path, required=False, help="Local path to the source image")
        self.add_argument("--depth-image-path", type=Path, required=False, help="Local path to the depth image")
        self.add_argument("--save-depth-map", action="store_true", required=False, help="If set, save the depth map created from the source image.")

    def add_save_depth_arguments(self) -> None:
        self.add_argument("--image-path", type=Path, required=True, help="Local path to the source image")
        self.add_argument("--quantize",  "-q", type=int, choices=ui_defaults.QUANTIZE_CHOICES, default=None, required=False, help=f"Quantize the model ({' or '.join(map(str, ui_defaults.QUANTIZE_CHOICES))}, Default is None)")

    def add_redux_arguments(self) -> None:
        self.add_argument("--redux-image-paths", type=Path, nargs="*", required=True, help="Local path to the source image")
        self.add_argument("--redux-image-strengths", type=float, nargs="*", default=None, help="Strength values (between 0.0 and 1.0) for each reference image. Default is 1.0 for all images.")

    def add_output_arguments(self) -> None:
        self.add_argument("--metadata", action="store_true", help="Export image metadata as a JSON file.")
        self.add_argument("--output", type=str, default="image.png", help="The filename for the output image. Default is \"image.png\".")
        self.add_argument('--stepwise-image-output-dir', type=str, default=None, help='[EXPERIMENTAL] Output dir to write step-wise images and their final composite image to. This feature may change in future versions.')

    def add_image_outpaint_arguments(self, required=False) -> None:
        self.supports_image_outpaint = True
        self.add_argument("--image-outpaint-padding", type=str, default=None, required=required, help="For outpainting mode: CSS-style box padding values to extend the canvas of image specified by--image-path. E.g. '20', '50%%'")

    def add_controlnet_arguments(self, mode: str | None = None, require_image=False) -> None:
        self.supports_controlnet = True
        self.add_argument("--controlnet-image-path", type=str, required=require_image, help="Local path of the image to use as input for controlnet.")
        self.add_argument("--controlnet-strength", type=float, default=ui_defaults.CONTROLNET_STRENGTH, help=f"Controls how strongly the control image influences the output image. A value of 0.0 means no influence. (Default is {ui_defaults.CONTROLNET_STRENGTH})")
        if mode == 'canny':
            self.add_argument("--controlnet-save-canny", action="store_true", help="If set, save the Canny edge detection reference input image.")

    def add_concept_attention_arguments(self) -> None:
        concept_group = self.add_argument_group("Concept Attention configuration")
        concept_group.add_argument("--concept", type=str, required=True, help="The concept prompt to use for attention visualization")
        concept_group.add_argument("--input-image-path", type=Path, required=False, default=None, help="Local path to reference image for concept attention analysis (uses Flux1ConceptFromImage instead of text-based concept)")
        concept_group.add_argument("--heatmap-layer-indices", type=int, nargs="*", default=list(range(15, 19)), help="Layer indices to use for heatmap generation (default: 15-18)")
        concept_group.add_argument("--heatmap-timesteps", type=int, nargs="*", default=None, help="Timesteps to use for heatmap generation (default: all timesteps)")

    def add_concept_from_image_arguments(self) -> None:
        concept_group = self.add_argument_group("Concept Attention from Image configuration")
        concept_group.add_argument("--concept", type=str, required=True, help="The concept prompt to use for attention visualization")
        concept_group.add_argument("--input-image-path", type=Path, required=True, help="Local path to reference image for concept attention analysis")
        concept_group.add_argument("--heatmap-layer-indices", type=int, nargs="*", default=list(range(15, 19)), help="Layer indices to use for heatmap generation (default: 15-18)")
        concept_group.add_argument("--heatmap-timesteps", type=int, nargs="*", default=None, help="Timesteps to use for heatmap generation (default: all timesteps)")

    def add_metadata_config(self) -> None:
        self.supports_metadata_config = True
        self.add_argument("--config-from-metadata", "-C", type=Path, required=False, default=argparse.SUPPRESS, help="Re-use the parameters from prior metadata. Params from metadata are secondary to other args you provide.")

    def add_training_arguments(self) -> None:
        train_group = self.add_mutually_exclusive_group(required=True)
        train_group.add_argument(
            "--config",
            dest="config",
            type=Path,
            required=False,
            help="Local path of the training configuration file.",
        )
        train_group.add_argument(
            "--resume",
            dest="resume",
            type=Path,
            required=False,
            help="Path to a training checkpoint zip to resume.",
        )
        self.add_argument(
            "--dry-run",
            action="store_true",
            help="Validate training config/checkpoint and exit.",
        )

    def add_info_arguments(self) -> None:
        self.add_argument("image_path", type=str, help="Path to the image file to inspect")

    @staticmethod
    def _option_was_provided(*option_names: str) -> bool:
        argv = sys.argv[1:]
        for token in argv:
            if token in option_names:
                return True
            for option in option_names:
                if option.startswith("--") and token.startswith(f"{option}="):
                    return True
                if option.startswith("-") and not option.startswith("--") and len(option) == 2 and token.startswith(option) and token != option:
                    return True
        return False

    def _normalize_atomic_lora_args(self, namespace: argparse.Namespace) -> None:
        if not self.supports_lora or not hasattr(namespace, "lora") or namespace.lora is None:
            return
        if self._option_was_provided("--lora-paths", "--lora-scales"):
            self.error("Use either --lora or the legacy --lora-paths/--lora-scales, not both.\nTip: --lora pairs each path with its scale: --lora A.safetensors 0.7 --lora B.safetensors 0.4")  # fmt: off
        paths: list[str] = []
        scales: list[float] = []
        for group in namespace.lora:
            if len(group) == 1:
                path, scale = group[0], 1.0
            elif len(group) == 2:
                path = group[0]
                try:
                    scale = float(group[1])
                except ValueError:
                    self.error(f"Invalid LoRA scale '{group[1]}' for '{group[0]}'.\nTip: --lora takes a PATH and an optional numeric SCALE: --lora {group[0]} 0.7")  # fmt: off
            else:
                self.error(f"--lora takes one PATH and an optional SCALE but got {len(group)} values: {' '.join(group)}\nTip: give each adapter its own --lora: --lora A.safetensors 0.7 --lora B.safetensors")  # fmt: off
            paths.append(path)
            scales.append(scale)
        namespace.lora_paths = paths
        namespace.lora_scales = scales

    def _normalize_atomic_image_args(self, namespace: argparse.Namespace) -> None:
        if not hasattr(namespace, "image") or namespace.image is None:
            return
        if self._option_was_provided("--image-path", "--image-strength"):
            self.error("Use either --image or the legacy --image-path/--image-strength, not both.\nTip: --image pairs the path with its strength: --image photo.jpg 0.6")  # fmt: off
        group = namespace.image
        if len(group) == 1:
            namespace.image_path = Path(group[0])
        elif len(group) == 2:
            namespace.image_path = Path(group[0])
            try:
                namespace.image_strength = float(group[1])
            except ValueError:
                self.error(f"Invalid image strength '{group[1]}' for '{group[0]}'.\nTip: --image takes a PATH and an optional numeric STRENGTH: --image {group[0]} 0.6")  # fmt: off
        else:
            self.error(f"--image takes one PATH and an optional STRENGTH but got {len(group)} values: {' '.join(group)}\nTip: --image photo.jpg 0.6")  # fmt: off

    def parse_args(self) -> argparse.Namespace:  # type: ignore
        namespace = super().parse_args()

        # Fold the atomic --lora / --image flags into the legacy lora_paths/lora_scales
        # and image_path/image_strength fields so all downstream logic (metadata merge,
        # path resolution, model init) stays unchanged. Runs before the metadata block.
        self._normalize_atomic_lora_args(namespace)
        self._normalize_atomic_image_args(namespace)

        # Check if either training arguments are provided
        has_training_args = (hasattr(namespace, "config") and namespace.config is not None) or \
                            (hasattr(namespace, "resume") and namespace.resume is not None)

        # Only enforce model requirement for path if we're not in training mode
        if hasattr(namespace, "path") and namespace.path is not None and namespace.model is None and not has_training_args:
            self.error("--model must be specified when using --path")

        if getattr(namespace, "config_from_metadata", False):
            prior_gen_metadata = json.load(namespace.config_from_metadata.open("rt"))

            if hasattr(namespace, "model") and not self._option_was_provided("--model", "-m"):
                # When --model was not provided explicitly, metadata should win
                # even if the parser set a command-specific default model.
                namespace.model = prior_gen_metadata.get("model", namespace.model)

            if namespace.base_model is None:
                namespace.base_model = prior_gen_metadata.get("base_model", None)

            if namespace.prompt is None:
                namespace.prompt = prior_gen_metadata.get("prompt", None)

            # all configs from the metadata config defers to any explicitly defined args
            guidance_default = self.get_default("guidance")
            guidance_from_metadata = prior_gen_metadata.get("guidance")
            if namespace.guidance == guidance_default and guidance_from_metadata:
                namespace.guidance = guidance_from_metadata
            if namespace.quantize is None:
                namespace.quantize = prior_gen_metadata.get("quantize", None)
            seed_from_metadata = prior_gen_metadata.get("seed", None)
            if namespace.seed is None and seed_from_metadata is not None:
                namespace.seed = [seed_from_metadata]

            if namespace.seed is None:
                # not passed by user, not populated by metadata
                namespace.seed = [int(time.time())]

            if namespace.steps is None:
                namespace.steps = prior_gen_metadata.get("steps", None)

            if self.supports_lora:
                if namespace.lora_paths is None:
                    namespace.lora_paths = prior_gen_metadata.get("lora_paths", None)
                elif namespace.lora_paths:
                    # merge the loras from cli and config file
                    namespace.lora_paths = prior_gen_metadata.get("lora_paths", []) + namespace.lora_paths

                if namespace.lora_scales is None:
                    namespace.lora_scales = prior_gen_metadata.get("lora_scales", None)
                elif namespace.lora_scales:
                    # merge the loras from cli and config file
                    namespace.lora_scales = prior_gen_metadata.get("lora_scales", []) + namespace.lora_scales

            if hasattr(namespace, "image_path") and namespace.image_path is None:
                namespace.image_path = prior_gen_metadata.get("image_path", None)

            if hasattr(namespace, "mask_path") and namespace.mask_path is None:
                namespace.mask_path = (
                    prior_gen_metadata.get("masked_image_path", None) or prior_gen_metadata.get("mask_path", None)
                )

            if self.supports_image_to_image:
                if namespace.image_strength == self.get_default("image_strength") and (img_strength_from_metadata := prior_gen_metadata.get("image_strength", None)):
                    namespace.image_strength = img_strength_from_metadata

            if self.supports_controlnet:
                if namespace.controlnet_image_path is None:
                    namespace.controlnet_image_path = prior_gen_metadata.get("controlnet_image_path", None)
                if namespace.controlnet_strength == self.get_default("controlnet_strength") and (cnet_strength_from_metadata := prior_gen_metadata.get("controlnet_strength", None)):
                    namespace.controlnet_strength = cnet_strength_from_metadata
                if namespace.controlnet_save_canny == self.get_default("controlnet_save_canny") and (cnet_canny_from_metadata := prior_gen_metadata.get("controlnet_save_canny", None)):
                    namespace.controlnet_save_canny = cnet_canny_from_metadata


            if self.supports_image_outpaint:
                if namespace.image_outpaint_padding is None:
                    namespace.image_outpaint_padding = prior_gen_metadata.get("image_outpaint_padding", None)

        # Only require model if we're not in training mode and require_model_arg is True
        if hasattr(namespace, "model") and namespace.model is None and not has_training_args and self.require_model_arg:
            self.error("--model / -m must be provided, or 'model' must be specified in the config file.")

        if self.require_init_image and getattr(namespace, "image_path", None) is None:
            self.error("An init image is required. Provide one with --image PATH [STRENGTH] (e.g. --image photo.jpg 0.8).")

        if self.supports_image_generation and namespace.seed is None and namespace.auto_seeds > 0:
            # choose N unique int seeds in the range of  0 < value < 1 billion
            # Use random.sample to guarantee uniqueness
            max_seed_value = int(1e7)
            if namespace.auto_seeds > max_seed_value + 1:
                # If requesting more seeds than possible unique values, allow duplicates
                namespace.seed = [random.randint(0, max_seed_value) for _ in range(namespace.auto_seeds)]
            else:
                namespace.seed = random.sample(range(max_seed_value + 1), namespace.auto_seeds)

        if self.supports_image_generation and namespace.seed is None:
            # final default: did not obtain seed from metadata, --seed, or --auto-seeds
            namespace.seed = [int(time.time())]

        if self.supports_image_generation and len(namespace.seed) > 1:
            # auto append seed-$value to output names for multi image generations
            # e.g. output.png -> output_seed_101.png output_seed_102.png, etc
            output_path = Path(namespace.output)
            namespace.output = str(output_path.with_stem(output_path.stem + "_seed_{seed}"))

        if hasattr(namespace, "image_path") and isinstance(namespace.image_path, list) and len(namespace.image_path) > 1:
            # auto append image-$name to output names for multi image generations
            output_path = Path(namespace.output)
            namespace.output = str(output_path.with_stem(output_path.stem + "_{image_name}"))

        if self.supports_image_generation and getattr(namespace, "prompt", None) is None and getattr(namespace, "prompt_file", None) is None:
            # when metadata config is supported but neither prompt nor prompt-file is provided
            # Only error if prompt is actually required
            if getattr(self, 'require_prompt', True):
                self.error("Either --prompt or --prompt-file argument is required, or 'prompt' required in metadata config file")

        if self.supports_image_generation and getattr(namespace, "steps", None) is None:
            model_name = getattr(namespace, "model", None)
            namespace.steps = ui_defaults.MODEL_INFERENCE_STEPS.get(model_name, 25)

        # In-context edit specific validations
        if getattr(self, 'supports_in_context_edit', False):
            if not getattr(namespace, 'prompt', None) and not getattr(namespace, 'instruction', None):
                self.error("Either --prompt or --instruction argument is required for in-context editing")

            if getattr(namespace, 'prompt', None) and getattr(namespace, 'instruction', None):
                self.error("Cannot use both --prompt and --instruction. Choose one.")

        if self.supports_image_outpaint and namespace.image_outpaint_padding is not None:
            # parse and normalize any acceptable 1,2,3,4-tuple box value to 4-tuple
            namespace.image_outpaint_padding = box_values.BoxValues.parse(namespace.image_outpaint_padding)
            print(f"{namespace.image_outpaint_padding=}")

        # Resolve lora paths from library if needed
        if self.supports_lora and hasattr(namespace, "lora_paths") and namespace.lora_paths:
            resolved_paths = []
            for lora_path in namespace.lora_paths:
                try:
                    resolved_path = LoraResolution.resolve(lora_path)
                    resolved_paths.append(resolved_path)
                except (FileNotFoundError, ValueError) as e:  # noqa: PERF203
                    self.error(str(e))
            namespace.lora_paths = resolved_paths

        # Compute model_path: None for predefined names, otherwise use the model value
        # Predefined names like "schnell", "dev" are handled by ModelConfig, not PathResolution
        if hasattr(namespace, "model") and namespace.model is not None:
            namespace.model_path = None if namespace.model in ui_defaults.MODEL_CHOICES else namespace.model
        else:
            namespace.model_path = None

        return namespace
