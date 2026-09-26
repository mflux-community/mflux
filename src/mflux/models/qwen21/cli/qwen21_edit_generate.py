import argparse
import math
from pathlib import Path

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.qwen21.reference import QwenImage21Edit
from mflux.models.qwen21.reference.latent_creator.qwen_image21_latent_creator import QwenImage21LatentCreator
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import ModelConfigError, PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil
from mflux.utils.scale_factor import ScaleFactor

CONDITIONAL_OPTIONS = {
    "--scheduler": {
        "condition": "linear Euler scheduler only",
        "reason": "Other scheduler values exit with an error before model loading.",
    },
    "--negative-prompt": {
        "condition": "guidance greater than 1",
        "reason": "Guidance 1 runs only the positive branch.",
    },
}


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate and edit RGB/RGBA images with Qwen-Image-2.1.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False, default_model="qwen-image-2.1")
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.set_defaults(width=None, height=None)
    for flag in ("--width", "--height"):
        parser._option_string_actions[flag].help = (
            "Output size: a multiple of 32 or a reference-image scale factor (auto means 1x). "
            "If omitted, derive from output-resolution and the last reference's aspect ratio."
        )
    parser.add_image_paths_arguments(required=False)
    parser.add_output_arguments()
    parser.add_argument(
        "--output-resolution",
        type=int,
        default=1024,
        help="Pixel-area budget for reference images and automatic output dimensions (default: 1024).",
    )
    parser.add_argument(
        "--use-kv-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse text and reference image prefix attention across steps (default: on).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.guidance is None or args.guidance == 1:
        CommandLineParser.warn_ignored_options(
            {"--negative-prompt": CONDITIONAL_OPTIONS["--negative-prompt"]["reason"]}
        )
    if Path(args.output).suffix.lower() not in (".png", ".webp", ".tif", ".tiff"):
        parser.error("Qwen-Image-2.1 outputs RGBA; use PNG, WebP or TIFF to retain transparency.")
    if args.scheduler != "linear":
        parser.error("Qwen-Image-2.1 currently supports the default linear Euler scheduler only.")
    paths = args.image_paths or []
    if len(paths) > 10:
        parser.error("Qwen-Image-2.1 supports at most 10 reference images.")
    try:
        guidance = args.guidance if args.guidance is not None else 1.0
        if not math.isfinite(guidance) or guidance < 1:
            raise ValueError("guidance must be finite and at least 1.")
        width, height = args.width, args.height
        if isinstance(width, ScaleFactor) or isinstance(height, ScaleFactor):
            width, height = DimensionResolver.resolve(
                width=width if width is not None else ScaleFactor(1),
                height=height if height is not None else ScaleFactor(1),
                reference_image_path=paths[-1] if paths else None,
            )
        QwenImage21LatentCreator.validate(
            width if width is not None else 32, height if height is not None else 32, args.steps, len(paths)
        )
        QwenImage21LatentCreator.validate_resolution(args.output_resolution)
        model_config = ConfigResolution.resolve_restricted(
            args.model, "qwen-image-2.1", model_path=args.model_path, base_model=args.base_model
        )
    except (ModelConfigError, ValueError) as exc:
        parser.error(str(exc))
    model = QwenImage21Edit(
        quantize=args.quantize,
        model_path=args.model_path,
        model_config=model_config,
    )
    memory_saver = CallbackManager.register_callbacks(args, model, QwenImage21LatentCreator)
    try:
        for seed in args.seed:
            image = model.generate_image(
                seed=seed,
                prompt=PromptUtil.read_prompt(args),
                negative_prompt=PromptUtil.read_negative_prompt(args),
                width=width,
                height=height,
                num_inference_steps=args.steps,
                guidance=guidance,
                image_paths=paths,
                output_resolution=args.output_resolution,
                use_kv_cache=args.use_kv_cache,
            )
            image.save(Path(args.output.format(seed=seed)), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
