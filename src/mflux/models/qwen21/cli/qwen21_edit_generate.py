import argparse
from pathlib import Path

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.qwen21.latent_creator.qwen21_latent_creator import Qwen21LatentCreator
from mflux.models.qwen21.variants.edit.qwen_image_21_edit import QwenImage21Edit
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil

DEFAULT_MODEL = "qwen-image-2.1"


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Edit images using Qwen Image 2.1 with natural-language instructions.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False, default_model=DEFAULT_MODEL)
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_image_paths_arguments()
    parser.add_output_arguments()
    parser.add_argument(
        "--mask-image",
        type=str,
        default=None,
        help="Path to an inpaint mask (white = repaint, black = preserve) aligned with the first condition image.",
    )
    parser.add_argument(
        "--auto-mask",
        type=str,
        default=None,
        help="Describe an object to mask ('the red shirt'); the in-memory Qwen3-VL locates it. Ignored with --mask-image.",
    )
    parser.add_argument(
        "--use-step-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip unchanged transformer blocks on nearby denoising steps (faster, slightly different output).",
    )
    parser.add_argument(
        "--step-cache-threshold",
        type=float,
        default=0.12,
        help="Step-cache aggressiveness: higher skips more (default: 0.12).",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="Edit strength in (0, 1]: 1.0 fully re-denoises from noise, lower values "
        "start from the reference partway down the schedule for subtler edits.",
    )
    parser.add_argument(
        "--enhance-prompt",
        action="store_true",
        help="Rewrite the instruction into a detailed prompt first (official serving recipe).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After generating, self-check the result with the built-in Qwen3-VL.",
    )
    parser.add_argument(
        "--verify-retries",
        type=int,
        default=0,
        help="Regenerate with a new seed when verification fails, at most this many times.",
    )
    parser.add_argument(
        "--rgba-output",
        action="store_true",
        help="Keep the decoder's alpha channel and save RGBA (transparent stickers); "
        "requires a PNG/WebP/TIFF --output.",
    )
    return parser


def validate_args(parser: CommandLineParser, args) -> None:
    """Cheap checks that must fail BEFORE the ~33 GB model load."""
    image_paths = args.image_paths or []
    if not image_paths:
        parser.error("at least one --image-paths image is required")
    if len(image_paths) > 10:
        parser.error(f"Qwen-Image-2.1 supports at most 10 reference images, got {len(image_paths)}")
    missing = [p for p in image_paths if not Path(p).exists()]
    if missing:
        parser.error(f"condition image(s) not found: {missing}")
    if args.mask_image is not None and not Path(args.mask_image).exists():
        parser.error(f"--mask-image not found: {args.mask_image}")
    if not 0.0 < args.strength <= 1.0:
        parser.error(f"--strength must be in (0, 1], got {args.strength}")
    if args.scheduler != "linear":
        parser.error("Qwen-Image-2.1 editing currently supports the default linear Euler scheduler only.")
    if args.rgba_output and Path(str(args.output)).suffix.lower() not in (".png", ".webp", ".tif", ".tiff"):
        parser.error("--rgba-output needs a PNG, WebP or TIFF --output to retain transparency.")
    if args.guidance is not None and args.guidance < 1.0:
        parser.error(f"--guidance must be >= 1.0, got {args.guidance}")
    if args.steps < 1:
        parser.error(f"--steps must be >= 1, got {args.steps}")
    # plain-int dimensions are floored to /16 by the config; flag clearly wrong values
    dims_specified = CommandLineParser._option_was_provided("--width", "--height")
    if dims_specified and isinstance(args.width, int) and isinstance(args.height, int):
        if args.width < 32 or args.height < 32:
            parser.error(f"--width/--height must be >= 32, got {args.width}x{args.height}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    model_config = ConfigResolution.resolve_restricted(
        args.model,
        DEFAULT_MODEL,
        model_path=args.model_path,
        base_model=args.base_model,
    )

    qwen = QwenImage21Edit(
        quantize=args.quantize,
        model_path=args.model_path,
        model_config=model_config,
    )

    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=qwen,
        latent_creator=Qwen21LatentCreator,
    )

    try:
        image_paths = [str(p) for p in args.image_paths]
        width, height = DimensionResolver.resolve_output_dimensions(
            args.width,
            args.height,
            image_paths[-1],
            dims_specified=CommandLineParser._option_was_provided("--width", "--height"),
        )

        for seed in args.seed:
            image = qwen.generate_image(
                seed=seed,
                prompt=PromptUtil.read_prompt(args),
                negative_prompt=PromptUtil.read_negative_prompt(args),
                image_paths=image_paths,
                width=width,
                height=height,
                guidance=args.guidance if args.guidance is not None else 1.0,
                scheduler=args.scheduler,
                num_inference_steps=args.steps,
                mask_image=args.mask_image,
                auto_mask=args.auto_mask,
                use_step_cache=args.use_step_cache,
                step_cache_threshold=args.step_cache_threshold,
                strength=args.strength,
                enhance_prompt=args.enhance_prompt,
                verify=args.verify,
                verify_retries=args.verify_retries,
                rgba_output=args.rgba_output,
            )
            if getattr(image, "verification", None):
                print(f"verification: {image.verification}")
            image.save(path=Path(args.output.format(seed=seed)), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
