import argparse
import math
from pathlib import Path

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser, lora_init_kwargs_from_args
from mflux.models.common.compute_precision import ComputePrecision
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.qwen21.latent_creator.qwen_image21_latent_creator import QwenImage21LatentCreator
from mflux.models.qwen21.model.qwen21_scheduler import Qwen21TurboScheduler, ViggleTurboScheduler
from mflux.models.qwen21.variants.edit.qwen_image_21_edit import EDIT_SCHEDULERS, MAX_REFERENCES, QwenImage21Edit
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import ModelConfigError, PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil
from mflux.utils.scale_factor import ScaleFactor

# Same architecture as qwen-image-2.1. The Turbo checkpoint brings its own schedule.
FAMILY_MODELS = ("qwen-image-2.1-turbo",)
IGNORED_OPTIONS = {"--lora-style": "Named LoRA styles are only supported by the Flux in-context CLI; use --lora."}
CONDITIONAL_OPTIONS = {
    "--scheduler": {
        "condition": "linear Euler or viggle_turbo scheduler only",
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
    parser.add_compute_precision_arguments()
    parser.add_lora_arguments()
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
    parser.add_argument(
        "--mask-image",
        type=str,
        default=None,
        help="Inpaint mask (white = repaint, black = preserve) over the first reference image.",
    )
    parser.add_argument(
        "--auto-mask",
        type=str,
        default=None,
        help="Describe an object to repaint ('the red shirt'); the built-in Qwen3-VL locates it. "
        "Ignored with --mask-image.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="Edit strength in (0, 1]: 1.0 denoises from pure noise, lower values start from the "
        "first reference partway down the schedule for subtler edits (default: 1.0).",
    )
    parser.add_argument(
        "--use-step-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip unchanged transformer blocks on similar steps (faster, slightly different output; "
        "needs --use-kv-cache; runs under 10 steps skip nothing).",
    )
    parser.add_argument(
        "--step-cache-threshold",
        type=float,
        default=0.12,
        help="Step-cache aggressiveness: higher skips more (default: 0.12).",
    )
    parser.add_argument(
        "--enhance-prompt",
        action="store_true",
        help="Rewrite the instruction into a detailed prompt with the built-in Qwen3-VL first.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After generating, check the edit with the built-in Qwen3-VL.",
    )
    parser.add_argument(
        "--verify-retries",
        type=int,
        default=0,
        help="With --verify, regenerate with the next seed on a failed check, at most this many times.",
    )
    return parser


def validate_edit_args(parser: CommandLineParser, args, paths: list) -> None:
    # Cheap checks that must fail before the model load.
    edits = {
        "--mask-image": args.mask_image is not None,
        "--auto-mask": args.auto_mask is not None,
        "--strength": args.strength != 1.0,
        "--enhance-prompt": args.enhance_prompt,
        "--verify": args.verify,
    }
    used = [flag for flag, on in edits.items() if on]
    if used and not paths:
        parser.error(f"{', '.join(used)} need at least one --image-paths reference image.")
    if args.mask_image is not None and not Path(args.mask_image).exists():
        parser.error(f"--mask-image not found: {args.mask_image}")
    if args.auto_mask is not None and not args.auto_mask.strip():
        parser.error("--auto-mask must name an object to locate.")
    if not 0 < args.strength <= 1:
        parser.error(f"--strength must be in (0, 1], got {args.strength}")
    if args.verify_retries < 0:
        parser.error(f"--verify-retries must be >= 0, got {args.verify_retries}")
    if args.verify_retries and not args.verify:
        parser.error("--verify-retries needs --verify")
    if args.use_step_cache and not args.use_kv_cache:
        parser.error("--use-step-cache needs --use-kv-cache (the default)")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    CommandLineParser.warn_ignored_options(IGNORED_OPTIONS)
    if args.guidance is None or args.guidance == 1:
        CommandLineParser.warn_ignored_options(
            {"--negative-prompt": CONDITIONAL_OPTIONS["--negative-prompt"]["reason"]}
        )
    if Path(args.output).suffix.lower() not in (".png", ".webp", ".tif", ".tiff"):
        parser.error("Qwen-Image-2.1 outputs RGBA; use PNG, WebP or TIFF to retain transparency.")
    if args.scheduler not in EDIT_SCHEDULERS:
        parser.error("Qwen-Image-2.1 supports the default linear Euler scheduler or viggle_turbo only.")
    ViggleTurboScheduler.check_args(parser, args)
    paths = args.image_paths or []
    if len(paths) > MAX_REFERENCES:
        parser.error(f"Qwen-Image-2.1 supports at most {MAX_REFERENCES} reference images.")
    validate_edit_args(parser, args, paths)
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
            args.model,
            "qwen-image-2.1",
            model_path=args.model_path,
            extra_keys=FAMILY_MODELS,
            base_model=args.base_model,
        )
    except (ModelConfigError, ValueError) as exc:
        parser.error(str(exc))
    Qwen21TurboScheduler.check_args(parser, args, model_config)
    model = QwenImage21Edit(
        quantize=args.quantize,
        model_path=args.model_path,
        model_config=model_config,
        compute_precision=ComputePrecision.dtype_for(args.compute_precision),
        **lora_init_kwargs_from_args(args),
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
                mask_image=args.mask_image,
                auto_mask=args.auto_mask,
                strength=args.strength,
                use_step_cache=args.use_step_cache,
                step_cache_threshold=args.step_cache_threshold,
                enhance_prompt=args.enhance_prompt,
                verify=args.verify,
                verify_retries=args.verify_retries,
                scheduler=args.scheduler,
            )
            if image.verification is not None:
                print(f"verification: {image.verification}")
            image.save(Path(args.output.format(seed=seed)), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
