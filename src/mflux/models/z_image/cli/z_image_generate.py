import sys
import warnings

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser, lora_init_kwargs_from_args
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.z_image.latent_creator import ZImageLatentCreator
from mflux.models.z_image.variants.z_image import ZImage
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil

# The model this CLI runs when --model is omitted. The parser needs it too, to key the
# --steps default off the right model instead of falling back to FLUX.1-dev's 25.
DEFAULT_MODEL = "z-image"

# This CLI also runs the distilled sibling (CONDITIONAL_OPTIONS is written for exactly
# that split); the ControlNet variant has its own command and needs a control image.
FAMILY_MODELS = ("z-image-turbo",)

# Single source of truth for CFG-dependent options: main() warns from these and the
# mflux-capabilities dump reads them. Both flags depend on what --model resolves to,
# so they are conditional, not statically ignored.
CONDITIONAL_OPTIONS = {
    "--guidance": {
        "condition": "the resolved model supports guidance",
        "reason": "guidance-distilled Z-Image variants force guidance to 0.0.",
    },
    "--negative-prompt": {
        "condition": "the resolved model supports guidance and guidance > 1.0",
        "reason": "CFG is disabled at guidance <= 1.0 (the default is 0.0) and on guidance-distilled variants.",
    },
}


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image using Z-Image.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False, default_model=DEFAULT_MODEL)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_image_to_image_arguments()
    parser.add_pid_decode_arguments()
    parser.add_output_arguments()
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if "--scheduler" not in sys.argv:
        args.scheduler = "flow_match_euler_discrete"

    model_config = ConfigResolution.resolve_restricted(
        args.model, DEFAULT_MODEL, model_path=args.model_path, extra_keys=FAMILY_MODELS
    )

    # Warn on the EFFECTIVE behavior, not the flag value: --model may resolve to a
    # guidance-distilled variant (guidance forced to 0.0 regardless of the flag), and on
    # CFG-capable models the negative prompt is only encoded at guidance > 1.0, where an
    # omitted --guidance defaults to 0.0.
    if not model_config.supports_guidance:
        CommandLineParser.warn_ignored_options(
            {
                "--guidance": CONDITIONAL_OPTIONS["--guidance"]["reason"],
                "--negative-prompt": CONDITIONAL_OPTIONS["--negative-prompt"]["reason"],
            }
        )
    elif CommandLineParser._option_was_provided("--negative-prompt") and (
        args.guidance is None or args.guidance <= 1.0
    ):
        warnings.warn(
            f"--negative-prompt has no effect: {CONDITIONAL_OPTIONS['--negative-prompt']['reason']}"
            " Pass --guidance above 1.0 to enable it.",
            stacklevel=1,
        )

    # 1. Load the model
    model = ZImage(
        model_config=model_config,
        quantize=args.quantize,
        model_path=args.model_path,
        **lora_init_kwargs_from_args(args),
    )

    # 2. Register callbacks
    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=ZImageLatentCreator,
    )

    try:
        # Resolve dimensions
        width, height = DimensionResolver.resolve(
            width=args.width,
            height=args.height,
            reference_image_path=args.image_path,
        )

        for seed in args.seed:
            # 3. Generate an image for each seed value
            image = model.generate_image(
                seed=seed,
                prompt=PromptUtil.read_prompt(args),
                width=width,
                height=height,
                guidance=args.guidance,
                image_path=args.image_path,
                num_inference_steps=args.steps,
                image_strength=args.image_strength,
                scheduler=args.scheduler,
                negative_prompt=args.negative_prompt,
                pid_decode=args.pid_decode,
                pid_degrade_sigma=args.pid_degrade_sigma,
            )
            # 4. Save the image
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
