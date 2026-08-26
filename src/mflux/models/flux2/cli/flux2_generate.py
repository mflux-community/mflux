from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser, lora_init_kwargs_from_args
from mflux.models.common.config.model_config import AVAILABLE_MODELS
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
from mflux.models.flux2.variants import Flux2Klein
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil

# The model this CLI runs when --model is omitted. The parser needs it too, to key the
# --steps default off the right model instead of falling back to FLUX.1-dev's 25.
DEFAULT_MODEL = "flux2-klein-4b"

# The other klein entries this one command equally serves, read off the registry so a
# new flux2- entry is accepted without touching this file.
FAMILY_MODELS = tuple(key for key in AVAILABLE_MODELS if key.startswith("flux2-") and key != DEFAULT_MODEL)

REJECTED_OPTIONS = {
    "--negative-prompt": "FLUX.2 has no negative branch; the CLI exits with an error when this is set.",
}

CONDITIONAL_OPTIONS = {
    "--guidance": {
        "condition": "base (non-distilled) FLUX.2 checkpoints",
        "reason": "distilled checkpoints require guidance 1.0; any other value exits with an error.",
    },
}


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image using Flux2 Klein.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False, default_model=DEFAULT_MODEL)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_image_to_image_arguments(required=False)
    parser.add_pid_decode_arguments()
    parser.add_output_arguments()
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Keyed on the option, not on the value: a --config-from-metadata sidecar written by a
    # CFG model restores a negative prompt the user never typed, and rejecting the rerun for
    # it would blame them for an argument that is not on the command line.
    if CommandLineParser._option_was_provided("--negative-prompt"):
        parser.error("--negative-prompt is not supported for FLUX.2. Focus on describing what you want.")

    # One command serves the whole klein family; anything outside it (a FLUX.1 name, a
    # qwen alias) errors instead of silently loading a foreign config into this pipeline.
    model_config = ConfigResolution.resolve_restricted(
        args.model,
        DEFAULT_MODEL,
        model_path=args.model_path,
        extra_keys=FAMILY_MODELS,
    )

    if args.guidance is None:
        args.guidance = 1.0
    # Judged only for builtin names: a custom checkpoint (model_path set) keeps the
    # default entry's config, whose name says nothing about the weights actually loaded,
    # and a klein-base fine-tune must not have its guidance rejected for it.
    is_distilled = args.model_path is None and "base" not in model_config.model_name.lower()
    if args.guidance != 1.0 and is_distilled:
        parser.error("--guidance is only supported for FLUX.2 base models. Use --guidance 1.0.")

    model = Flux2Klein(
        model_config=model_config,
        quantize=args.quantize,
        model_path=args.model_path,
        **lora_init_kwargs_from_args(args),
    )

    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=Flux2LatentCreator,
    )

    try:
        width, height = DimensionResolver.resolve(
            width=args.width,
            height=args.height,
            reference_image_path=args.image_path,
        )

        for seed in args.seed:
            image = model.generate_image(
                seed=seed,
                prompt=PromptUtil.read_prompt(args),
                width=width,
                height=height,
                guidance=args.guidance,
                image_path=args.image_path,
                num_inference_steps=args.steps,
                image_strength=args.image_strength,
                scheduler="flow_match_euler_discrete",
                pid_decode=args.pid_decode,
                pid_degrade_sigma=args.pid_degrade_sigma,
            )
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
