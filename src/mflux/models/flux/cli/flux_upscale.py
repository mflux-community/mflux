from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser, lora_init_kwargs_from_args
from mflux.models.common.config import ModelConfig
from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
from mflux.models.flux.variants.controlnet.flux_controlnet import Flux1Controlnet
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil

# The upscaler's effective guidance before --guidance was wired up: Flux1Controlnet's own
# generate_image default, not ui_defaults.GUIDANCE_SCALE (3.5). Kept as-is so an existing
# invocation that never passed --guidance keeps producing the image it produced before.
DEFAULT_GUIDANCE = 4.0

# Single source of truth for options this CLI accepts but cannot honour: the runtime
# warning and the mflux-capabilities dump both read it.
IGNORED_OPTIONS = {
    "--negative-prompt": "FLUX.1 uses distilled guidance and has no negative branch.",
    "--base-model": "the config is always dev-controlnet-upscaler; nothing reads this.",
}
CONDITIONAL_OPTIONS = {
    "--model": {
        "condition": "a local path or HuggingFace repo id",
        "reason": "weights load from the path, but the config is always dev-controlnet-upscaler, so a built-in model name has no effect.",  # fmt: off
    },
}


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Upscale an image.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=False, supports_dimension_scale_factor=True)
    parser.add_controlnet_arguments(require_image=True)
    parser.add_output_arguments()
    return parser


def main():
    # 0. Parse command line arguments
    parser = build_parser()
    args = parser.parse_args()
    CommandLineParser.warn_ignored_options(IGNORED_OPTIONS)

    if args.guidance is None:
        args.guidance = DEFAULT_GUIDANCE

    # 1. Load the model
    flux = Flux1Controlnet(
        model_config=ModelConfig.dev_controlnet_upscaler(),
        quantize=args.quantize,
        model_path=args.model_path,
        **lora_init_kwargs_from_args(args),
    )

    # 2. Register the optional callbacks
    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=flux,
        latent_creator=FluxLatentCreator,
    )

    try:
        # Resolve dimensions using the controlnet image as reference
        width, height = DimensionResolver.resolve(
            height=args.height,
            width=args.width,
            reference_image_path=args.controlnet_image_path,
        )

        for seed in args.seed:
            # 3. Generate an upscaled image for each seed value
            image = flux.generate_image(
                seed=seed,
                prompt=PromptUtil.read_prompt(args),
                width=width,
                height=height,
                num_inference_steps=args.steps,
                guidance=args.guidance,
                scheduler=args.scheduler,
                controlnet_strength=args.controlnet_strength,
                controlnet_image_path=args.controlnet_image_path,
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
