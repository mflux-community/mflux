from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser, lora_init_kwargs_from_args
from mflux.models.common.config import ModelConfig
from mflux.models.z_image.latent_creator import ZImageLatentCreator
from mflux.models.z_image.variants.z_image import ZImage
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil

# The model this CLI runs. The parser needs it to key the --steps default off the right
# model instead of falling back to FLUX.1-dev's 25.
DEFAULT_MODEL = "z-image-turbo"

# Single source of truth for options this CLI accepts but cannot honour: the runtime
# warning and the mflux-capabilities dump both read it.
IGNORED_OPTIONS = {
    "--guidance": "Z-Image Turbo is guidance-distilled; guidance is forced to 0.0.",
    "--negative-prompt": "CFG is disabled on Z-Image Turbo, so the negative prompt is never encoded.",
}


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image using Z-Image Turbo based on a prompt.")
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
    CommandLineParser.warn_ignored_options(IGNORED_OPTIONS)

    # 1. Load the model
    model = ZImage(
        model_config=ModelConfig.z_image_turbo(),
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
        # Resolve dimensions (supports ScaleFactor like "2x" when --image-path is provided)
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
