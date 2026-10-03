from argparse import Namespace

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser, lora_init_kwargs_from_args
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.z_image.latent_creator import ZImageLatentCreator
from mflux.models.z_image.variants.z_image import ZImage
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
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
    parser.add_float32_arguments()
    parser.add_output_arguments()
    return parser


class ZImageTurboCommand:
    latent_creator = ZImageLatentCreator

    @staticmethod
    def validate(args: Namespace) -> ModelConfig:
        # Weight-free: resolves --model against the in-memory registry only. --model accepts
        # only z-image-turbo aliases; the ControlNet entry shares this repo id but is a
        # different model, so it is rejected too.
        return ConfigResolution.resolve_restricted(
            args.model, DEFAULT_MODEL, model_path=args.model_path, base_model=args.base_model
        )

    @staticmethod
    def load(args: Namespace) -> ZImage:
        return ZImage(
            model_config=ZImageTurboCommand.validate(args),
            quantize=args.quantize,
            model_path=args.model_path,
            float32=args.float32,
            **lora_init_kwargs_from_args(args),
        )

    @staticmethod
    def generate(model: ZImage, args: Namespace, seed: int, prompt: str) -> GeneratedImage:
        # ScaleFactor dims ("2x") need the reference image; resolved here so a caller of
        # generate() gets them without going through main().
        width, height = DimensionResolver.resolve(
            width=args.width, height=args.height, reference_image_path=args.image_path
        )
        return model.generate_image(
            seed=seed,
            prompt=prompt,
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


def main():
    parser = build_parser()
    args = parser.parse_args()
    CommandLineParser.warn_ignored_options(IGNORED_OPTIONS)

    # 1. Load the model
    model = ZImageTurboCommand.load(args)

    # 2. Register callbacks
    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=ZImageTurboCommand.latent_creator,
    )

    try:
        for seed in args.seed:
            # 3. Generate an image for each seed value
            image = ZImageTurboCommand.generate(model, args, seed, PromptUtil.read_prompt(args))
            # 4. Save the image
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
