from argparse import Namespace

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser, lora_init_kwargs_from_args
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.ernie_image.latent_creator import ErnieLatentCreator
from mflux.models.ernie_image.variants.txt2img.ernie_image import ErnieImage
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import ModelConfigError, PromptFileReadError, StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.prompt_util import PromptUtil

IGNORED_OPTIONS = {
    "--negative-prompt": "turbo pins guidance to 1.0, so there is no unconditional pass for a negative prompt to steer.",
}

CONDITIONAL_OPTIONS = {
    "--guidance": {
        "condition": "must be 1.0",
        "reason": "turbo is guidance-distilled; any other value exits with an error.",
    },
}


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(
        description="Generate an image using ERNIE-Image-Turbo (distilled, 8 steps) based on a prompt."
    )
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_image_to_image_arguments(required=False)
    parser.add_pid_decode_arguments()
    parser.add_output_arguments()
    parser.set_defaults(model="ernie-image-turbo")
    return parser


class ErnieImageTurboCommand:
    latent_creator = ErnieLatentCreator

    @staticmethod
    def validate(args: Namespace) -> ModelConfig:
        # Weight-free. Guidance is judged before --model, so a request wrong on both reports
        # the guidance first.
        ErnieImageTurboCommand._guidance(args)
        # --model accepts only ernie-image-turbo aliases; base ernie-image has its own CLI
        # and anything else errors instead of being silently run as ERNIE-Image-Turbo.
        return ConfigResolution.resolve_restricted(args.model, "ernie-image-turbo", model_path=args.model_path)

    @staticmethod
    def load(args: Namespace) -> ErnieImage:
        return ErnieImage(
            model_config=ErnieImageTurboCommand.validate(args),
            quantize=args.quantize,
            model_path=args.model_path,
            **lora_init_kwargs_from_args(args),
        )

    @staticmethod
    def generate(model: ErnieImage, args: Namespace, seed: int, prompt: str) -> GeneratedImage:
        # Judged first, so a bad guidance is reported before the reference image is opened.
        guidance = ErnieImageTurboCommand._guidance(args)
        # ScaleFactor dims ("2x", and the "auto" default) need the reference image; resolved
        # here so a caller of generate() gets them without going through main().
        width, height = DimensionResolver.resolve(
            width=args.width, height=args.height, reference_image_path=args.image_path
        )
        return model.generate_image(
            seed=seed,
            prompt=prompt,
            width=width,
            height=height,
            guidance=guidance,
            image_path=args.image_path,
            num_inference_steps=args.steps,
            image_strength=args.image_strength,
            scheduler=args.scheduler,
            negative_prompt=args.negative_prompt,
            pid_decode=args.pid_decode,
            pid_degrade_sigma=args.pid_degrade_sigma,
        )

    @staticmethod
    def _guidance(args: Namespace) -> float:
        # Turbo is guidance-distilled: an omitted --guidance means 1.0 and anything else is
        # refused, here so validate() and generate() both enforce it.
        guidance = 1.0 if args.guidance is None else args.guidance
        if guidance != 1.0:
            raise ValueError("--guidance is only supported for base ERNIE-Image. Use --guidance 1.0 for turbo.")
        return guidance


def main():
    parser = build_parser()
    args = parser.parse_args()
    CommandLineParser.warn_ignored_options(IGNORED_OPTIONS)

    try:
        ErnieImageTurboCommand.validate(args)
    except ModelConfigError:
        # A --model outside this command's aliases is a traceback, not a usage error.
        raise
    except ValueError as e:
        parser.error(str(e))

    # 1. Load the model
    model = ErnieImageTurboCommand.load(args)

    # 2. Register callbacks
    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=ErnieImageTurboCommand.latent_creator,
    )

    try:
        for seed in args.seed:
            # 3. Generate an image for each seed value
            image = ErnieImageTurboCommand.generate(model, args, seed, PromptUtil.read_prompt(args))
            # 4. Save the image
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
