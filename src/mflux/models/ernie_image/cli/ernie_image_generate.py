import sys
from argparse import Namespace

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser, lora_init_kwargs_from_args
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.ernie_image.latent_creator import ErnieLatentCreator
from mflux.models.ernie_image.variants.txt2img.ernie_image import ErnieImage
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.prompt_util import PromptUtil


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image using ERNIE-Image (50 steps, CFG) based on a prompt.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_image_to_image_arguments(required=False)
    parser.add_pid_decode_arguments()
    parser.add_output_arguments()
    parser.set_defaults(model="ernie-image", guidance=4.0)
    return parser


class ErnieImageCommand:
    latent_creator = ErnieLatentCreator

    @staticmethod
    def validate(args: Namespace) -> ModelConfig:
        # Weight-free: resolves --model against the in-memory registry only. --model accepts
        # only ernie-image aliases; ernie-image-turbo has its own CLI and anything else errors
        # instead of being silently run as base ERNIE-Image.
        return ConfigResolution.resolve_restricted(args.model, "ernie-image", model_path=args.model_path)

    @staticmethod
    def load(args: Namespace) -> ErnieImage:
        return ErnieImage(
            model_config=ErnieImageCommand.validate(args),
            quantize=args.quantize,
            model_path=args.model_path,
            **lora_init_kwargs_from_args(args),
        )

    @staticmethod
    def generate(model: ErnieImage, args: Namespace, seed: int, prompt: str) -> GeneratedImage:
        width, height = DimensionResolver.resolve(
            width=args.width, height=args.height, reference_image_path=args.image_path
        )
        return model.generate_image(
            seed=seed,
            prompt=prompt,
            width=width,
            height=height,
            guidance=ErnieImageCommand._guidance(args),
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
        # The parser defaults --guidance to 4.0; a namespace built or edited by a caller may
        # still hold None.
        return 4.0 if args.guidance is None else args.guidance


def main():
    parser = build_parser()
    args = parser.parse_args()

    # The command's scheduler default; it reads sys.argv, so it stays here, not in generate().
    if "--scheduler" not in sys.argv:
        args.scheduler = "linear"

    # 1. Load the model
    model = ErnieImageCommand.load(args)

    # 2. Register callbacks
    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=ErnieImageCommand.latent_creator,
    )

    try:
        for seed in args.seed:
            # 3. Generate an image for each seed value
            image = ErnieImageCommand.generate(model, args, seed, PromptUtil.read_prompt(args))
            # 4. Save the image
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
