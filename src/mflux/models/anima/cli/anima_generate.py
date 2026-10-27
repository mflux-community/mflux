import warnings
from argparse import Namespace

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser, lora_init_kwargs_from_args
from mflux.models.anima.latent_creator.anima_latent_creator import AnimaLatentCreator
from mflux.models.anima.model.anima_scheduler import AnimaSampler, AnimaScheduler
from mflux.models.anima.variants.txt2img.anima import Anima
from mflux.models.common.compute_precision import ComputePrecision
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.prompt_util import PromptUtil

# The model this CLI runs when --model is omitted: Turbo, the variant the model card recommends starting with.
# The parser also reads it for the --steps default.
DEFAULT_MODEL = "anima-turbo"

# The other variants in the same repo: Aesthetic (best quality) and Base (for LoRA training).
FAMILY_MODELS = ("anima-aesthetic", "anima-base")

CONDITIONAL_OPTIONS = {
    "--negative-prompt": {
        "condition": "guidance other than 1.0",
        "reason": "at guidance 1.0 (the Turbo default) there is no unconditional pass, so the negative prompt is never encoded.",  # fmt: off
    },
}


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image using Anima (Turbo, Aesthetic or Base).")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False, default_model=DEFAULT_MODEL)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_compute_precision_arguments()
    parser.add_output_arguments()
    # The command's sampler: er_sde (the model card's default) or euler.
    parser.set_defaults(scheduler=AnimaSampler.DEFAULT)
    return parser


class AnimaCommand:
    latent_creator = AnimaLatentCreator

    @staticmethod
    def validate(args: Namespace) -> ModelConfig:
        # Weight-free: resolves --model against the in-memory registry only, and checks the sampler and the size
        # before the weights load.
        model_config = ConfigResolution.resolve_restricted(
            args.model,
            DEFAULT_MODEL,
            model_path=args.model_path,
            extra_keys=FAMILY_MODELS,
            base_model=args.base_model,
        )
        AnimaSampler.resolve(args.scheduler)
        if args.steps is not None:
            AnimaScheduler.check_steps(args.steps)
        width, height = DimensionResolver.resolve(width=args.width, height=args.height, reference_image_path=None)
        AnimaLatentCreator.check_size(height=height, width=width)
        return model_config

    @staticmethod
    def load(args: Namespace) -> Anima:
        return Anima(
            model_config=AnimaCommand.validate(args),
            quantize=args.quantize,
            model_path=args.model_path,
            compute_precision=ComputePrecision.dtype_for(args.compute_precision),
            **lora_init_kwargs_from_args(args),
        )

    @staticmethod
    def generate(model: Anima, args: Namespace, seed: int, prompt: str) -> GeneratedImage:
        # Steps and guidance left unset take the variant's defaults from the model.
        width, height = DimensionResolver.resolve(width=args.width, height=args.height, reference_image_path=None)
        return model.generate_image(
            seed=seed,
            prompt=prompt,
            num_inference_steps=args.steps,
            height=height,
            width=width,
            guidance=args.guidance,
            negative_prompt=PromptUtil.read_negative_prompt(args),
            scheduler=args.scheduler,
        )


def main():
    # 0. Parse command line arguments
    parser = build_parser()
    args = parser.parse_args()

    # 1. Load the model
    model = AnimaCommand.load(args)

    # 2. Register callbacks
    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=AnimaCommand.latent_creator,
    )

    try:
        if model.guidance_or_default(args.guidance) == 1.0 and CommandLineParser._option_was_provided("--negative-prompt"):  # fmt: off
            warnings.warn(
                "--negative-prompt is ignored at guidance 1.0; " + CONDITIONAL_OPTIONS["--negative-prompt"]["reason"],
                stacklevel=2,
            )
        for seed in args.seed:
            # 3. Generate an image for each seed value
            image = AnimaCommand.generate(model, args, seed, PromptUtil.read_prompt(args))
            # 4. Save the image
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
