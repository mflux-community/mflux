import warnings
from argparse import Namespace

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser, positive_float
from mflux.models.common.compute_precision import ComputePrecision
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.dreamlite.cli.dreamlite_generate import (
    COMPUTE_PRECISION_HELP,
    DEFAULT_MLX_CACHE_LIMIT_GB,
    DEFAULT_MODEL,
    FAMILY_MODELS,
    GUIDANCE_OPTIONS,
    IGNORED_OPTIONS,
    DreamLiteCommand,
)
from mflux.models.dreamlite.latent_creator.dreamlite_latent_creator import DreamLiteLatentCreator
from mflux.models.dreamlite.variants.edit.dreamlite_edit import DreamLiteEdit
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.prompt_util import PromptUtil

DEFAULT_IMAGE_GUIDANCE = DreamLiteEdit.DEFAULT_IMAGE_GUIDANCE

# Options that act through guidance's extra passes, which do not run at guidance 1.0
EDIT_GUIDANCE_OPTIONS = {
    **GUIDANCE_OPTIONS,
    "--image-guidance": {
        "condition": "guidance above 1.0",
        "reason": "image guidance weighs a pass that only runs with guidance, so at guidance 1.0 it changes nothing.",
    },
}

CONDITIONAL_OPTIONS = {
    **EDIT_GUIDANCE_OPTIONS,
}


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(
        description=(
            "Edit an image with DreamLite. The prompt is the edit instruction. The edit runs on a 1024x1024 copy "
            "of the image, and the result is saved at the image's own size."
        )
    )
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False, default_model=DEFAULT_MODEL)
    parser.add_image_paths_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_argument("--image-guidance", type=positive_float, default=None, help=f"How strongly the result keeps to the source image, beside --guidance for the instruction (Default is {DEFAULT_IMAGE_GUIDANCE}). Any other value adds a third pass to each step. Not used by DreamLite-mobile.")  # fmt: off
    parser.add_compute_precision_arguments(help_text=COMPUTE_PRECISION_HELP)
    parser.add_output_arguments()
    parser.set_defaults(mlx_cache_limit_gb=DEFAULT_MLX_CACHE_LIMIT_GB)
    return parser


class DreamLiteEditCommand:
    latent_creator = DreamLiteLatentCreator

    @staticmethod
    def validate(args: Namespace) -> ModelConfig:
        if len(args.image_paths) != 1:
            raise ValueError(f"--image-paths takes one image for DreamLite, got {len(args.image_paths)}.")
        # A scale factor ("auto", "2x") is read against the source image later; a plain size is checked here
        for option, value in (("--width", args.width), ("--height", args.height)):
            if isinstance(value, int) and value <= 0:
                raise ValueError(f"{option} must be positive, got {value}.")
        return ConfigResolution.resolve_restricted(
            args.model,
            DEFAULT_MODEL,
            model_path=args.model_path,
            extra_keys=FAMILY_MODELS,
            base_model=args.base_model,
        )

    @staticmethod
    def load(args: Namespace) -> DreamLiteEdit:
        return DreamLiteEdit(
            model_config=DreamLiteEditCommand.validate(args),
            quantize=args.quantize,
            model_path=args.model_path,
            compute_precision=ComputePrecision.dtype_for(args.compute_precision),
        )

    @staticmethod
    def warn_unused_guidance_options(args: Namespace, model_config: ModelConfig) -> None:
        guidance_off = not model_config.supports_guidance or (args.guidance is not None and args.guidance <= 1.0)
        for option, entry in EDIT_GUIDANCE_OPTIONS.items():
            if guidance_off and CommandLineParser._option_was_provided(option):
                warnings.warn(f"{option} is ignored at guidance 1.0; {entry['reason']}", stacklevel=2)

    @staticmethod
    def generate(model: DreamLiteEdit, args: Namespace, seed: int, prompt: str) -> GeneratedImage:
        # "auto" (the default) is the source image's size; a scale factor applies to it
        image_path = args.image_paths[0]
        width, height = DimensionResolver.resolve(width=args.width, height=args.height, reference_image_path=image_path)  # fmt: off
        return model.generate_image(
            seed=seed,
            prompt=prompt,
            image_path=image_path,
            negative_prompt=PromptUtil.read_negative_prompt(args),
            width=width,
            height=height,
            guidance=args.guidance,
            image_guidance=args.image_guidance,
            num_inference_steps=args.steps,
        )


def main():
    # 0. Parse command line arguments
    parser = build_parser()
    args = parser.parse_args()

    # 1. Check the flags, then load the model
    try:
        model_config = DreamLiteEditCommand.validate(args)
    except ValueError as exc:
        parser.error(str(exc))
    CommandLineParser.warn_ignored_options(IGNORED_OPTIONS)
    DreamLiteCommand.warn_guidance_on_mobile(args, model_config)
    DreamLiteEditCommand.warn_unused_guidance_options(args, model_config)
    model = DreamLiteEditCommand.load(args)

    # 2. Register callbacks
    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=DreamLiteEditCommand.latent_creator,
    )

    try:
        for seed in args.seed:
            # 3. Generate an image for each seed value
            image = DreamLiteEditCommand.generate(model, args, seed, PromptUtil.read_prompt(args))
            # 4. Save the image
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
