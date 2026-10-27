import warnings
from argparse import Namespace
from pathlib import Path

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser, lora_init_kwargs_from_args
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.fibo2.cli.fibo2_generate import Fibo2Command
from mflux.models.fibo2.latent_creator.fibo2_latent_creator import Fibo2LatentCreator
from mflux.models.fibo2.model.fibo2_text_encoder.prompt_encoder import Fibo2PromptEncoder
from mflux.models.fibo2.variants import Fibo2Edit
from mflux.models.fibo2.variants.edit.fibo2_edit_images import Fibo2EditImages
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.prompt_util import PromptUtil
from mflux.utils.scale_factor import ScaleFactor

# The model this CLI runs. The parser needs it to key the --steps default off the right model.
DEFAULT_MODEL = "fibo2"

# Options the shared parser adds that FIBO 2 cannot use: kept accepted so scripts do not break, but reported
IGNORED_OPTIONS = {
    "--scheduler": "FIBO 2 always runs the flow-match Euler schedule it was trained with.",
    "--lora-style": "Named LoRA styles are only supported by the Flux in-context CLI; use --lora.",
}

TURBO_GUIDANCE_WARNING = (
    "Edit Turbo is distilled to run without guidance; a value above 1 still runs the negative pass."
)


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(
        description=(
            "Edit 1-5 images with FIBO 2. The prompt is the edit instruction, or a JSON object with an "
            '"edit_instruction" key and a caption of the result. Refer to several images as <image_1>, <image_2>...'
        )
    )
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False, default_model=DEFAULT_MODEL)
    parser.add_image_paths_arguments()
    parser.add_argument("--mask-path", type=Path, default=None, help="Mask of the region to regenerate, white on black, at the image's size. Single image only.")  # fmt: off
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_argument("--no-text-images", action="store_true", help="Do not show the text encoder images of the caption's text_render texts. FIBO 2 renders text better with them.")  # fmt: off
    parser.add_lora_arguments()
    parser.add_output_arguments()
    return parser


class Fibo2EditCommand:
    latent_creator = Fibo2LatentCreator

    @staticmethod
    def validate(args: Namespace) -> ModelConfig:
        if not 1 <= len(args.image_paths) <= Fibo2EditImages.MAX_IMAGES:
            raise ValueError(f"--image-paths takes 1 to {Fibo2EditImages.MAX_IMAGES} images, got {len(args.image_paths)}.")  # fmt: off
        if args.mask_path is not None and len(args.image_paths) != 1:
            raise ValueError("--mask-path can only be used with a single image.")
        # stdin ("-") is read once, at generation, and checked there
        if args.prompt and args.prompt != "-":
            Fibo2PromptEncoder.check_edit_prompt(args.prompt)
        elif args.prompt_file is not None and Path(args.prompt_file).is_file():
            Fibo2PromptEncoder.check_edit_prompt(Path(args.prompt_file).read_text().strip())
        return ConfigResolution.resolve_restricted(
            args.model,
            DEFAULT_MODEL,
            model_path=args.model_path,
            extra_keys=("fibo2-edit-turbo",),
            base_model=args.base_model,
        )

    @staticmethod
    def load(args: Namespace) -> Fibo2Edit:
        return Fibo2Edit(
            model_config=Fibo2EditCommand.validate(args),
            quantize=args.quantize,
            model_path=args.model_path,
            **lora_init_kwargs_from_args(args),
        )

    @staticmethod
    def generate(model: Fibo2Edit, args: Namespace, seed: int, prompt: str) -> GeneratedImage:
        width, height = Fibo2EditCommand._explicit_size(args.width, args.height, args.image_paths[0])
        return model.generate_image(
            seed=seed,
            prompt=prompt,
            image_paths=args.image_paths,
            mask_path=args.mask_path,
            negative_prompt=PromptUtil.read_negative_prompt(args),
            width=width,
            height=height,
            guidance=args.guidance,
            num_inference_steps=args.steps,
            render_text_images=not args.no_text_images,
        )

    @staticmethod
    def warn_guidance_on_turbo(args: Namespace, model_config: ModelConfig) -> None:
        if not model_config.supports_guidance and args.guidance is not None and args.guidance > 1.0:
            warnings.warn(f"--guidance {args.guidance}: {TURBO_GUIDANCE_WARNING}", stacklevel=2)

    @staticmethod
    def _explicit_size(
        width: int | ScaleFactor, height: int | ScaleFactor, image_path: Path
    ) -> tuple[int | None, int | None]:
        # "auto" (1x) leaves the size to the model; other scales apply to the first image
        resolved_width, resolved_height = DimensionResolver.resolve(width=width, height=height, reference_image_path=image_path)  # fmt: off
        width_is_auto = isinstance(width, ScaleFactor) and width.value == 1
        height_is_auto = isinstance(height, ScaleFactor) and height.value == 1
        return (None if width_is_auto else resolved_width), (None if height_is_auto else resolved_height)


def main():
    # 0. Parse command line arguments
    parser = build_parser()
    args = parser.parse_args()

    # 1. Check the flags, then load the model
    try:
        model_config = Fibo2EditCommand.validate(args)
    except ValueError as exc:
        parser.error(str(exc))
    CommandLineParser.warn_ignored_options(IGNORED_OPTIONS)
    Fibo2EditCommand.warn_guidance_on_turbo(args, model_config)
    Fibo2Command.warn_unused_negative_prompt(args, model_config)
    model = Fibo2EditCommand.load(args)

    # 2. Register callbacks
    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=Fibo2EditCommand.latent_creator,
    )

    try:
        for seed in args.seed:
            # 3. Generate an image for each seed value
            image = Fibo2EditCommand.generate(model, args, seed, PromptUtil.read_prompt(args))
            # 4. Save the image
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
