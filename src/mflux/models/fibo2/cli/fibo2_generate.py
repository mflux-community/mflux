import copy
import warnings
from argparse import Namespace

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser, lora_init_kwargs_from_args
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.fibo2.latent_creator.fibo2_latent_creator import Fibo2LatentCreator
from mflux.models.fibo2.variants.txt2img.fibo2 import Fibo2
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.prompt_util import PromptUtil

# The model this CLI runs. The parser needs it to key the --steps default off the right model.
DEFAULT_MODEL = "fibo2"

# Options the shared parser adds that FIBO 2 cannot use: kept accepted so scripts do not break, but reported
IGNORED_OPTIONS = {
    "--scheduler": "FIBO 2 always runs the flow-match Euler schedule it was trained with.",
    "--lora-style": "Named LoRA styles are only supported by the Flux in-context CLI; use --lora.",
}

CONDITIONAL_OPTIONS = {
    "--negative-prompt": {
        "condition": "guidance above 1.0",
        "reason": "the negative prompt is guidance's second pass, so at guidance 1.0 it is never encoded.",
    },
}


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image using the FIBO 2 model.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False, default_model=DEFAULT_MODEL)
    # Editing has its own command, mflux-generate-fibo2-edit.
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_argument("--no-text-images", action="store_true", help="Do not show the text encoder images of the caption's text_render texts. FIBO 2 renders text better with them.")  # fmt: off
    parser.add_lora_arguments()
    parser.add_output_arguments()
    return parser


class Fibo2Command:
    latent_creator = Fibo2LatentCreator

    @staticmethod
    def validate(args: Namespace) -> ModelConfig:
        return ConfigResolution.resolve_restricted(
            args.model,
            DEFAULT_MODEL,
            model_path=args.model_path,
            base_model=args.base_model,
        )

    @staticmethod
    def load(args: Namespace) -> Fibo2:
        return Fibo2(
            model_config=Fibo2Command.validate(args),
            quantize=args.quantize,
            model_path=args.model_path,
            **lora_init_kwargs_from_args(args),
        )

    @staticmethod
    def warn_unused_negative_prompt(args: Namespace, model_config: ModelConfig) -> None:
        # Guidance is off when set to 1 or below, or left to Edit Turbo's default of 1
        guidance_off = args.guidance <= 1.0 if args.guidance is not None else not model_config.supports_guidance
        if guidance_off and CommandLineParser._option_was_provided("--negative-prompt"):
            reason = CONDITIONAL_OPTIONS["--negative-prompt"]["reason"]
            warnings.warn(f"--negative-prompt is ignored at guidance 1.0; {reason}", stacklevel=2)

    @staticmethod
    def generate(model: Fibo2, args: Namespace, seed: int, prompt: str) -> GeneratedImage:
        width, height = DimensionResolver.resolve(width=args.width, height=args.height)
        return model.generate_image(
            seed=seed,
            prompt=prompt,
            negative_prompt=PromptUtil.read_negative_prompt(args),
            width=width,
            height=height,
            guidance=args.guidance,
            num_inference_steps=args.steps,
            render_text_images=not args.no_text_images,
        )


def main():
    # 0. Parse command line arguments
    parser = build_parser()
    args = parser.parse_args()

    # 1. Load the model
    CommandLineParser.warn_ignored_options(IGNORED_OPTIONS)
    Fibo2Command.warn_unused_negative_prompt(args, Fibo2Command.validate(args))
    model = Fibo2Command.load(args)

    # 2. Register callbacks
    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=Fibo2Command.latent_creator,
    )

    try:
        run_args = copy.copy(args)
        run_args.width, run_args.height = DimensionResolver.resolve(width=args.width, height=args.height)
        for seed in args.seed:
            # 3. Generate an image for each seed value
            image = Fibo2Command.generate(model, run_args, seed, PromptUtil.read_prompt(args))
            # 4. Save the image
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
