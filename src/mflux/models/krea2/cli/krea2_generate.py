import warnings
from argparse import Namespace

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser, lora_init_kwargs_from_args
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.krea2.latent_creator import Krea2LatentCreator
from mflux.models.krea2.variants.txt2img.krea2 import Krea2
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.prompt_util import PromptUtil

# Krea-2 turbo defaults (reference: 8 steps, CFG 1.0, er_sde; sigmas use the official
# dynamic exponential shift, base/max 0.5/1.15 over image seq len 256..6400). The 8-step
# count lives in ui_defaults.MODEL_INFERENCE_STEPS under this model's registry key; the
# parser applies it, so main() sees an already-resolved args.steps.
DEFAULT_MODEL = "krea-2"
DEFAULT_GUIDANCE = 1.0


CONDITIONAL_OPTIONS = {
    "--negative-prompt": {
        "condition": "guidance other than 1.0",
        "reason": "the encoder builds the unconditional branch only when guidance != 1.0, so at the "
        "distilled default of 1.0 the negative prompt is never encoded.",
    },
}


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image using Krea-2 based on a prompt.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False, default_model=DEFAULT_MODEL)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_image_to_image_arguments(required=False)
    parser.add_pid_decode_arguments()
    parser.add_output_arguments()
    return parser


class Krea2Command:
    latent_creator = Krea2LatentCreator

    @staticmethod
    def validate(args: Namespace) -> ModelConfig:
        # Weight-free: resolves --model against the in-memory registry only. --model accepts
        # only krea-2 aliases, so a foreign name is never silently run as Krea-2-Turbo.
        return ConfigResolution.resolve_restricted(args.model, DEFAULT_MODEL, model_path=args.model_path)

    @staticmethod
    def load(args: Namespace) -> Krea2:
        return Krea2(
            model_config=Krea2Command.validate(args),
            quantize=args.quantize,
            model_path=args.model_path,
            **lora_init_kwargs_from_args(args),
        )

    @staticmethod
    def generate(model: Krea2, args: Namespace, seed: int, prompt: str) -> GeneratedImage:
        # ScaleFactor dims ("2x", and the "auto" default) need the reference image; resolved
        # here so a caller of generate() gets them without going through main().
        width, height = DimensionResolver.resolve(
            width=args.width,
            height=args.height,
            reference_image_path=args.image_path,
        )
        return model.generate_image(
            seed=seed,
            prompt=prompt,
            num_inference_steps=args.steps,
            height=height,
            width=width,
            guidance=Krea2Command._guidance(args),
            scheduler=args.scheduler,
            negative_prompt=args.negative_prompt,
            image_path=args.image_path,
            image_strength=args.image_strength,
            pid_decode=args.pid_decode,
            pid_degrade_sigma=args.pid_degrade_sigma,
        )

    @staticmethod
    def _guidance(args: Namespace) -> float:
        # Read by generate() and by main()'s warning; never written back into args.
        return args.guidance if args.guidance is not None else DEFAULT_GUIDANCE


def main():
    # 0. Parse command line arguments
    parser = build_parser()
    args = parser.parse_args()

    # 1. Load the model
    model = Krea2Command.load(args)

    # 2. Register callbacks (stepwise image output, memory stats, battery saver)
    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=Krea2Command.latent_creator,
    )

    try:
        # The declared condition, checked once the default has resolved: the encoder
        # only builds the unconditional branch when guidance != 1.0. It reads sys.argv,
        # so it stays here, not in generate().
        if Krea2Command._guidance(args) == 1.0 and CommandLineParser._option_was_provided("--negative-prompt"):
            warnings.warn(
                "--negative-prompt is ignored at guidance 1.0; " + CONDITIONAL_OPTIONS["--negative-prompt"]["reason"],
                stacklevel=2,
            )
        # Sizes are resolved once per run, before the first prompt is read. generate()
        # resolves again for direct callers, which changes nothing on plain numbers.
        args.width, args.height = DimensionResolver.resolve(
            width=args.width,
            height=args.height,
            reference_image_path=args.image_path,
        )
        for seed in args.seed:
            # 3. Generate an image for each seed value
            image = Krea2Command.generate(model, args, seed, PromptUtil.read_prompt(args))
            # 4. Save the image
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
