import warnings
from argparse import Namespace

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.compute_precision import ComputePrecision
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.dreamlite.latent_creator.dreamlite_latent_creator import DreamLiteLatentCreator
from mflux.models.dreamlite.variants.txt2img.dreamlite import DreamLite
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.prompt_util import PromptUtil

# The model this CLI runs when --model is omitted. The parser also reads it for the --steps default.
DEFAULT_MODEL = "dreamlite"
# Mobile is the same network, step-distilled to 4 steps without guidance
FAMILY_MODELS = ("dreamlite-mobile",)

# Without a limit, MLX keeps every buffer a UNet pass frees: the process then holds about 9 GB where 4.4 GB is
# enough, and a step is no faster. --mlx-cache-limit-gb sets another limit.
DEFAULT_MLX_CACHE_LIMIT_GB = 1.0

# Options the shared parser adds that DreamLite cannot use: kept accepted so scripts do not break, but reported
IGNORED_OPTIONS = {
    "--scheduler": "DreamLite always runs the flow-match Euler schedule it was trained with.",
    "--vae-tiling": "DreamLite decodes with a tiny autoencoder (TAESDXL); there is nothing to gain from tiles.",
    "--vae-tile-size": "DreamLite decodes with a tiny autoencoder (TAESDXL); there is nothing to gain from tiles.",
}

MOBILE_GUIDANCE_WARNING = "DreamLite-mobile is distilled to run without guidance; the value is ignored."

COMPUTE_PRECISION_HELP = "Run the UNet in this precision; the text encoder and the VAE keep their own. float16 makes a step about 20%% faster on an M4. The image changes slightly for the same seed. Default: off, the UNet runs in float32."  # fmt: skip

# Options that act through guidance's second pass, which does not run at guidance 1.0
GUIDANCE_OPTIONS = {
    "--negative-prompt": {
        "condition": "guidance above 1.0",
        "reason": "the negative prompt is guidance's second pass, so at guidance 1.0 it is never encoded.",
    },
}

CONDITIONAL_OPTIONS = {
    **GUIDANCE_OPTIONS,
}


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(
        description="Generate an image using DreamLite (28 steps with guidance 3.5; DreamLite-mobile runs 4 steps without)."
    )
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False, default_model=DEFAULT_MODEL)
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_compute_precision_arguments(help_text=COMPUTE_PRECISION_HELP)
    parser.add_output_arguments()
    parser.set_defaults(mlx_cache_limit_gb=DEFAULT_MLX_CACHE_LIMIT_GB)
    return parser


class DreamLiteCommand:
    latent_creator = DreamLiteLatentCreator

    @staticmethod
    def validate(args: Namespace) -> ModelConfig:
        # Weight-free: a bad size fails before the checkpoint loads
        model_config = ConfigResolution.resolve_restricted(
            args.model,
            DEFAULT_MODEL,
            model_path=args.model_path,
            extra_keys=FAMILY_MODELS,
            base_model=args.base_model,
        )
        width, height = DimensionResolver.resolve(width=args.width, height=args.height)
        DreamLiteLatentCreator.check_size(height=height, width=width)
        return model_config

    @staticmethod
    def load(args: Namespace) -> DreamLite:
        return DreamLite(
            model_config=DreamLiteCommand.validate(args),
            quantize=args.quantize,
            model_path=args.model_path,
            compute_precision=ComputePrecision.dtype_for(args.compute_precision),
        )

    @staticmethod
    def warn_guidance_on_mobile(args: Namespace, model_config: ModelConfig) -> None:
        if not model_config.supports_guidance and CommandLineParser._option_was_provided("--guidance"):
            warnings.warn(f"--guidance {args.guidance}: {MOBILE_GUIDANCE_WARNING}", stacklevel=2)

    @staticmethod
    def warn_unused_guidance_options(args: Namespace, model_config: ModelConfig) -> None:
        # Guidance is off on Mobile, and on the base model when set to 1 or below
        guidance_off = not model_config.supports_guidance or (args.guidance is not None and args.guidance <= 1.0)
        for option, entry in GUIDANCE_OPTIONS.items():
            if guidance_off and CommandLineParser._option_was_provided(option):
                warnings.warn(f"{option} is ignored at guidance 1.0; {entry['reason']}", stacklevel=2)

    @staticmethod
    def generate(model: DreamLite, args: Namespace, seed: int, prompt: str) -> GeneratedImage:
        width, height = DimensionResolver.resolve(width=args.width, height=args.height)
        return model.generate_image(
            seed=seed,
            prompt=prompt,
            negative_prompt=PromptUtil.read_negative_prompt(args),
            width=width,
            height=height,
            guidance=args.guidance,
            num_inference_steps=args.steps,
        )


def main():
    # 0. Parse command line arguments
    parser = build_parser()
    args = parser.parse_args()

    # 1. Check the flags, then load the model
    try:
        model_config = DreamLiteCommand.validate(args)
    except ValueError as exc:
        parser.error(str(exc))
    CommandLineParser.warn_ignored_options(IGNORED_OPTIONS)
    DreamLiteCommand.warn_guidance_on_mobile(args, model_config)
    DreamLiteCommand.warn_unused_guidance_options(args, model_config)
    model = DreamLiteCommand.load(args)

    # 2. Register callbacks
    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=DreamLiteCommand.latent_creator,
    )

    try:
        for seed in args.seed:
            # 3. Generate an image for each seed value
            image = DreamLiteCommand.generate(model, args, seed, PromptUtil.read_prompt(args))
            # 4. Save the image
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
