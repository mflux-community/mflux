from argparse import Namespace

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser, lora_init_kwargs_from_args
from mflux.models.common.config.model_config import AVAILABLE_MODELS, ModelConfig
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.moss_image.latent_creator import MOSSImageLatentCreator
from mflux.models.moss_image.model.moss_image_scheduler import MOSSImageScheduler
from mflux.models.moss_image.model.moss_image_transformer.moss_image_transformer import MOSSImageTransformer
from mflux.models.moss_image.variants import MOSSImage
from mflux.models.moss_image.weights.moss_image_weight_mapping import MOSSImageWeightMapping
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.prompt_util import PromptUtil

# The model this CLI runs when --model is omitted. The parser also reads it for the --steps default.
DEFAULT_MODEL = "moss-image-sft"
DEFAULT_GUIDANCE = MOSSImage.DEFAULT_GUIDANCE
DEFAULT_UNCOND_SKIP_LAYERS = MOSSImage.DEFAULT_UNCOND_SKIP_LAYERS
DEFAULT_BAKE_LORA = MOSSImage.DEFAULT_BAKE_LORA

# The other entry this CLI serves: MOSS-Image-Base. Its model card gives the same sampling settings as SFT.
FAMILY_MODELS = tuple(key for key in AVAILABLE_MODELS if key.startswith("moss-image-") and key != DEFAULT_MODEL)

# Options this CLI accepts but cannot honour. The runtime warning and mflux-capabilities read this.
IGNORED_OPTIONS = {
    "--scheduler": "MOSS-Image samples on the snapped flow-matching schedule it was trained with.",
    "--lora-style": "Named LoRA styles are only supported by the Flux in-context CLI; use --lora.",
}


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(
        description="Generate an image using MOSS-Image (SFT or Base, 50 steps, CFG 3) based on a prompt."
    )
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False, default_model=DEFAULT_MODEL)
    parser.add_lora_arguments(bake_by_default=DEFAULT_BAKE_LORA)
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_step_cache_arguments()
    parser.add_ssd_stream_arguments()
    parser.add_stream_lora_arguments()
    parser.add_argument("--uncond-skip-layers", type=int, nargs="*", default=None, metavar="LAYER", help=f"Zero-based transformer layers that the negative CFG branch skips. Pass the flag with no values to skip no layer. Default: {' '.join(map(str, DEFAULT_UNCOND_SKIP_LAYERS))}.")  # fmt: off
    parser.add_output_arguments()
    parser.set_defaults(guidance=DEFAULT_GUIDANCE)
    return parser


# Options that have no effect when --ssd-stream is on (the VAE then always decodes in row strips).
SSD_STREAM_IGNORED_OPTIONS = {
    "--vae-tiling": "--ssd-stream decodes the image in row strips instead.",
    "--vae-tile-size": "--ssd-stream decodes the image in row strips instead.",
}


class MOSSImageCommand:
    latent_creator = MOSSImageLatentCreator

    @staticmethod
    def validate(args: Namespace) -> ModelConfig:
        # Weight-free: resolves --model against the in-memory registry only. The size and step
        # checks run here so that a bad value fails before the 27 GB checkpoint loads.
        model_config = ConfigResolution.resolve_restricted(
            args.model,
            DEFAULT_MODEL,
            model_path=args.model_path,
            extra_keys=FAMILY_MODELS,
            base_model=args.base_model,
        )
        MOSSImageLatentCreator.check_size(height=args.height, width=args.width)
        if args.steps is not None:
            MOSSImageScheduler.compute_sigmas(args.steps)
        MOSSImageTransformer.check_skip_layers(
            MOSSImageCommand._uncond_skip_layers(args), MOSSImageWeightMapping.NUM_TRANSFORMER_LAYERS
        )
        return model_config

    @staticmethod
    def load(args: Namespace) -> MOSSImage:
        return MOSSImage(
            model_config=MOSSImageCommand.validate(args),
            quantize=args.quantize,
            model_path=args.model_path,
            ssd_stream=args.ssd_stream,
            stream_lora=args.stream_lora,
            **lora_init_kwargs_from_args(args),
        )

    @staticmethod
    def generate(model: MOSSImage, args: Namespace, seed: int, prompt: str) -> GeneratedImage:
        return model.generate_image(
            seed=seed,
            prompt=prompt,
            width=args.width,
            height=args.height,
            num_inference_steps=args.steps,
            guidance=MOSSImageCommand._guidance(args),
            negative_prompt=args.negative_prompt,
            step_cache_ratio=args.step_cache_ratio,
            uncond_skip_layers=MOSSImageCommand._uncond_skip_layers(args),
        )

    @staticmethod
    def _uncond_skip_layers(args: Namespace) -> tuple[int, ...]:
        layers = getattr(args, "uncond_skip_layers", None)
        return DEFAULT_UNCOND_SKIP_LAYERS if layers is None else tuple(layers)

    @staticmethod
    def _guidance(args: Namespace) -> float:
        # The parser sets the default. A namespace that a caller builds or edits can still hold None.
        return DEFAULT_GUIDANCE if args.guidance is None else args.guidance


def main():
    # 0. Parse command line arguments
    parser = build_parser()
    args = parser.parse_args()
    CommandLineParser.warn_ignored_options(IGNORED_OPTIONS)
    if args.ssd_stream:
        CommandLineParser.warn_ignored_options(SSD_STREAM_IGNORED_OPTIONS)

    # 1. Load the model
    model = MOSSImageCommand.load(args)

    # 2. Register callbacks
    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=MOSSImageCommand.latent_creator,
    )

    try:
        for seed in args.seed:
            # 3. Generate an image for each seed value
            image = MOSSImageCommand.generate(model, args, seed, PromptUtil.read_prompt(args))
            # 4. Save the image
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
