from argparse import Namespace

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser, lora_init_kwargs_from_args
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.sensenova.latent_creator.sensenova_latent_creator import SenseNovaLatentCreator
from mflux.models.sensenova.model.sensenova_scheduler.sensenova_scheduler import SenseNovaScheduler
from mflux.models.sensenova.variants import SenseNova
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.prompt_util import PromptUtil

# The model this CLI runs when --model is omitted. The parser also reads it for the --steps default.
DEFAULT_MODEL = "sensenova-u1.5"
DEFAULT_GUIDANCE = SenseNova.DEFAULT_GUIDANCE
DEFAULT_BAKE_LORA = SenseNova.DEFAULT_BAKE_LORA

# Options this CLI accepts but cannot honour. The runtime warning and mflux-capabilities read this.
REJECTED_OPTIONS = {
    "--negative-prompt": "SenseNova-U1.5 has no negative prompt; the CLI exits with an error when this is set.",
}

IGNORED_OPTIONS = {
    "--scheduler": "SenseNova-U1.5 samples on the shifted flow schedule it was trained with.",
    "--lora-style": "Named LoRA styles are only supported by the Flux in-context CLI; use --lora.",
    "--vae-tiling": "SenseNova-U1.5 generates directly in pixel space; there is no VAE to tile.",
    "--vae-tile-size": "SenseNova-U1.5 generates directly in pixel space; there is no VAE to tile.",
}


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(
        description="Generate an image using SenseNova-U1.5 (50 steps, CFG 4; 8 steps and CFG 1 with the 8-step LoRA)."
    )
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False, default_model=DEFAULT_MODEL)
    parser.add_lora_arguments(bake_by_default=DEFAULT_BAKE_LORA)
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_output_arguments()
    parser.set_defaults(guidance=DEFAULT_GUIDANCE)
    return parser


class SenseNovaCommand:
    latent_creator = SenseNovaLatentCreator

    @staticmethod
    def validate(args: Namespace) -> ModelConfig:
        # Weight-free: a bad value fails before the 35 GB checkpoint loads.
        model_config = ConfigResolution.resolve_restricted(
            args.model,
            DEFAULT_MODEL,
            model_path=args.model_path,
            base_model=args.base_model,
        )
        SenseNovaCommand.check_size_and_steps(args)
        return model_config

    @staticmethod
    def check_size_and_steps(args: Namespace) -> None:
        SenseNovaLatentCreator.check_size(height=args.height, width=args.width)
        if args.steps is not None:
            SenseNovaScheduler.check_steps(args.steps)

    @staticmethod
    def load(args: Namespace) -> SenseNova:
        return SenseNova(
            model_config=SenseNovaCommand.validate(args),
            quantize=args.quantize,
            model_path=args.model_path,
            **lora_init_kwargs_from_args(args),
        )

    @staticmethod
    def generate(model: SenseNova, args: Namespace, seed: int, prompt: str) -> GeneratedImage:
        return model.generate_image(
            seed=seed,
            prompt=prompt,
            width=args.width,
            height=args.height,
            num_inference_steps=args.steps,
            guidance=DEFAULT_GUIDANCE if args.guidance is None else args.guidance,
        )


def main():
    # 0. Parse command line arguments
    parser = build_parser()
    args = parser.parse_args()
    # Keyed on the option, not the value: a --config-from-metadata sidecar can carry a negative prompt the user never typed
    if CommandLineParser._option_was_provided("--negative-prompt"):
        parser.error("--negative-prompt is not supported for SenseNova-U1.5: its unconditional branch is a fixed empty query.")  # fmt: skip
    CommandLineParser.warn_ignored_options(IGNORED_OPTIONS)
    # A bad size or step count is a usage error before the checkpoint loads, not a traceback
    try:
        SenseNovaCommand.check_size_and_steps(args)
    except ValueError as exc:
        parser.error(str(exc))

    # 1. Load the model
    model = SenseNovaCommand.load(args)

    # 2. Register callbacks
    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=SenseNovaCommand.latent_creator,
    )

    try:
        for seed in args.seed:
            # 3. Generate an image for each seed value
            image = SenseNovaCommand.generate(model, args, seed, PromptUtil.read_prompt(args))
            # 4. Save the image
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
