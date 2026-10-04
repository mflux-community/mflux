from argparse import Namespace

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
from mflux.models.lens.variants.txt2img.lens_image import LensImage
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.prompt_util import PromptUtil

# The model this CLI runs when --model is omitted. The parser needs it too, to key the
# --steps default off the right model instead of falling back to FLUX.1-dev's 25.
DEFAULT_MODEL = "lens-turbo"

# Single source of truth for options this CLI accepts but cannot honour: the runtime
# warning and the mflux-capabilities dump both read it.
IGNORED_OPTIONS = {
    "--guidance": "Lens Turbo is a 4-step distillation with CFG internalized; guidance is never applied.",
    "--negative-prompt": "CFG is disabled on Lens Turbo, so the negative prompt is never encoded.",
    "--scheduler": "Lens Turbo runs the shifted sigma schedule its distillation was trained on.",
}


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image using Microsoft Lens (Turbo).")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False, default_model=DEFAULT_MODEL)
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_output_arguments()
    return parser


class LensCommand:
    # Lens rides FLUX.2 latents, so the flux2 unpacker is the one that turns its packed
    # (1, seq, 128) tensor back into something the VAE can decode for stepwise output.
    latent_creator = Flux2LatentCreator

    @staticmethod
    def validate(args: Namespace) -> ModelConfig:
        # Weight-free: resolves --model against the in-memory registry only. --model accepts
        # only lens aliases; anything else errors instead of being silently run as Lens Turbo.
        return ConfigResolution.resolve_restricted(args.model, DEFAULT_MODEL, model_path=args.model_path)

    @staticmethod
    def load(args: Namespace) -> LensImage:
        return LensImage(
            model_config=LensCommand.validate(args),
            quantize=args.quantize,
            model_path=args.model_path,
        )

    @staticmethod
    def generate(model: LensImage, args: Namespace, seed: int, prompt: str) -> GeneratedImage:
        width, height = DimensionResolver.resolve(width=args.width, height=args.height)
        return model.generate_image(
            seed=seed,
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=args.steps,
        )


def main():
    parser = build_parser()
    args = parser.parse_args()
    CommandLineParser.warn_ignored_options(IGNORED_OPTIONS)

    model = LensCommand.load(args)

    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=LensCommand.latent_creator,
    )

    try:
        for seed in args.seed:
            image = LensCommand.generate(model, args, seed, PromptUtil.read_prompt(args))
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
