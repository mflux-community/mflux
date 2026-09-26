from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.ming_image.latent_creator.ming_latent_creator import MingLatentCreator
from mflux.models.ming_image.variants.ming_image import MingImage
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil

DEFAULT_MODEL = "ming-image-design"

IGNORED_OPTIONS = {
    "--negative-prompt": "Ming's negative condition is the zeroed prompt embedding; there is no negative text.",
    "--scheduler": "Ming runs the flow-match Euler schedule with the checkpoint's static shift of 6.",
}


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image (RGBA) using inclusionAI Ming-Image-0.1-Design.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False, default_model=DEFAULT_MODEL)
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_output_arguments()
    parser.add_argument(
        "--text-encoder-quantize",
        type=int,
        choices=[3, 4, 5, 6, 8],
        default=None,
        help="Quantize the Ling-mini-2.0 text encoder separately (defaults to --quantize). "
        "4 with --quantize 8 keeps the DiT at 8-bit while the ~16B MoE text encoder fits in ~9 GB.",
    )
    parser.add_argument(
        "--flatten-alpha",
        action="store_true",
        help="Composite the RGBA output onto white and save RGB.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    CommandLineParser.warn_ignored_options(IGNORED_OPTIONS)

    model = MingImage(
        model_config=ConfigResolution.resolve_restricted(args.model, DEFAULT_MODEL, model_path=args.model_path),
        quantize=args.quantize,
        text_encoder_quantize=args.text_encoder_quantize,
        model_path=args.model_path,
        low_ram=args.low_ram,
    )

    memory_saver = CallbackManager.register_callbacks(args=args, model=model, latent_creator=MingLatentCreator)

    try:
        width, height = DimensionResolver.resolve(width=args.width, height=args.height)
        prompt = PromptUtil.read_prompt(args)
        # Every seed reuses this one prompt, so the ~16B-parameter text encoder can be freed
        # before the DiT runs: peak memory is then max(text side, DiT) rather than their sum.
        model.encode_prompt(prompt)
        model.release_text_encoder()
        for seed in args.seed:
            image = model.generate_image(
                seed=seed,
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=args.steps,
                guidance=args.guidance,
            )
            if args.flatten_alpha:
                image.image = MingImage.to_rgb(image.image)
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
