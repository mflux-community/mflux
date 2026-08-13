from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.config import ModelConfig
from mflux.models.lens.variants.txt2img.lens_image import LensImage
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import ModelConfigError, PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil

# Single source of truth for options this CLI accepts but cannot honour: the runtime
# warning and the mflux-capabilities dump both read it.
IGNORED_OPTIONS = {
    "--guidance": "Lens Turbo is a 4-step distillation with CFG internalized; guidance is never applied.",
    "--negative-prompt": "CFG is disabled on Lens Turbo, so the negative prompt is never encoded.",
}


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image using Microsoft Lens (Turbo).")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_output_arguments()
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    model_config = ModelConfig.lens_turbo()
    if args.model is not None:
        try:
            model_config = ModelConfig.from_name(args.model)
        except ModelConfigError:
            if args.model_path is None:
                raise
        if model_config.model_name != ModelConfig.lens_turbo().model_name:
            raise ModelConfigError(
                f"'{args.model}' is not a Lens model; this CLI only accepts the lens aliases "
                f"{ModelConfig.lens_turbo().aliases}."
            )

    model = LensImage(
        model_config=model_config,
        quantize=args.quantize,
        model_path=args.model_path,
    )

    try:
        width, height = DimensionResolver.resolve(width=args.width, height=args.height)
        for seed in args.seed:
            image = model.generate_image(
                seed=seed,
                prompt=PromptUtil.read_prompt(args),
                width=width,
                height=height,
                num_inference_steps=args.steps,
            )
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)


if __name__ == "__main__":
    main()
