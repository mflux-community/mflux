from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.defaults import defaults as ui_defaults
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.fibo.latent_creator.fibo_latent_creator import FiboLatentCreator
from mflux.models.fibo.variants.txt2img.fibo import FIBO
from mflux.models.fibo.variants.txt2img.util import FiboUtil
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import ModelConfigError, PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image using FIBO model.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.set_defaults(model="fibo")
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_image_to_image_arguments(required=False)
    parser.add_output_arguments()
    return parser


def main():
    # 0. Parse command line arguments
    parser = build_parser()
    args = parser.parse_args()

    try:
        model_config = ModelConfig.from_name(model_name=args.model, base_model=args.base_model)
    except ModelConfigError:
        model_config = ModelConfig.fibo()

    # 0. Set default guidance value if not provided by user
    if args.guidance is None:
        if "fibo-lite" in model_config.aliases:
            args.guidance = 1.0  # distilled, no CFG
        elif "fibo" in model_config.aliases:
            args.guidance = 5.0  # base FIBO typical
        else:
            args.guidance = ui_defaults.GUIDANCE_SCALE

    json_prompt = FiboUtil.get_json_prompt(args, quantize=args.quantize)

    # 1. Load the FIBO model
    fibo = FIBO(
        quantize=args.quantize,
        model_path=args.model_path,
        model_config=model_config,
    )

    # 2. Register callbacks
    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=fibo,
        latent_creator=FiboLatentCreator,
    )

    try:
        # Resolve dimensions (supports ScaleFactor like "2x" when --image-path is provided)
        width, height = DimensionResolver.resolve(
            width=args.width,
            height=args.height,
            reference_image_path=args.image_path,
        )

        for seed in args.seed:
            # 3. Generate an image for each seed value
            image = fibo.generate_image(
                seed=seed,
                prompt=json_prompt,
                width=width,
                height=height,
                guidance=args.guidance,
                image_path=args.image_path,
                num_inference_steps=args.steps,
                image_strength=args.image_strength,
                scheduler="flow_match_euler_discrete",
                negative_prompt=PromptUtil.read_negative_prompt(args),
            )

            # 4. Save the image
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
