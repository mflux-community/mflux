import warnings

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser, lora_init_kwargs_from_args
from mflux.models.common.config import ModelConfig
from mflux.models.ideogram4.latent_creator import Ideogram4LatentCreator
from mflux.models.ideogram4.model.ideogram4_scheduler import Ideogram4Scheduler
from mflux.models.ideogram4.variants.txt2img.ideogram4 import Ideogram4
from mflux.models.ideogram4.weights.ideogram4_weight_definition import Ideogram4WeightDefinition
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil


def main():
    parser = CommandLineParser(description="Generate an image using Ideogram 4.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_output_arguments()
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=sorted(Ideogram4Scheduler.PRESETS),
        help="Ideogram 4 sampler preset (step count, guidance schedule, and noise schedule). Default is V4_DEFAULT_20.",
    )
    parser.add_argument(
        "--strict-caption-validation",
        action="store_true",
        help="Fail when an Ideogram 4 JSON caption has schema warnings.",
    )
    parser.add_argument(
        "--cfg-end",
        type=float,
        default=None,
        help="Fraction of steps (0-1) that run CFG; the remaining steps run cond-only "
        "(guidance 1.0, skipping the unconditional forward). Lower = faster. Default: full CFG.",
    )
    args = parser.parse_args()

    model_name = args.model or "ideogram4"
    if Ideogram4WeightDefinition.is_builtin_name(model_name):
        model_config = ModelConfig.from_name(model_name)
        model_path = None
    else:
        model_config = ModelConfig.ideogram4_fp8()
        model_path = args.model_path
    if CommandLineParser._option_was_provided("--steps"):
        warnings.warn("--steps is ignored; Ideogram 4 presets define the step count.", stacklevel=1)
    if CommandLineParser._option_was_provided("--guidance"):
        warnings.warn("--guidance is ignored; Ideogram 4 presets define the guidance schedule.", stacklevel=1)

    model = Ideogram4(
        model_config=model_config,
        quantize=args.quantize,
        model_path=model_path,
        **lora_init_kwargs_from_args(args),
    )

    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=Ideogram4LatentCreator,
    )

    try:
        width, height = DimensionResolver.resolve(
            width=args.width,
            height=args.height,
            reference_image_path=None,
        )

        for seed in args.seed:
            image = model.generate_image(
                seed=seed,
                prompt=PromptUtil.read_prompt(args),
                width=width,
                height=height,
                preset=args.preset,
                strict_caption_validation=args.strict_caption_validation,
                cfg_end=args.cfg_end,
            )
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
