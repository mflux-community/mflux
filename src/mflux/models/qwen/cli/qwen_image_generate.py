import copy
from argparse import Namespace

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.defaults import defaults as ui_defaults
from mflux.cli.parser.parsers import CommandLineParser, lora_init_kwargs_from_args
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.prompt_util import PromptUtil

# The model this CLI runs. The parser needs it to key the --steps default off the right
# model instead of falling back to FLUX.1-dev's 25.
DEFAULT_MODEL = "qwen-image"


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image using Qwen Image model.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False, default_model=DEFAULT_MODEL)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_image_to_image_arguments(required=False)
    parser.add_pid_decode_arguments()
    parser.add_output_arguments()
    return parser


class QwenImageCommand:
    latent_creator = QwenLatentCreator

    @staticmethod
    def validate(args: Namespace) -> ModelConfig:
        # Weight-free: resolves --model against the in-memory registry only, so a name this
        # command cannot run fails here, before anything loads.
        return ConfigResolution.resolve_restricted(
            args.model,
            DEFAULT_MODEL,
            model_path=args.model_path,
            base_model=args.base_model,
        )

    @staticmethod
    def load(args: Namespace) -> QwenImage:
        return QwenImage(
            model_config=QwenImageCommand.validate(args),
            quantize=args.quantize,
            model_path=args.model_path,
            **lora_init_kwargs_from_args(args),
        )

    @staticmethod
    def generate(model: QwenImage, args: Namespace, seed: int, prompt: str) -> GeneratedImage:
        # ScaleFactor dims ("2x", and the "auto" default) need the reference image; resolved
        # here so a caller of generate() gets them without going through main().
        width, height = DimensionResolver.resolve(
            width=args.width, height=args.height, reference_image_path=args.image_path
        )
        return model.generate_image(
            seed=seed,
            prompt=prompt,
            negative_prompt=PromptUtil.read_negative_prompt(args),
            width=width,
            height=height,
            guidance=QwenImageCommand._guidance(args),
            scheduler=args.scheduler,
            image_path=args.image_path,
            num_inference_steps=args.steps,
            image_strength=args.image_strength,
            pid_decode=args.pid_decode,
            pid_degrade_sigma=args.pid_degrade_sigma,
        )

    @staticmethod
    def _guidance(args: Namespace) -> float:
        # The command's default. generate() applies it, so a direct caller gets it too, and
        # args stays as passed.
        return ui_defaults.GUIDANCE_SCALE if args.guidance is None else args.guidance


def main():
    # 0. Parse command line arguments
    parser = build_parser()
    args = parser.parse_args()

    # 1. Load the model
    model = QwenImageCommand.load(args)

    # 2. Register callbacks
    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=QwenImageCommand.latent_creator,
    )

    try:
        # Sizes are resolved once per run, before the first prompt is read, into a copy so the
        # args the callbacks were given keep the flags as passed. generate() resolves again for
        # direct callers, which changes nothing on plain numbers.
        run_args = copy.copy(args)
        run_args.width, run_args.height = DimensionResolver.resolve(
            width=args.width, height=args.height, reference_image_path=args.image_path
        )
        for seed in args.seed:
            # 3. Generate an image for each seed value
            image = QwenImageCommand.generate(model, run_args, seed, PromptUtil.read_prompt(args))
            # 4. Save the image
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
