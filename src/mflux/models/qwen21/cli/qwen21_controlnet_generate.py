from argparse import Namespace

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser, lora_init_kwargs_from_args
from mflux.models.common.compute_precision import ComputePrecision
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.qwen21.latent_creator.qwen_image21_latent_creator import QwenImage21LatentCreator
from mflux.models.qwen21.variants.controlnet.qwen_image_21_controlnet import QwenImage21Controlnet
from mflux.utils.exceptions import ModelConfigError, PromptFileReadError, StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.prompt_util import PromptUtil

DEFAULT_MODEL = "qwen-image-2.1-controlnet"
IGNORED_OPTIONS = {
    "--lora-style": "Named LoRA styles are only supported by the Flux in-context CLI; use --lora.",
    "--scheduler": "The ControlNet runs the linear Euler schedule of Qwen-Image-2.1; other schedulers are not wired.",
}
CONDITIONAL_OPTIONS = {
    "--negative-prompt": {
        "condition": "guidance greater than 1",
        "reason": "Guidance 1 runs only the positive branch.",
    },
}


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(
        description="Generate with Qwen-Image-2.1 and its ControlNet Union: a control image, an inpaint, or both."
    )
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False, default_model=DEFAULT_MODEL)
    # --model is optional here: without it the registry entry of the ControlNet is the model.
    parser.set_defaults(model=DEFAULT_MODEL)
    parser.add_compute_precision_arguments()
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=False)
    parser.set_defaults(width=None, height=None)
    for flag in ("--width", "--height"):
        parser._option_string_actions[flag].help = (
            "Output size, a multiple of 32. If omitted, derive it from --output-resolution and the aspect "
            "ratio of the control image (or of the inpaint source when there is no control image)."
        )
    parser.add_controlnet_arguments()
    parser.set_defaults(controlnet_strength=1.0)
    parser._option_string_actions["--controlnet-image-path"].help = (
        "Control image at the target framing: canny, depth, grayscale, HED, lineart, MLSD, pose or scribble. "
        "One checkpoint reads all of them; no type flag is needed."
    )
    parser._option_string_actions[
        "--controlnet-strength"
    ].help = "Scale of the control skips: 1.0 (default) is the strongest, lower weakens it, 0 turns the control off."
    parser.add_argument("--image-path", type=str, default=None, help="Inpaint source image. Needs --mask-image.")
    parser.add_argument(
        "--mask-image",
        type=str,
        default=None,
        help="Inpaint mask for --image-path: white is regenerated from the prompt, black is kept.",
    )
    parser.add_argument(
        "--output-resolution",
        type=int,
        default=1024,
        help="Pixel-area budget for automatic output dimensions (default: 1024).",
    )
    parser.add_output_arguments()
    return parser


class Qwen21ControlnetCommand:
    latent_creator = QwenImage21LatentCreator

    @staticmethod
    def validate(args: Namespace) -> ModelConfig:
        # Weight-free: checked before the model loads, so a bad request fails fast.
        model_config = ConfigResolution.resolve(args.model, args.base_model)
        if not model_config.controlnet_model:
            raise ValueError(
                f"--model {args.model!r} is not a Qwen-Image-2.1 ControlNet model. Use {DEFAULT_MODEL} (the default)."
            )
        if args.controlnet_image_path is None and args.image_path is None:
            raise ValueError("Give --controlnet-image-path, or --image-path with --mask-image to inpaint.")
        if (args.image_path is None) != (args.mask_image is None):
            raise ValueError("Inpainting needs both --image-path and --mask-image.")
        for name in ("width", "height"):
            value = getattr(args, name)
            if value is not None and (not isinstance(value, int) or value < 32 or value % 32):
                raise ValueError(f"--{name} must be a positive multiple of 32.")
        QwenImage21LatentCreator.validate_resolution(args.output_resolution)
        return model_config

    @staticmethod
    def load(args: Namespace) -> QwenImage21Controlnet:
        return QwenImage21Controlnet(
            model_config=Qwen21ControlnetCommand.validate(args),
            quantize=args.quantize,
            model_path=args.model_path,
            compute_precision=ComputePrecision.dtype_for(args.compute_precision),
            **lora_init_kwargs_from_args(args),
        )

    @staticmethod
    def generate(model: QwenImage21Controlnet, args: Namespace, seed: int, prompt: str) -> GeneratedImage:
        return model.generate_image(
            seed=seed,
            prompt=prompt,
            controlnet_image_path=args.controlnet_image_path,
            controlnet_strength=args.controlnet_strength,
            num_inference_steps=args.steps,
            height=args.height,
            width=args.width,
            guidance=1.0 if args.guidance is None else args.guidance,
            negative_prompt=args.negative_prompt,
            output_resolution=args.output_resolution,
            image_path=args.image_path,
            mask_image=args.mask_image,
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    CommandLineParser.warn_ignored_options(IGNORED_OPTIONS)
    try:
        Qwen21ControlnetCommand.validate(args)
    except ModelConfigError:
        raise
    except ValueError as exc:
        parser.error(str(exc))
    model = Qwen21ControlnetCommand.load(args)
    memory_saver = CallbackManager.register_callbacks(
        args=args, model=model, latent_creator=Qwen21ControlnetCommand.latent_creator
    )
    try:
        for seed in args.seed:
            image = Qwen21ControlnetCommand.generate(model, args, seed, PromptUtil.read_prompt(args))
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
