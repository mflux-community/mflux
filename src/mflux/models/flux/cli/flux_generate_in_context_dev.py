from pathlib import Path

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.defaults import defaults as ui_defaults
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.config import ModelConfig
from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
from mflux.models.flux.variants.in_context.flux_in_context_dev import Flux1InContextDev
from mflux.models.flux.variants.in_context.utils.in_context_loras import get_lora_path
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil

# Single source of truth for options this CLI accepts but cannot honour: the runtime
# warning and the mflux-capabilities dump both read it.
IGNORED_OPTIONS = {
    "--negative-prompt": "FLUX.1 uses distilled guidance and has no negative branch.",
}


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image using in-context LoRA with a reference image.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_in_context_dev_arguments()
    parser.add_in_context_arguments()
    parser.add_output_arguments()
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    CommandLineParser.warn_ignored_options(IGNORED_OPTIONS)

    # 0. Set default guidance value if not provided by user
    if args.guidance is None:
        args.guidance = ui_defaults.GUIDANCE_SCALE

    # Build lora_paths: style LoRA (if specified) + user-provided LoRAs
    lora_paths = []
    lora_scales = []
    if args.lora_style:
        lora_paths.append(get_lora_path(args.lora_style))
        lora_scales.append(1.0)
    if args.lora_paths:
        lora_paths.extend(args.lora_paths)
        lora_scales.extend(args.lora_scales or [1.0] * len(args.lora_paths))

    # 1. Load the model
    flux = Flux1InContextDev(
        model_config=ModelConfig.dev(),
        quantize=args.quantize,
        model_path=args.model_path,
        lora_paths=lora_paths or None,
        lora_scales=lora_scales or None,
    )

    # 2. Register callbacks
    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=flux,
        latent_creator=FluxLatentCreator,
    )

    try:
        for seed in args.seed:
            # 3. Generate an image for each seed value
            image = flux.generate_image(
                seed=seed,
                prompt=PromptUtil.read_prompt(args),
                width=args.width,
                height=args.height,
                guidance=args.guidance,
                scheduler=args.scheduler,
                image_path=args.image_path,
                num_inference_steps=args.steps,
            )
            # 4. Save the image
            output_path = Path(args.output.format(seed=seed))
            image.get_right_half().save(path=output_path, export_json_metadata=args.metadata)
            if args.save_full_image:
                image.save(path=output_path.with_stem(output_path.stem + "_full"))

    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
