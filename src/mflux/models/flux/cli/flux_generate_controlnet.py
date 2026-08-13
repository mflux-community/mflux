from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.defaults import defaults as ui_defaults
from mflux.cli.parser.parsers import CommandLineParser, lora_init_kwargs_from_args
from mflux.models.common.config import ModelConfig
from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
from mflux.models.flux.variants.controlnet.flux_controlnet import Flux1Controlnet
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil

# Single source of truth for options this CLI accepts but cannot honour: the runtime
# warning and the mflux-capabilities dump both read it.
IGNORED_OPTIONS = {
    "--negative-prompt": "FLUX.1 uses distilled guidance and has no negative branch.",
}
CONDITIONAL_OPTIONS = {
    "--guidance": {
        "condition": "the resolved model supports guidance (dev does; schnell does not)",
        "reason": "schnell builds no guidance embedder, so the value has no path to affect the output.",
    },
}


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image based on a prompt and a controlnet reference image.")  # fmt: off
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=True)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=False)
    parser.add_controlnet_arguments(mode="canny")
    parser.add_output_arguments()
    return parser


def main():
    # 0. Parse command line arguments
    parser = build_parser()
    args = parser.parse_args()
    CommandLineParser.warn_ignored_options(IGNORED_OPTIONS)

    # 0. Set default guidance value if not provided by user
    if args.guidance is None:
        args.guidance = ui_defaults.GUIDANCE_SCALE

    # 1. Load the model
    model_config = _get_controlnet_model_config(args.model)
    if not model_config.supports_guidance:
        CommandLineParser.warn_ignored_options({"--guidance": CONDITIONAL_OPTIONS["--guidance"]["reason"]})

    flux = Flux1Controlnet(
        model_config=model_config,
        quantize=args.quantize,
        model_path=args.model_path,
        **lora_init_kwargs_from_args(args),
    )

    # 2. Register callbacks
    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=flux,
        latent_creator=FluxLatentCreator,
        enable_canny_saver=True,
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
                num_inference_steps=args.steps,
                controlnet_strength=args.controlnet_strength,
                controlnet_image_path=args.controlnet_image_path,
            )

            # 4. Save the image
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


def _get_controlnet_model_config(model_name: str) -> ModelConfig:
    if model_name == "schnell":
        return ModelConfig.schnell_controlnet_canny()
    return ModelConfig.dev_controlnet_canny()


if __name__ == "__main__":
    main()
