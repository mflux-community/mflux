from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.boogu.variants import BooguImage
from mflux.models.common.config import ModelConfig
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil

# The model this CLI runs when --model is omitted. The parser needs it too, to key the
# --steps default off the right model instead of falling back to FLUX.1-dev's 25.
DEFAULT_MODEL = "boogu-image-turbo"

# Single source of truth for options this CLI accepts but cannot honour: the runtime
# warning and the mflux-capabilities dump both read it.
IGNORED_OPTIONS = {
    "--guidance": "Boogu Image Turbo is guidance-distilled; CFG is disabled.",
    "--negative-prompt": "CFG is disabled on Boogu Image Turbo, so the negative prompt is never encoded.",
}


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(
        description="Generate an image using Boogu-Image-Turbo (4-step DMD). "
        "Tip: 4 steps is enough up to ~768px; use --steps 8 at 1024x1024, where 4 steps under-resolves detail."
    )
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False, default_model=DEFAULT_MODEL)
    # No add_lora_arguments(): mflux has no LoRA mapping for Boogu, and accepting the
    # flags meant resolving and downloading the adapter before dropping it silently.
    # Same shape as mflux-generate-lens, which has never taken them.
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_output_arguments()
    return parser


def main():
    # 0. Parse command line arguments
    parser = build_parser()
    args = parser.parse_args()
    CommandLineParser.warn_ignored_options(IGNORED_OPTIONS)

    model_config = ModelConfig.from_name(model_name=args.model or DEFAULT_MODEL)

    model = BooguImage(
        model_config=model_config,
        quantize=args.quantize,
        model_path=args.model_path,
    )

    # Boogu builds its own noise latents (no LatentCreator); stepwise output is unsupported.
    memory_saver = CallbackManager.register_callbacks(args=args, model=model, latent_creator=None)

    try:
        for seed in args.seed:
            image = model.generate_image(
                seed=seed,
                prompt=PromptUtil.read_prompt(args),
                width=args.width,
                height=args.height,
                num_inference_steps=args.steps,
            )
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
