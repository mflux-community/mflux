from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.boogu.variants import BooguImage
from mflux.models.common.config import ModelConfig
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.image_util import ImageUtil
from mflux.utils.prompt_util import PromptUtil


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(
        description="Generate an image using Boogu-Image-Turbo (4-step DMD). "
        "Tip: 4 steps is enough up to ~768px; use --steps 8 at 1024x1024, where 4 steps under-resolves detail."
    )
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_output_arguments()
    args = parser.parse_args()

    model_config = ModelConfig.from_name(model_name=args.model or "boogu-image-turbo")

    model = BooguImage(
        model_config=model_config,
        quantize=args.quantize,
        model_path=args.model_path,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
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
            ImageUtil.save_image(
                image=image,
                path=args.output.format(seed=seed),
                export_json_metadata=args.metadata,
            )
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
