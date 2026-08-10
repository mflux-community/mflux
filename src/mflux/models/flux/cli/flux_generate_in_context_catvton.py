from pathlib import Path

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.defaults import defaults as ui_defaults
from mflux.cli.parser.parsers import CommandLineParser, lora_init_kwargs_from_args
from mflux.models.common.config import ModelConfig
from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
from mflux.models.flux.variants.in_context.flux_in_context_fill import Flux1InContextFill
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Generate virtual try-on images using in-context learning.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=False, require_prompt=False)
    parser.add_catvton_arguments()
    parser.add_in_context_arguments()
    parser.add_output_arguments()
    args = parser.parse_args()

    # 0. Default to a higher guidance value for fill
    if args.guidance is None:
        args.guidance = ui_defaults.DEFAULT_DEV_FILL_GUIDANCE

    # Set default CATVTON prompt if none provided
    if not args.prompt and not args.prompt_file:
        args.prompt = "The pair of images highlights a clothing and its styling on a model, high resolution, 4K, 8K; [IMAGE1] Detailed product shot of a clothing; [IMAGE2] The same cloth is worn by a model in a lifestyle setting."

    # 1. Load the model
    flux = Flux1InContextFill(
        model_config=ModelConfig.dev_fill_catvton(),
        quantize=args.quantize,
        model_path=args.model_path,
        **lora_init_kwargs_from_args(args),
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
                num_inference_steps=args.steps,
                left_image_path=args.garment_image,
                right_image_path=args.person_image,
                masked_image_path=args.person_mask,
            )

            # 4. Save the image(s)
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
