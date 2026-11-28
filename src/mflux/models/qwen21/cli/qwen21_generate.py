from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser, lora_init_kwargs_from_args
from mflux.models.common.compute_precision import ComputePrecision
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.common.vae.vae_decode_options import VAEDecodeOptions
from mflux.models.qwen21.cli import qwen21_vae_options
from mflux.models.qwen21.latent_creator.qwen21_latent_creator import Qwen21LatentCreator
from mflux.models.qwen21.model.qwen21_scheduler import Qwen21TurboScheduler, ViggleTurboScheduler
from mflux.models.qwen21.model.qwen21_small_text_encoder.qwen21_small_text_encoder import Qwen21SmallTextEncoder
from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil

DEFAULT_MODEL = "qwen-image-2.1"
# Same architecture as the default model. The Turbo checkpoint brings its own schedule.
FAMILY_MODELS = ("qwen-image-2.1-turbo",)
IGNORED_OPTIONS = {"--lora-style": "Named LoRA styles are only supported by the Flux in-context CLI; use --lora."}
CONDITIONAL_OPTIONS = {
    **qwen21_vae_options.CONDITIONAL_TILING,
    "--ssd-stream": {
        "condition": "a bf16 or pre-quantized checkpoint (the original repo or an mflux-save folder), without -q, MFLUX_QWEN21_W8A8 or --bake-lora with a LoRA",
        "reason": "the other combinations exit with an error before a weight is read. With --ssd-stream a LoRA stays a runtime layer unless --bake-lora is given.",
    },
}


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image using Qwen Image 2.1 model.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False, default_model=DEFAULT_MODEL)
    parser.add_compute_precision_arguments()
    parser.add_small_text_encoder_arguments(Qwen21SmallTextEncoder.CLI_HELP)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_image_to_image_arguments(required=False)
    parser.add_vae_full_precision_arguments(help=qwen21_vae_options.FULL_PRECISION_HELP)
    parser.add_output_arguments()
    parser.add_step_cache_arguments()
    parser.add_ssd_stream_arguments()
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    CommandLineParser.warn_ignored_options(IGNORED_OPTIONS)
    qwen21_vae_options.warn_encode_only_tiling()
    ViggleTurboScheduler.check_args(parser, args)

    model_config = ConfigResolution.resolve_restricted(
        args.model,
        DEFAULT_MODEL,
        model_path=args.model_path,
        base_model=args.base_model,
        extra_keys=FAMILY_MODELS,
    )
    Qwen21TurboScheduler.check_args(parser, args, model_config)

    lora_kwargs = lora_init_kwargs_from_args(args)
    if args.ssd_stream and not CommandLineParser._option_was_provided("--bake-lora", "--no-bake-lora"):
        # The streamer installs the saved weights over the layers and would drop a baked LoRA: keep it a runtime layer.
        lora_kwargs["bake_lora"] = False
    qwen = QwenImage21(
        quantize=args.quantize,
        model_path=args.model_path,
        model_config=model_config,
        compute_precision=ComputePrecision.dtype_for(args.compute_precision),
        small_text_encoder=args.small_text_encoder,
        ssd_stream=args.ssd_stream,
        **lora_kwargs,
    )
    VAEDecodeOptions.apply(args, qwen)

    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=qwen,
        latent_creator=Qwen21LatentCreator,
    )

    try:
        width, height = DimensionResolver.resolve(
            width=args.width,
            height=args.height,
            reference_image_path=args.image_path,
        )

        for seed in args.seed:
            image = qwen.generate_image(
                seed=seed,
                prompt=PromptUtil.read_prompt(args),
                negative_prompt=PromptUtil.read_negative_prompt(args),
                width=width,
                height=height,
                guidance=args.guidance if args.guidance is not None else 1.0,
                scheduler=args.scheduler,
                image_path=args.image_path,
                num_inference_steps=args.steps,
                image_strength=args.image_strength,
                step_cache_ratio=args.step_cache_ratio,
            )
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
