import warnings
from argparse import Namespace

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser, lora_init_kwargs_from_args
from mflux.models.common.compute_precision import ComputePrecision
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.common.vae.vae_decode_options import VAEDecodeOptions
from mflux.models.z_image.latent_creator import ZImageLatentCreator
from mflux.models.z_image.variants.z_image import ZImage
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.prompt_util import PromptUtil

# The model this CLI runs when --model is omitted. The parser needs it too, to key the
# --steps default off the right model instead of falling back to FLUX.1-dev's 25.
DEFAULT_MODEL = "z-image"

# This CLI also runs the distilled sibling (CONDITIONAL_OPTIONS is written for exactly
# that split); the ControlNet variant has its own command and needs a control image.
FAMILY_MODELS = ("z-image-turbo",)

# Single source of truth for CFG-dependent options: main() warns from these and the
# mflux-capabilities dump reads them. Both flags depend on what --model resolves to,
# so they are conditional, not statically ignored.
CONDITIONAL_OPTIONS = {
    **VAEDecodeOptions.conditional_options(),
    "--guidance": {
        "condition": "the resolved model supports guidance",
        "reason": "guidance-distilled Z-Image variants force guidance to 0.0.",
    },
    "--negative-prompt": {
        "condition": "the resolved model supports guidance and guidance > 1.0",
        "reason": "CFG is disabled at guidance <= 1.0 (the default is 0.0) and on guidance-distilled variants.",
    },
}


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image using Z-Image.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False, default_model=DEFAULT_MODEL)
    parser.add_compute_precision_arguments()
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_image_to_image_arguments()
    parser.add_pid_decode_arguments()
    parser.add_float32_arguments()
    parser.add_ane_arguments()
    parser.add_output_arguments()
    parser.add_vae_full_precision_arguments()
    # The command's scheduler. A parser default, so --scheduler=NAME, a --config-from-conf replay
    # and a script that parses these flags all see it the way the command line does.
    parser.set_defaults(scheduler="flow_match_euler_discrete")
    return parser


class ZImageCommand:
    latent_creator = ZImageLatentCreator

    @staticmethod
    def validate(args: Namespace) -> ModelConfig:
        # Weight-free: resolves --model against the in-memory registry only.
        return ConfigResolution.resolve_restricted(
            args.model,
            DEFAULT_MODEL,
            model_path=args.model_path,
            extra_keys=FAMILY_MODELS,
            base_model=args.base_model,
        )

    @staticmethod
    def load(args: Namespace) -> ZImage:
        model = ZImage(
            model_config=ZImageCommand.validate(args),
            quantize=args.quantize,
            model_path=args.model_path,
            float32=args.float32,
            compute_precision=ComputePrecision.dtype_for(args.compute_precision),
            ane=args.ane,
            ane_precision=args.ane_precision,
            ane_gpu_share=args.ane_gpu_share,
            ane_cache=args.ane_cache,
            dequantize_on_load=args.dequantize_on_load,
            ane_allow_m5=args.ane_allow_m5,
            **lora_init_kwargs_from_args(args),
        )
        VAEDecodeOptions.apply(args, model)
        return model

    @staticmethod
    def generate(model: ZImage, args: Namespace, seed: int, prompt: str) -> GeneratedImage:
        width, height = DimensionResolver.resolve(
            width=args.width, height=args.height, reference_image_path=args.image_path
        )
        return model.generate_image(
            seed=seed,
            prompt=prompt,
            width=width,
            height=height,
            guidance=args.guidance,
            image_path=args.image_path,
            num_inference_steps=args.steps,
            image_strength=args.image_strength,
            scheduler=args.scheduler,
            negative_prompt=args.negative_prompt,
            pid_decode=args.pid_decode,
            pid_degrade_sigma=args.pid_degrade_sigma,
        )


def main():
    parser = build_parser()
    args = parser.parse_args()
    VAEDecodeOptions.warn_encode_only_tiling()

    model_config = ZImageCommand.validate(args)

    # Warn on the EFFECTIVE behavior, not the flag value: --model may resolve to a
    # guidance-distilled variant (guidance forced to 0.0 regardless of the flag), and on
    # CFG-capable models the negative prompt is only encoded at guidance > 1.0, where an
    # omitted --guidance defaults to 0.0.
    if not model_config.supports_guidance:
        CommandLineParser.warn_ignored_options(
            {
                "--guidance": CONDITIONAL_OPTIONS["--guidance"]["reason"],
                "--negative-prompt": CONDITIONAL_OPTIONS["--negative-prompt"]["reason"],
            }
        )
    elif CommandLineParser._option_was_provided("--negative-prompt") and (
        args.guidance is None or args.guidance <= 1.0
    ):
        warnings.warn(
            f"--negative-prompt has no effect: {CONDITIONAL_OPTIONS['--negative-prompt']['reason']}"
            " Pass --guidance above 1.0 to enable it.",
            stacklevel=1,
        )

    # 1. Load the model
    model = ZImageCommand.load(args)

    # 2. Register callbacks
    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=ZImageCommand.latent_creator,
    )

    try:
        for seed in args.seed:
            # 3. Generate an image for each seed value
            image = ZImageCommand.generate(model, args, seed, PromptUtil.read_prompt(args))
            # 4. Save the image
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
