from argparse import Namespace

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser, lora_init_kwargs_from_args
from mflux.models.common.compute_precision import ComputePrecision
from mflux.models.common.config.model_config import AVAILABLE_MODELS, ModelConfig
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.common.vae.vae_decode_options import VAEDecodeOptions
from mflux.models.flux2.flux2_initializer import Flux2Initializer
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
from mflux.models.flux2.variants import Flux2Klein
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import ModelConfigError, PromptFileReadError, StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.prompt_util import PromptUtil

# The model this CLI runs when --model is omitted. The parser needs it too, to key the
# --steps default off the right model instead of falling back to FLUX.1-dev's 25.
DEFAULT_MODEL = "flux2-klein-4b"
DEFAULT_GUIDANCE = 1.0
# The only guidance distilled FLUX.2 checkpoints accept; deliberately separate from DEFAULT_GUIDANCE.
DISTILLED_GUIDANCE = 1.0

# The other klein entries this one command equally serves, read off the registry so a
# new flux2- entry is accepted without touching this file.
FAMILY_MODELS = tuple(key for key in AVAILABLE_MODELS if key.startswith("flux2-") and key != DEFAULT_MODEL)

REJECTED_OPTIONS = {
    "--negative-prompt": "FLUX.2 has no negative branch; the CLI exits with an error when this is set.",
}

CONDITIONAL_OPTIONS = {
    **VAEDecodeOptions.conditional_options(),
    "--small-text-encoder": {
        "condition": "FLUX.2-klein-4B and klein-base-4B",
        "reason": "the adapter only fits the 4B conditioning width; any other model exits with an error.",
    },
    "--guidance": {
        "condition": "base (non-distilled) FLUX.2 checkpoints",
        "reason": "distilled checkpoints require guidance 1.0; any other value exits with an error.",
    },
    "--small-decoder": {
        "condition": "runs without --pid-decode",
        "reason": "PiD replaces the VAE decoder; setting both flags exits with an error.",
    },
}


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image using Flux2 Klein.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False, default_model=DEFAULT_MODEL)
    parser.add_compute_precision_arguments()
    parser.add_small_text_encoder_arguments()
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_image_to_image_arguments(required=False)
    parser.add_pid_decode_arguments()
    parser.add_small_decoder_arguments()
    parser.add_output_arguments()
    parser.add_vae_full_precision_arguments()
    return parser


class Flux2Command:
    latent_creator = Flux2LatentCreator

    @staticmethod
    def validate(args: Namespace) -> ModelConfig:
        # Weight-free. A refused flag raises ValueError; a --model outside the family raises
        # ModelConfigError. Checked in this order, so a request wrong on several reports the first.
        if args.small_decoder and args.pid_decode:
            raise ValueError("--small-decoder cannot be used with --pid-decode: PiD replaces the VAE decoder.")

        # One command serves the whole klein family; anything outside it (a FLUX.1 name, a
        # qwen alias) errors instead of silently loading a foreign config into this pipeline.
        model_config = ConfigResolution.resolve_restricted(
            args.model,
            DEFAULT_MODEL,
            model_path=args.model_path,
            extra_keys=FAMILY_MODELS,
            base_model=args.base_model,
        )
        Flux2Command._check_guidance(args, model_config)

        if args.small_text_encoder and not Flux2Initializer.supports_small_text_encoder(model_config):
            raise ValueError("--small-text-encoder only works with FLUX.2-klein-4B / klein-base-4B.")
        return model_config

    @staticmethod
    def load(args: Namespace) -> Flux2Klein:
        model = Flux2Klein(
            model_config=Flux2Command.validate(args),
            quantize=args.quantize,
            model_path=args.model_path,
            compute_precision=ComputePrecision.dtype_for(args.compute_precision),
            small_text_encoder=args.small_text_encoder,
            small_decoder=args.small_decoder,
            **lora_init_kwargs_from_args(args),
        )
        VAEDecodeOptions.apply(args, model)
        return model

    @staticmethod
    def generate(model: Flux2Klein, args: Namespace, seed: int, prompt: str) -> GeneratedImage:
        # ScaleFactor dims ("2x", and the "auto" default) need the reference image; resolved
        # here so a caller of generate() gets them without going through main().
        width, height = DimensionResolver.resolve(
            width=args.width,
            height=args.height,
            reference_image_path=args.image_path,
        )
        return model.generate_image(
            seed=seed,
            prompt=prompt,
            width=width,
            height=height,
            guidance=Flux2Command._guidance(args),
            image_path=args.image_path,
            num_inference_steps=args.steps,
            image_strength=args.image_strength,
            scheduler="flow_match_euler_discrete",
            pid_decode=args.pid_decode,
            pid_degrade_sigma=args.pid_degrade_sigma,
        )

    @staticmethod
    def _guidance(args: Namespace) -> float:
        # Read by validate() and generate(); never written back into args.
        return DEFAULT_GUIDANCE if args.guidance is None else args.guidance

    @staticmethod
    def _check_guidance(args: Namespace, model_config: ModelConfig) -> None:
        # Judged whenever the resolved config reflects the loaded weights: a builtin name always does; so does a custom
        # checkpoint (model_path set) once --base-model declared one of the family's entries, or once its name selected
        # a sibling (a name trusted for the geometry is trusted for the distillation too; a klein-base fine-tune with a
        # misleading name declares itself with --base-model). A name that matches nothing keeps the default entry's
        # config, whose name says nothing about the weights actually loaded, so it stays unjudged. Asked by name rather
        # than by comparing configs: a checkpoint spelled after the default entry resolves to that same object.
        named = args.model_path is not None and args.base_model is None
        named = (
            named and ConfigResolution.infer_family_member(args.model_path, DEFAULT_MODEL, FAMILY_MODELS) is not None
        )
        judged = args.model_path is None or args.base_model is not None or named
        if (
            judged
            and Flux2Command._guidance(args) != DISTILLED_GUIDANCE
            and "base" not in model_config.model_name.lower()
        ):
            raise ValueError("--guidance is only supported for FLUX.2 base models. Use --guidance 1.0, or declare a base checkpoint with --base-model.")  # fmt: off


def main():
    # 0. Parse command line arguments
    parser = build_parser()
    args = parser.parse_args()
    VAEDecodeOptions.warn_encode_only_tiling()

    # Keyed on the option, not on the value: a --config-from-metadata sidecar written by a
    # CFG model restores a negative prompt the user never typed, and rejecting the rerun for
    # it would blame them for an argument that is not on the command line. It reads sys.argv,
    # so it stays here, not in validate().
    if CommandLineParser._option_was_provided("--negative-prompt"):
        parser.error("--negative-prompt is not supported for FLUX.2. Focus on describing what you want.")

    # 1. Check the flags, then load the model
    try:
        Flux2Command.validate(args)
    except ModelConfigError:
        # A --model outside this command's family is a traceback, not a usage error.
        raise
    except ValueError as exc:
        parser.error(str(exc))
    model = Flux2Command.load(args)

    # 2. Register callbacks (stepwise image output, memory stats, battery saver)
    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=Flux2Command.latent_creator,
    )

    try:
        # Sizes are resolved once per run, before the first prompt is read. generate()
        # resolves again for direct callers, which changes nothing on plain numbers.
        args.width, args.height = DimensionResolver.resolve(
            width=args.width,
            height=args.height,
            reference_image_path=args.image_path,
        )
        for seed in args.seed:
            # 3. Generate an image for each seed value
            image = Flux2Command.generate(model, args, seed, PromptUtil.read_prompt(args))
            # 4. Save the image
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
