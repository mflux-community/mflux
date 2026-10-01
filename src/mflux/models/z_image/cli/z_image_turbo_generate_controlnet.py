from __future__ import annotations

import math
from argparse import Namespace
from pathlib import Path

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.z_image.latent_creator import ZImageLatentCreator
from mflux.models.z_image.variants.controlnet.control_types import ControlSpec, ControlType
from mflux.models.z_image.variants.controlnet.z_image_turbo_controlnet import ZImageTurboControlnet
from mflux.utils.exceptions import ModelConfigError, PromptFileReadError, StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.prompt_util import PromptUtil

# Single source of truth for options this CLI accepts but cannot honour: the runtime
# warning and the mflux-capabilities dump both read it.
IGNORED_OPTIONS = {
    "--guidance": "Z-Image Turbo is guidance-distilled; guidance is forced to 0.0.",
    "--negative-prompt": "CFG is disabled on Z-Image Turbo, so the negative prompt is never encoded.",
}


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image using Z-Image Turbo + ControlNet Union.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=False)
    parser.add_union_controlnet_arguments(require_controls=True)
    parser.add_output_arguments()
    # Default to the Union ControlNet model so step-count normalization resolves to its 8 steps
    # instead of the generic fallback when --model is omitted.
    parser.set_defaults(model="z-image-controlnet")
    return parser


class ZImageTurboControlnetCommand:
    latent_creator = ZImageLatentCreator

    @staticmethod
    def validate(args: Namespace) -> ModelConfig:
        # Weight-free: the registry lookup and the --control specs are checked before the
        # expensive model construction, so a bad request fails fast. Input images are not opened.
        model_config = ConfigResolution.resolve(args.model, args.base_model)
        if not model_config.controlnet_model:
            raise ValueError(
                f"--model {args.model!r} is not a Union ControlNet model. "
                f"Use z-image-controlnet (the default) or a repo/path carrying a Union ControlNet checkpoint."
            )
        ZImageTurboControlnetCommand._parse_controls(args)
        return model_config

    @staticmethod
    def load(args: Namespace) -> ZImageTurboControlnet:
        # validate() is cheap and pure, so load() runs it again rather than trusting the caller did.
        return ZImageTurboControlnet(
            model_config=ZImageTurboControlnetCommand.validate(args),
            quantize=args.quantize,
            model_path=args.model_path,
            lora_paths=args.lora_paths,
            lora_scales=args.lora_scales,
        )

    @staticmethod
    def generate(model: ZImageTurboControlnet, args: Namespace, seed: int, prompt: str) -> GeneratedImage:
        return model.generate_image(
            seed=seed,
            prompt=prompt,
            width=args.width,
            height=args.height,
            scheduler=args.scheduler,
            num_inference_steps=args.steps,
            controlnet_strength=args.controlnet_strength,
            controls=ZImageTurboControlnetCommand._parse_controls(args),
        )

    @staticmethod
    def parse_control_spec(spec: str) -> ControlSpec:
        # Format: type:path[:strength]
        parts = spec.split(":")
        if len(parts) < 2:
            raise ValueError(f"Invalid --control spec {spec!r}. Expected format type:path[:strength].")

        type_str = parts[0]
        strength = 1.0
        path_parts = parts[1:]

        if len(parts) >= 3:
            # If the last segment parses as float, treat it as strength
            try:
                parsed = float(parts[-1])
            except ValueError:
                parsed = None
            if parsed is not None:
                if not math.isfinite(parsed):
                    raise ValueError(f"Invalid --control spec {spec!r}. Strength must be a finite number.")
                strength = parsed
                path_parts = parts[1:-1]

        image_path = ":".join(path_parts)
        if not image_path:
            raise ValueError(f"Invalid --control spec {spec!r}. Missing image path.")

        return ControlSpec(type=ControlType(type_str), image_path=Path(image_path), strength=strength)

    @staticmethod
    def _parse_controls(args: Namespace) -> list[ControlSpec]:
        return [ZImageTurboControlnetCommand.parse_control_spec(spec) for spec in args.control]


def main():
    parser = build_parser()
    args = parser.parse_args()
    CommandLineParser.warn_ignored_options(IGNORED_OPTIONS)

    try:
        ZImageTurboControlnetCommand.validate(args)
    except ModelConfigError:
        # A --model the registry cannot place keeps its traceback, as before.
        raise
    except ValueError as e:
        parser.error(str(e))

    model = ZImageTurboControlnetCommand.load(args)

    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=ZImageTurboControlnetCommand.latent_creator,
    )

    try:
        for seed in args.seed:
            image = ZImageTurboControlnetCommand.generate(model, args, seed, PromptUtil.read_prompt(args))
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
