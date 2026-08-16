import inspect
import textwrap

from mflux.cli.parser.parsers import CommandLineParser, lora_init_kwargs_from_args
from mflux.models.boogu.variants.txt2img.boogu_image import BooguImage
from mflux.models.common.config import ModelConfig
from mflux.models.common.config.model_config import AVAILABLE_MODELS
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.ernie_image.variants.txt2img.ernie_image import ErnieImage
from mflux.models.fibo.variants.edit.fibo_edit import FIBOEdit
from mflux.models.fibo.variants.txt2img.fibo import FIBO
from mflux.models.flux.variants.controlnet.flux_controlnet import Flux1Controlnet
from mflux.models.flux.variants.txt2img.flux import Flux1
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
from mflux.models.ideogram4.variants.txt2img.ideogram4 import Ideogram4
from mflux.models.ideogram4.weights.ideogram4_weight_definition import Ideogram4WeightDefinition
from mflux.models.krea2.variants.txt2img.krea2 import Krea2
from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
from mflux.models.z_image import ZImage, ZImageTurbo, ZImageTurboControlnet
from mflux.utils.exceptions import ModelConfigError

# Which class owns each model's weights, keyed by the canonical AVAILABLE_MODELS key so the
# name the user typed — canonical key, alias, repo id, or a local checkpoint plus
# --base-model — is resolved by the registry rather than pattern-matched here.
#
# The substring chain this replaces ("qwen" in name … else Flux1) misfiled every name it
# had not been taught: lens, klein-*, and seedvr2 were saved as Flux1, fibo-edit as
# txt2img FIBO, and the Z-Image ControlNet as a plain ZImage — each one writing a
# checkpoint under the wrong architecture with no error at all.
MODEL_CLASSES: dict[str, type] = {
    "boogu-image-turbo": BooguImage,
    "dev": Flux1,
    "dev-controlnet-canny": Flux1Controlnet,
    "dev-controlnet-upscaler": Flux1Controlnet,
    "dev-depth": Flux1,
    "dev-fill": Flux1,
    "dev-fill-catvton": Flux1,
    "dev-kontext": Flux1,
    "dev-redux": Flux1,
    "ernie-image": ErnieImage,
    "ernie-image-turbo": ErnieImage,
    "fibo": FIBO,
    "fibo-edit": FIBOEdit,
    "fibo-edit-rmbg": FIBOEdit,
    "fibo-lite": FIBO,
    "flux2-klein-4b": Flux2Klein,
    "flux2-klein-9b": Flux2Klein,
    "flux2-klein-9b-kv": Flux2Klein,
    "flux2-klein-base-4b": Flux2Klein,
    "flux2-klein-base-9b": Flux2Klein,
    "ideogram-4-fp8": Ideogram4,
    "krea-2": Krea2,
    "krea-2-raw": Krea2,
    "krea-dev": Flux1,
    "qwen-image": QwenImage,
    "qwen-image-edit": QwenImageEdit,
    "schnell": Flux1,
    "schnell-controlnet-canny": Flux1Controlnet,
    "z-image": ZImage,
    "z-image-turbo": ZImageTurbo,
    "z-image-turbo-controlnet-union-2.1": ZImageTurboControlnet,
}

# Registry entries with no save path: LensImage and SeedVR2 implement no save_model(), so
# there is nothing to dispatch to. Listed rather than left out so the drift test can tell
# "not supported yet" from "someone added a model and forgot this table".
UNSUPPORTED_MODELS = {
    "lens-turbo": "LensImage has no save_model(); its transformer is a single-file checkpoint",
    "seedvr2-3b": "the SeedVR2 upscalers have no save_model()",
    "seedvr2-7b": "the SeedVR2 upscalers have no save_model()",
}


class SaveDispatch:
    @staticmethod
    def saveable_models_hint() -> str:
        # Canonical keys in registry order, matching --model's help text rather than this
        # module's alphabetical table: aliases would be noise, and a user comparing the error
        # against --help should see the same names in the same order. Wrapped because the list
        # is long enough that argparse's unwrapped stderr output is unreadable on one line.
        # break_on_hyphens must stay off: almost every key is hyphenated, and the default would
        # wrap mid-name ("z-image-turbo-controlnet-\n  union-2.1"), leaving nothing copy-pastable.
        names = ", ".join(key for key in AVAILABLE_MODELS if key in MODEL_CLASSES)
        return textwrap.fill(
            f"mflux-save can save: {names}",
            width=100,
            subsequent_indent="  ",
            break_on_hyphens=False,
            break_long_words=False,
        )

    @staticmethod
    def resolve_registry_key(parser: CommandLineParser, model: str, base_model: str | None) -> str:
        # The canonical key of the model being saved, or exit 2. Unmatched names used to fall
        # through to Flux1 and only fail later, deep in a weight load, if they failed at all.
        # Every exit here names the models that would have worked, since the failure is always
        # "that is not one of these" and the user has no other way to see the saveable subset.
        hint = SaveDispatch.saveable_models_hint()
        try:
            key = ConfigResolution.resolve_key(model, base_model=base_model)
        except ModelConfigError:
            parser.error(
                f"argument --model: cannot tell what {model!r} is based on. "
                f"Pass --base-model to name the built-in model it derives from.\n{hint}"
            )

        if key in UNSUPPORTED_MODELS:
            parser.error(f"argument --model: mflux-save cannot save {key} — {UNSUPPORTED_MODELS[key]}.\n{hint}")

        if key not in MODEL_CLASSES:
            parser.error(f"argument --model: mflux-save has no save path for {key}.\n{hint}")
        return key


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Save a quantized version of a model to disk.")  # fmt: off
    parser.add_model_arguments(path_type="save", require_model_arg=True)
    parser.add_lora_arguments()
    args = parser.parse_args()

    # 1. Determine model class based on the registry entry the model name resolves to
    key = SaveDispatch.resolve_registry_key(parser, args.model, args.base_model)
    model_class = MODEL_CLASSES[key]

    if key == "ideogram-4-fp8":
        model_config = Ideogram4WeightDefinition.resolve_inference_config(
            args.model,
            args.base_model,
            args.model_path,
        )
        model_path = None if Ideogram4WeightDefinition.is_builtin_name(args.model) else args.model_path
    else:
        model_config = ModelConfig.from_name(args.model, base_model=args.base_model)
        model_path = args.model_path

    # 2. Load, quantize and save the model
    # The Z-Image ControlNet takes LoRA paths but no bake_lora flag; passing every LoRA
    # kwarg unconditionally is a TypeError before a single weight is read. Filtering is
    # only safe for the flags a LoRA-capable class merely lacks: a class that takes no
    # adapter at all must say so rather than have the request quietly filtered away,
    # which is what baked an unmodified checkpoint for `--model boogu --lora ...`.
    accepted = inspect.signature(model_class).parameters
    if args.lora_paths and "lora_paths" not in accepted:
        parser.error(f"argument --lora: mflux-save cannot apply LoRA weights to {key}; mflux has no LoRA support for that model.")  # fmt: skip
    lora_kwargs = {name: value for name, value in lora_init_kwargs_from_args(args).items() if name in accepted}
    model = model_class(
        quantize=args.quantize,
        **lora_kwargs,
        model_path=model_path,
        model_config=model_config,
    )
    model.save_model(args.path)


if __name__ == "__main__":
    main()
