import inspect
import textwrap

from mflux.cli.parser.parsers import CommandLineParser, lora_init_kwargs_from_args
from mflux.models.anima.variants.txt2img.anima import Anima
from mflux.models.boogu.variants.txt2img.boogu_image import BooguImage
from mflux.models.common.config import ModelConfig
from mflux.models.common.config.model_config import AVAILABLE_MODELS
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.common.resolution.lora_resolution import LoraResolution
from mflux.models.common.resolution.quantization_resolution import QuantizationResolution
from mflux.models.common.weights.saving.hf_cache_alias import HfCacheAlias
from mflux.models.common.weights.saving.saved_model_info import SavedModelInfo, SaveTarget
from mflux.models.ernie_image.variants.txt2img.ernie_image import ErnieImage
from mflux.models.fibo.variants.edit.fibo_edit import FIBOEdit
from mflux.models.fibo.variants.txt2img.fibo import FIBO
from mflux.models.fibo2.variants.edit.fibo2_edit import Fibo2Edit
from mflux.models.fibo2.variants.txt2img.fibo2 import Fibo2
from mflux.models.flux.variants.controlnet.flux_controlnet import Flux1Controlnet
from mflux.models.flux.variants.redux.flux_redux import Flux1Redux
from mflux.models.flux.variants.txt2img.flux import Flux1
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
from mflux.models.flux3.variants.txt2img.flux3_klein import Flux3Klein
from mflux.models.ideogram4.variants.txt2img.ideogram4 import Ideogram4
from mflux.models.ideogram4.weights.ideogram4_weight_definition import Ideogram4WeightDefinition
from mflux.models.iris.variants.txt2img.iris import Iris
from mflux.models.krea2.variants.txt2img.krea2 import Krea2
from mflux.models.ming_image import MingImage
from mflux.models.moss_image import MOSSImage
from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
from mflux.models.qwen21.variants.controlnet.qwen_image_21_controlnet import QwenImage21Controlnet
from mflux.models.qwen21.variants.edit.qwen_image_21_edit import QwenImage21Edit
from mflux.models.seedvr2.variants.upscale.seedvr2 import SeedVR2
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
    "anima-aesthetic": Anima,
    "anima-base": Anima,
    "anima-turbo": Anima,
    "boogu-image-turbo": BooguImage,
    "dev": Flux1,
    "dev-controlnet-canny": Flux1Controlnet,
    "dev-controlnet-upscaler": Flux1Controlnet,
    "dev-depth": Flux1,
    "dev-fill": Flux1,
    "dev-fill-catvton": Flux1,
    "dev-kontext": Flux1,
    "dev-redux": Flux1Redux,
    "ernie-image": ErnieImage,
    "ernie-image-turbo": ErnieImage,
    "fibo": FIBO,
    "fibo-edit": FIBOEdit,
    "fibo-edit-rmbg": FIBOEdit,
    "fibo-lite": FIBO,
    "fibo2": Fibo2,
    "fibo2-edit-turbo": Fibo2Edit,
    "flux2-klein-4b": Flux2Klein,
    "flux2-klein-9b": Flux2Klein,
    "flux2-klein-9b-kv": Flux2Klein,
    "flux2-klein-base-4b": Flux2Klein,
    "flux2-klein-base-9b": Flux2Klein,
    "flux3-image-klein": Flux3Klein,
    "flux3-image-klein-base": Flux3Klein,
    "ideogram-4-fp8": Ideogram4,
    "iris-3b-base": Iris,
    "iris-3b": Iris,
    "krea-2": Krea2,
    "krea-2-raw": Krea2,
    "krea-dev": Flux1,
    "ming-image-design": MingImage,
    "moss-image-base": MOSSImage,
    "moss-image-sft": MOSSImage,
    "qwen-image": QwenImage,
    # One Qwen-Image-2.1 checkpoint does text-to-image and editing. QwenImage21Edit saves all of it
    # (vision tower and lm_head too), and mflux-generate-qwen-2.1 also loads that save.
    "qwen-image-2.1": QwenImage21Edit,
    "qwen-image-2.1-controlnet-union": QwenImage21Controlnet,
    "qwen-image-2.1-turbo": QwenImage21Edit,
    "qwen-image-2.1-turbo-controlnet-union": QwenImage21Controlnet,
    "qwen-image-edit": QwenImageEdit,
    "qwen-image-edit-2511": QwenImageEdit,
    "schnell": Flux1,
    "schnell-controlnet-canny": Flux1Controlnet,
    "seedvr2-3b": SeedVR2,
    "seedvr2-7b": SeedVR2,
    "z-image": ZImage,
    "z-image-turbo": ZImageTurbo,
    "z-image-turbo-controlnet-union-2.1": ZImageTurboControlnet,
}

# Registry entries with no save path: LensImage implements no save_model(), so
# there is nothing to dispatch to. Listed rather than left out so the drift test can tell
# "not supported yet" from "someone added a model and forgot this table".
UNSUPPORTED_MODELS = {
    "lens-turbo": "LensImage has no save_model(); its transformer is a single-file checkpoint",
    "krea-2-controlnet-depth": "the depth ControlNet is added at load; save krea-2 instead and pass it to mflux-generate-krea2-controlnet with --base-model krea-2-controlnet-depth",
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
    parser.add_hf_symlink_argument()
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

    # Find and check the target folder and the alias name before the load, so a problem
    # stops the command in seconds, not after the quantize.
    # The bit width the save writes: a pre-quantized local model keeps its stored width.
    quantize, _ = QuantizationResolution.resolve(
        stored=SaveTarget.stored_quantization(model_path), requested=args.quantize
    )
    try:
        target = SaveTarget.resolve(args.path, key, model_path, quantize, bool(args.lora_paths))
    except ValueError as error:
        parser.error(f"argument --path: {error}")
    if args.hf_symlink is not None:
        hf_username, hf_username_source = HfCacheAlias.resolve_username(args.hf_symlink)
        try:
            HfCacheAlias.check(target, hf_username)
        except ValueError as error:
            parser.error(f"argument --hf-symlink: {error} (name from {hf_username_source})")

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
    model.save_model(str(target))

    # 3. Record what the folder holds. Written last, so a folder without it is an incomplete save.
    lora_files = LoraResolution.resolve_paths(args.lora_paths)
    SavedModelInfo.write(
        target=target,
        key=key,
        model_path=model_path,
        model_config_repo_id=model_config.model_name,
        quantize=quantize,
        lora_sources=list(args.lora_paths or []),
        lora_files=lora_files,
        lora_scales=LoraResolution.resolve_scales(args.lora_scales, len(lora_files)),
    )
    print(f"Saved model: {target}")

    # 4. Optionally link the saved folder into the Hugging Face cache
    if args.hf_symlink is not None:
        alias = HfCacheAlias.create(target, hf_username)
        print(f"Hugging Face alias: {alias.repo_id} (name from {hf_username_source})")
        print(f"  {alias.file_count} files linked in {alias.repo_cache}")


if __name__ == "__main__":
    main()
