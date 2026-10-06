from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

import mflux.models.qwen21.model.qwen21_scheduler  # noqa: F401 — register the viggle_turbo scheduler
from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.compute_precision import ComputePrecision
from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_text_encoder import Qwen21TextEncoder
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer
from mflux.models.qwen21.model.qwen21_vae.qwen21_vae import Qwen21VAE
from mflux.models.qwen21.weights.qwen21_lora_mapping import Qwen21LoRAMapping
from mflux.models.qwen21.weights.qwen21_weight_definition import Qwen21WeightDefinition


class Qwen21Initializer:
    @staticmethod
    def init(
        model,
        quantize: int | None,
        model_path: str | None,
        model_config: ModelConfig,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        bake_lora: bool = True,
        compute_precision: mx.Dtype | None = None,
    ) -> None:
        precision = ComputePrecision(compute_precision)
        path = model_path if model_path else model_config.model_name
        Qwen21Initializer.init_config(model, model_config)
        root = PathResolution.resolve(path, Qwen21WeightDefinition.get_download_patterns())
        if root is None:
            raise ValueError("No Qwen-Image-2.1 checkpoint path was provided.")
        Qwen21Initializer._init_tokenizers(model, path)
        Qwen21Initializer._init_models(model)
        Qwen21Initializer.load_components(model, root, Qwen21WeightDefinition, quantize, validate=True)
        Qwen21Initializer.apply_lora(model, lora_paths, lora_scales, bake_lora)
        model.compute_precision = precision
        if compute_precision is not None:
            # Last, so it casts the final parameters, whatever quantization and LoRA produced.
            model.transformer.apply_compute_precision(precision)

    @staticmethod
    def load_components(model, root: Path, weight_definition, quantize: int | None, *, validate: bool = False) -> None:
        # Load and check one component at a time; the parameters stay lazy (see the end of the loop).
        model.bits = None
        for component in weight_definition.get_components():
            module = getattr(model, component.model_attr or component.name)
            weights = WeightLoader.load_single_local(component, root)
            supplied = dict(tree_flatten(weights.components[component.name]))
            if component.name == "transformer":
                supplied = Qwen21Initializer._normalize_transformer_weights(supplied)
                weights.components[component.name] = tree_unflatten(list(supplied.items()))
            elif component.name == "vae":
                supplied = Qwen21Initializer._normalize_vae_weights(supplied)
                weights.components[component.name] = tree_unflatten(list(supplied.items()))
            if validate and weights.meta_data.quantization_level is None:
                Qwen21Initializer._validate_weights(component.name, module, supplied, weight_definition)
            bits = WeightApplier.apply_and_quantize(
                weights=weights,
                models={component.name: module},
                quantize_arg=quantize,
                weight_definition=weight_definition,
            )
            if bits is not None:
                if model.bits is not None and model.bits != bits:
                    raise ValueError(
                        f"Conflicting component quantization levels: {component.name} uses {bits}-bit, "
                        f"but an earlier component uses {model.bits}-bit."
                    )
                model.bits = bits
            if validate and weights.meta_data.quantization_level is not None:
                Qwen21Initializer._validate_weights(component.name, module, supplied, weight_definition)
            # The parameters stay lazy: each one is read (and quantized) when a forward pass first
            # needs it. Evaluating every component here held all of them at once, parts a run never
            # uses included, and --low-ram could only free the text encoder after that peak (#832).
            del weights, supplied

    @staticmethod
    def apply_lora(
        model,
        lora_paths: list[str] | None,
        lora_scales: list[float] | None,
        bake_lora: bool,
    ) -> None:
        model.lora_paths, model.lora_scales = LoRALoader.load_and_apply_lora(
            lora_mapping=Qwen21LoRAMapping.get_mapping(),
            transformer=model.transformer,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            bake_lora=bake_lora,
        )

    @staticmethod
    def init_config(model, model_config: ModelConfig) -> None:
        model.prompt_cache = {}
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = None

    @staticmethod
    def _init_tokenizers(model, model_path: str) -> None:
        model.tokenizers = TokenizerLoader.load_all(
            definitions=Qwen21WeightDefinition.get_tokenizers(),
            model_path=model_path,
        )

    @staticmethod
    def _init_models(model) -> None:
        model.vae = Qwen21VAE()
        model.transformer = Qwen21Transformer()
        model.text_encoder = Qwen21TextEncoder()

    @staticmethod
    def _normalize_transformer_weights(supplied: dict[str, mx.array]) -> dict[str, mx.array]:
        normalized = {}
        for key, value in supplied.items():
            if key == "time_text_embed.time_proj.freqs" or key in {
                f"pos_embed.{table}.{axis}" for table in ("cos_tables", "sin_tables") for axis in range(3)
            }:
                continue  # Older text-only exports included these deterministic, non-learned buffers.
            # Saved mflux exports bypass HF mappings; include packed weights and quantization metadata.
            target = key.replace("modulation.1.", "modulation.layers.1.", 1) if key.startswith("modulation.1.") else key
            if target in normalized:
                raise ValueError(f"Duplicate transformer checkpoint key after normalization: {target}")
            normalized[target] = value
        return normalized

    @staticmethod
    def _normalize_vae_weights(supplied: dict[str, mx.array]) -> dict[str, mx.array]:
        # Saved mflux exports bypass HF mappings. Text VAE exports from before the shared VAE
        # use Qwen21CausalConv (".conv.") and Qwen21RMSNorm (".weight") parameter names.
        normalized = {}
        for key, value in supplied.items():
            target = key.replace(".downsampler.conv.", ".downsampler.resample.1.")
            target = target.replace(".upsampler.conv.", ".upsampler.resample.1.")
            target = target.replace(".conv.", ".")
            if target.endswith(".weight") and any(
                part in {"norm", "norm1", "norm2", "norm_out"} for part in target.split(".")[:-1]
            ):
                target = target.removesuffix(".weight") + ".gamma"
            if target in normalized:
                raise ValueError(f"Duplicate VAE checkpoint key after normalization: {target}")
            normalized[target] = value
        return normalized

    @staticmethod
    def _validate_weights(name: str, module, supplied: dict[str, mx.array], weight_definition=None) -> None:
        expected = dict(tree_flatten(module.parameters()))
        missing = {key for key in set(expected) - set(supplied) if not key.endswith(".inv_freq")}
        is_generation_head = getattr(weight_definition, "is_generation_head", None)
        if name == "text_encoder" and is_generation_head is not None:
            head = {key for key in missing if is_generation_head(key)}
            module.has_generation_head = not head
            if head:
                # Drop the unloaded head so it is neither quantized nor re-saved as random weights.
                module.lm_head = None
                module.language_model.norm = None
            missing -= head
        unexpected = set(supplied) - set(expected)
        mismatched = [key for key in expected.keys() & supplied.keys() if expected[key].shape != supplied[key].shape]
        if missing or unexpected or mismatched:
            raise ValueError(
                f"{name} checkpoint mismatch: missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}, shapes={mismatched}"
            )
