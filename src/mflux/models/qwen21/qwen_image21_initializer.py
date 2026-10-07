import json
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten

from mflux.models.common.compute_precision import ComputePrecision
from mflux.models.common.config import ModelConfig
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.qwen21.model.qwen21_text_encoder.processor import QwenImage21Processor
from mflux.models.qwen21.model.qwen21_text_encoder.text_encoder import QwenImage21TextEncoder
from mflux.models.qwen21.model.qwen21_transformer.qwen21_attention import Qwen21Attention
from mflux.models.qwen21.model.qwen21_transformer.qwen21_feed_forward import Qwen21SwiGLUFeedForward
from mflux.models.qwen21.model.qwen21_transformer.qwen_image21_transformer import QwenImage21Transformer
from mflux.models.qwen21.model.qwen21_vae.vae import QwenImage21VAE
from mflux.models.qwen21.qwen21_initializer import Qwen21Initializer
from mflux.models.qwen21.weights.qwen_image21_weight_definition import QwenImage21WeightDefinition

if TYPE_CHECKING:
    from mflux.models.qwen21.variants.controlnet.qwen_image_21_controlnet import QwenImage21Controlnet
    from mflux.models.qwen21.variants.edit.qwen_image_21_edit import QwenImage21Edit


class QwenImage21Initializer:
    @staticmethod
    def init(
        model: "QwenImage21Edit",
        model_config: ModelConfig,
        quantize: int | None,
        model_path: str | None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        bake_lora: bool = True,
        compute_precision: mx.Dtype | None = None,
    ) -> None:
        precision = ComputePrecision(compute_precision)
        root = PathResolution.resolve(
            model_path or model_config.model_name, QwenImage21WeightDefinition.get_download_patterns()
        )
        if root is None:
            raise ValueError("No Qwen-Image-2.1 checkpoint path was provided.")
        Qwen21Initializer.init_config(model, model_config)
        model._checkpoint_path = str(root)
        missing_configs = [
            name for name in ("vae", "transformer", "text_encoder") if not (root / name / "config.json").is_file()
        ]
        if missing_configs:
            raise ValueError(
                f"Editing requires a complete checkpoint with component configs; missing: {missing_configs}. "
                "Use the original Qwen-Image-2.1 checkpoint or an export from QwenImage21Edit.save_model; "
                "text-only exports do not contain the visual encoder and processor needed for editing."
            )
        model._component_configs = {
            name: json.loads((root / name / "config.json").read_text())
            for name in ("vae", "transformer", "text_encoder")
        }
        model.tokenizers = TokenizerLoader.load_all(QwenImage21WeightDefinition.get_tokenizers(), str(root))
        model.processor = QwenImage21Processor(root / "processor", model.tokenizers["qwen21"].tokenizer)
        model.vae = QwenImage21VAE(model._component_configs["vae"])
        model.transformer = QwenImage21Transformer(model._component_configs["transformer"])
        model.text_encoder = QwenImage21TextEncoder(model._component_configs["text_encoder"])
        Qwen21Initializer.load_components(model, root, QwenImage21WeightDefinition, quantize, validate=True)
        Qwen21Initializer.apply_lora(model, lora_paths, lora_scales, bake_lora)
        model.compute_precision = precision
        if compute_precision is not None:
            # Last, so it casts the final parameters, whatever quantization and LoRA produced.
            model.transformer.apply_compute_precision(precision)

    @staticmethod
    def init_controlnet(
        model: "QwenImage21Controlnet",
        model_config: ModelConfig,
        quantize: int | None,
        model_path: str | None,
        controlnet_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        bake_lora: bool = True,
        compute_precision: mx.Dtype | None = None,
    ) -> None:
        from mflux.models.qwen21.variants.controlnet.qwen_image21_controlnet_transformer import QwenImage21ControlNet

        QwenImage21Initializer.init(
            model, model_config, quantize, model_path, lora_paths, lora_scales, bake_lora, compute_precision
        )
        model.controlnet = QwenImage21ControlNet(model._component_configs["transformer"])
        saved = Path(model._checkpoint_path) / "controlnet"
        if controlnet_path is None and any(saved.glob("*.safetensors")):
            # A model saved by mflux carries the branch under controlnet/, at the precision it was saved with.
            QwenImage21Initializer._load_saved_controlnet(model, quantize)
        else:
            QwenImage21Initializer._load_original_controlnet(model, controlnet_path or model_config.controlnet_model)
        if compute_precision is not None:
            # The same modules the base transformer casts, in the control blocks.
            model.compute_precision.apply(model.controlnet, (Qwen21Attention, Qwen21SwiGLUFeedForward))

    @staticmethod
    def _load_original_controlnet(model: "QwenImage21Controlnet", source: str | None) -> None:
        if source is None:
            raise ValueError("No ControlNet checkpoint: pass controlnet_path or a model config that names one.")
        root = PathResolution.resolve(source, ["*.safetensors"])
        files = [root] if root.is_file() else sorted(root.glob("*.safetensors"))
        if len(files) != 1:
            raise ValueError(f"Expected one ControlNet .safetensors file in {root}, found {len(files)}.")
        # The checkpoint holds the control branch only (control_img_in and the control blocks), named as the
        # module names them, so a strict load is the check that the file is this ControlNet.
        weights = [(key, value.astype(ModelConfig.precision)) for key, value in mx.load(str(files[0])).items()]
        model.controlnet.load_weights(weights, strict=True)
        if model.bits is not None:
            nn.quantize(
                model.controlnet, bits=model.bits, class_predicate=QwenImage21WeightDefinition.quantization_predicate
            )

    @staticmethod
    def _load_saved_controlnet(model: "QwenImage21Controlnet", quantize: int | None) -> None:
        from mflux.models.qwen21.weights.qwen_image21_controlnet_weight_definition import (
            QwenImage21ControlnetWeightDefinition,
        )

        component = QwenImage21ControlnetWeightDefinition.get_controlnet_component()
        weights = WeightLoader.load_single_local(component, Path(model._checkpoint_path))
        supplied = dict(tree_flatten(weights.components["controlnet"]))
        stored = weights.meta_data.quantization_level
        if stored is None:
            Qwen21Initializer._validate_weights("controlnet", model.controlnet, supplied)
        bits = WeightApplier.apply_and_quantize(
            weights=weights,
            models={"controlnet": model.controlnet},
            quantize_arg=quantize,
            weight_definition=QwenImage21ControlnetWeightDefinition,
        )
        if stored is not None:
            Qwen21Initializer._validate_weights("controlnet", model.controlnet, supplied)
        if bits != model.bits:
            raise ValueError(f"The saved ControlNet is {bits}-bit but the model beside it is {model.bits}-bit.")
