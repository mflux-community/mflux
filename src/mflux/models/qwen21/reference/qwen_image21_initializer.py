import json
from typing import TYPE_CHECKING

from mflux.models.common.config import ModelConfig
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.qwen21.qwen21_initializer import Qwen21Initializer
from mflux.models.qwen21.reference.model.qwen_image21_text_encoder.processor import QwenImage21Processor
from mflux.models.qwen21.reference.model.qwen_image21_text_encoder.text_encoder import QwenImage21TextEncoder
from mflux.models.qwen21.reference.model.qwen_image21_transformer.transformer import QwenImage21Transformer
from mflux.models.qwen21.reference.model.qwen_image21_vae.vae import QwenImage21VAE
from mflux.models.qwen21.reference.weights.qwen_image21_weight_definition import QwenImage21WeightDefinition

if TYPE_CHECKING:
    from mflux.models.qwen21.reference import QwenImage21Edit


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
    ) -> None:
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
