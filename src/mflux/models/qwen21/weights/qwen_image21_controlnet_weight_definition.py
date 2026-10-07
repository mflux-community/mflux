from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.qwen21.weights.qwen_image21_weight_definition import QwenImage21WeightDefinition


class QwenImage21ControlnetWeightDefinition:
    # The Qwen-Image-2.1 checkpoint plus the control branch. A model saved by mflux keeps the branch under
    # controlnet/; from the original repos it is a single safetensors file in the ControlNet repo's root.
    @staticmethod
    def get_controlnet_component() -> ComponentDefinition:
        return ComponentDefinition(name="controlnet", hf_subdir="controlnet", precision=ModelConfig.precision)

    @staticmethod
    def get_components() -> list[ComponentDefinition]:
        return QwenImage21WeightDefinition.get_components() + [
            QwenImage21ControlnetWeightDefinition.get_controlnet_component()
        ]

    @staticmethod
    def get_download_patterns() -> list[str]:
        return QwenImage21WeightDefinition.get_download_patterns() + ["controlnet/*.safetensors", "controlnet/*.json"]

    @staticmethod
    def get_tokenizers() -> list[TokenizerDefinition]:
        return QwenImage21WeightDefinition.get_tokenizers()

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        return QwenImage21WeightDefinition.quantization_predicate(path, module)
