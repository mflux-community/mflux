import dataclasses
from pathlib import Path
from typing import List

from mflux.models.anima.weights.anima_weight_mapping import AnimaWeightMapping
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.tokenizer import LanguageTokenizer
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.qwen.weights.qwen_weight_mapping import QwenWeightMapping


class AnimaWeightDefinition:
    # The mflux layout (what mflux-save writes and model_path reads): transformer/, text_encoder/, vae/, tokenizer/
    # and tokenizer_2/. The upstream release is ComfyUI single files without tokenizers. The source_* methods read
    # those files, and the tokenizers come from the Qwen3 and T5 repos the text path was built on.
    REPO = "circlestone-labs/Anima"
    QWEN_TOKENIZER_REPO = "Qwen/Qwen3-0.6B-Base"
    T5_TOKENIZER_REPO = "google/t5-v1_1-xxl"
    TRANSFORMER_DIR = "split_files/diffusion_models"
    TRANSFORMER_FILES = {
        "anima-turbo": "anima-turbo-v1.1.safetensors",
        "anima-aesthetic": "anima-aesthetic-v1.1.safetensors",
        "anima-base": "anima-base-v1.0.safetensors",
    }
    TEXT_ENCODER_DIR = "split_files/text_encoders"
    TEXT_ENCODER_FILE = "qwen_3_06b_base.safetensors"
    VAE_DIR = "split_files/vae"
    VAE_FILE = "qwen_image_vae.safetensors"
    # The reference does not truncate prompts; this only bounds pathological input.
    MAX_SEQUENCE_LENGTH = 4096

    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        # In a copy of the model repository (split_files/ at the root) each component reads its ComfyUI file instead,
        # so mflux-check can check either layout. The initializer loads that layout itself, for the variant asked for.
        return [
            dataclasses.replace(
                component,
                variant_selector=lambda root, name=component.name: AnimaWeightDefinition._layout_on_disk(name, root),
            )
            for component in AnimaWeightDefinition._saved_components()
        ]

    @staticmethod
    def is_repository_copy(root: Path) -> bool:
        return (root / "split_files").is_dir()

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return AnimaWeightDefinition._tokenizers(qwen_subdir="tokenizer", t5_subdir="tokenizer_2")

    @staticmethod
    def get_download_patterns() -> List[str]:
        return [
            "vae/*",
            "transformer/*",
            "text_encoder/*",
            "tokenizer/*",
            "tokenizer_2/*",
        ]

    @staticmethod
    def source_transformer(variant: str) -> ComponentDefinition:
        return ComponentDefinition(
            name="transformer",
            hf_subdir=AnimaWeightDefinition.TRANSFORMER_DIR,
            precision=ModelConfig.precision,
            mapping_getter=AnimaWeightMapping.get_transformer_mapping,
            weight_files=[AnimaWeightDefinition.TRANSFORMER_FILES[variant]],
            key_transform=AnimaWeightMapping.transformer_key,
        )

    @staticmethod
    def source_text_encoder() -> ComponentDefinition:
        return ComponentDefinition(
            name="text_encoder",
            hf_subdir=AnimaWeightDefinition.TEXT_ENCODER_DIR,
            precision=ModelConfig.precision,
            mapping_getter=AnimaWeightMapping.get_text_encoder_mapping,
            num_layers=AnimaWeightMapping.NUM_TEXT_ENCODER_LAYERS,
            weight_files=[AnimaWeightDefinition.TEXT_ENCODER_FILE],
            skip_quantization=True,
        )

    @staticmethod
    def source_vae() -> ComponentDefinition:
        return ComponentDefinition(
            name="vae",
            hf_subdir=AnimaWeightDefinition.VAE_DIR,
            precision=ModelConfig.precision,
            mapping_getter=AnimaWeightMapping.get_vae_mapping,
            weight_files=[AnimaWeightDefinition.VAE_FILE],
            key_transform=QwenWeightMapping.vae_key_from_comfy,
        )

    @staticmethod
    def source_tokenizers() -> dict[str, List[TokenizerDefinition]]:
        # Keyed by the repo each tokenizer is read from.
        qwen, t5 = AnimaWeightDefinition._tokenizers(qwen_subdir="", t5_subdir="")
        qwen.download_patterns = ["tokenizer.json", "tokenizer_config.json"]
        t5.download_patterns = ["spiece.model", "tokenizer_config.json", "special_tokens_map.json"]
        return {AnimaWeightDefinition.QWEN_TOKENIZER_REPO: [qwen], AnimaWeightDefinition.T5_TOKENIZER_REPO: [t5]}

    # Kept dense at every -q level. The text path (the Qwen3 encoder, skipped as a component, and the LLM adapter)
    # moves the context 40% at -q 4 and garbles lettering. The timestep embedding and the AdaLN modulations set the
    # shift, scale and gate of every token; dense, they cut the -q 4 error per step by a fifth.
    DENSE_MODULES = ("llm_adapter", "t_embedder", "adaln_modulation")

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        if not hasattr(module, "to_quantized") or any(name in path for name in AnimaWeightDefinition.DENSE_MODULES):
            return False
        # The patch embedding takes 68 inputs (16 latent channels and a mask channel per 2x2 patch), which is not
        # a multiple of the group size.
        weight = getattr(module, "weight", None)
        return weight is None or weight.shape[-1] % 64 == 0

    @staticmethod
    def _saved_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(name="vae", hf_subdir="vae", precision=ModelConfig.precision),
            ComponentDefinition(name="transformer", hf_subdir="transformer", precision=ModelConfig.precision),
            # The text path stays dense at every -q level: see quantization_predicate.
            ComponentDefinition(
                name="text_encoder", hf_subdir="text_encoder", precision=ModelConfig.precision, skip_quantization=True
            ),
        ]

    @staticmethod
    def _layout_on_disk(name: str, root: Path) -> ComponentDefinition:
        if not AnimaWeightDefinition.is_repository_copy(root):
            return next(c for c in AnimaWeightDefinition._saved_components() if c.name == name)
        if name == "vae":
            return AnimaWeightDefinition.source_vae()
        if name == "text_encoder":
            return AnimaWeightDefinition.source_text_encoder()
        # For a check any variant stands for the transformer: they share one layout.
        files = AnimaWeightDefinition.TRANSFORMER_FILES
        present = [
            key for key, file in files.items() if (root / AnimaWeightDefinition.TRANSFORMER_DIR / file).is_file()
        ]
        return AnimaWeightDefinition.source_transformer(present[0] if present else "anima-turbo")

    @staticmethod
    def _tokenizers(qwen_subdir: str, t5_subdir: str) -> List[TokenizerDefinition]:
        return [
            TokenizerDefinition(
                name="qwen3",
                hf_subdir=qwen_subdir,
                tokenizer_class="AutoTokenizer",
                encoder_class=LanguageTokenizer,
                max_length=AnimaWeightDefinition.MAX_SEQUENCE_LENGTH,
                padding="do_not_pad",
                add_special_tokens=False,
                download_patterns=[f"{qwen_subdir}/*"],
            ),
            TokenizerDefinition(
                name="t5",
                hf_subdir=t5_subdir,
                tokenizer_class="T5Tokenizer",
                encoder_class=LanguageTokenizer,
                max_length=AnimaWeightDefinition.MAX_SEQUENCE_LENGTH,
                padding="do_not_pad",
                add_special_tokens=True,
                download_patterns=[f"{t5_subdir}/*"],
            ),
        ]
