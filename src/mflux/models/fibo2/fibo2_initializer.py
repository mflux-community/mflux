import logging
from pathlib import Path

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.vae.block_decoder import VAEBlockDecoder
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.common_models.qwen3_vl.qwen3_vl_processor import Qwen3VLProcessor
from mflux.models.fibo2.model.fibo2_text_encoder.fibo2_text_encoder import Fibo2TextEncoder
from mflux.models.fibo2.model.fibo2_transformer.fibo2_transformer import Fibo2Transformer
from mflux.models.fibo2.weights.fibo2_lora_mapping import Fibo2LoRAMapping
from mflux.models.fibo2.weights.fibo2_weight_definition import Fibo2WeightDefinition
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE

logger = logging.getLogger(__name__)


class Fibo2Initializer:
    @staticmethod
    def init(
        model,
        model_config: ModelConfig,
        quantize: int | None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        bake_lora: bool = True,
    ) -> None:
        path = model_path if model_path else model_config.model_name
        model.prompt_cache = {}
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = None
        weights = WeightLoader.load(weight_definition=Fibo2WeightDefinition, model_path=path)
        model.tokenizers = TokenizerLoader.load_all(definitions=Fibo2WeightDefinition.get_tokenizers(), model_path=path)
        # Qwen3-VL's image processor reads processor/processor_config.json, which the weights download brought
        root = PathResolution.resolve(path, Fibo2WeightDefinition.get_download_patterns())
        model.processor = Qwen3VLProcessor(root / "processor", model.tokenizers["qwen3_vl"].tokenizer)
        model.fonts_dir = Fibo2Initializer._fonts_dir(path)
        Fibo2Initializer._init_models(model)
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=Fibo2WeightDefinition,
            models={
                "vae": model.vae,
                "transformer": model.transformer,
                "text_encoder": model.text_encoder,
            },
        )
        model.lora_paths, model.lora_scales = LoRALoader.load_and_apply_lora(
            lora_mapping=Fibo2LoRAMapping.get_mapping(),
            transformer=model.transformer,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            bake_lora=bake_lora,
            dense_weights=weights.dense_component("transformer"),
        )

    @staticmethod
    def _init_models(model) -> None:
        model.vae = Flux2VAE()
        # The FLUX.2 strip decode: bf16 row strips with full-image GroupNorm statistics
        model.vae.decode_mode = VAEBlockDecoder.LOW_MEMORY
        model.transformer = Fibo2Transformer(**model.model_config.transformer_overrides)
        model.text_encoder = Fibo2TextEncoder()

    @staticmethod
    def _fonts_dir(path: str) -> Path | None:
        # Fetched on their own: a snapshot cached without fonts/ counts as complete for the weights
        fonts = PathResolution.resolve(path, Fibo2WeightDefinition.FONT_PATTERNS) / "fonts"
        if fonts.is_dir():
            return fonts
        logger.warning("FIBO 2's fonts were not found; text images draw some characters differently than training.")
        return None
