import mlx.core as mx

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.compute_precision import ComputePrecision
from mflux.models.common.config import ModelConfig
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.common.resolution.prequantized_resolution import PrequantizedResolution
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.common_models.qwen3_vl.qwen3_vl_processor import Qwen3VLProcessor
from mflux.models.dreamlite.model.dreamlite_text_encoder.dreamlite_text_encoder import DreamLiteTextEncoder
from mflux.models.dreamlite.model.dreamlite_unet.unet import DreamLiteUNet
from mflux.models.dreamlite.model.dreamlite_vae.vae import DreamLiteVAE
from mflux.models.dreamlite.weights.dreamlite_weight_definition import DreamLiteWeightDefinition


class DreamLiteInitializer:
    @staticmethod
    def init(
        model,
        model_config: ModelConfig,
        quantize: int | None,
        model_path: str | None = None,
        compute_precision: mx.Dtype | None = None,
    ) -> None:
        definition = DreamLiteWeightDefinition
        # Checked before any weight is read: a bad value fails at once
        precision = ComputePrecision(compute_precision)
        path = PrequantizedResolution.weights_root(
            model_config,
            model_path,
            quantize,
            **PrequantizedResolution.layout(definition),
        )
        model.prompt_cache = {}
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = None
        model.lora_paths = None
        model.lora_scales = None
        weights = WeightLoader.load(weight_definition=definition, model_path=path)
        model.tokenizers = TokenizerLoader.load_all(definitions=definition.get_tokenizers(), model_path=path)
        # Qwen3-VL's image processor reads processor/preprocessor_config.json, which the weights download brought
        root = PathResolution.resolve(path, definition.get_download_patterns())
        model.processor = Qwen3VLProcessor(root / "processor", model.tokenizers["qwen3_vl"].tokenizer)
        model.vae = DreamLiteVAE()
        model.unet = DreamLiteUNet()
        model.text_encoder = DreamLiteTextEncoder()
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=definition,
            models={
                "vae": model.vae,
                "unet": model.unet,
                "text_encoder": model.text_encoder,
            },
        )
        # The whole UNet runs in the compute precision, after quantization so that its scales are cast too. Its
        # group norms and separable convolutions stay within the float16 range; the VAE and the text encoder are
        # not touched.
        model.compute_precision = precision
        precision.apply(model.unet, (DreamLiteUNet,))
