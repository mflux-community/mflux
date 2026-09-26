from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.ming_image.model.ming_text_encoder.ling_moe_encoder import LingMoeEncoder
from mflux.models.ming_image.model.ming_text_encoder.ming_condition_encoder import MingHeads
from mflux.models.ming_image.model.ming_text_encoder.ming_connector import MingConnector
from mflux.models.ming_image.model.ming_transformer.ming_transformer import MingTransformer
from mflux.models.ming_image.model.ming_vae.ming_vae import MingVAE
from mflux.models.ming_image.weights.ming_image_weight_definition import MingImageWeightDefinition


class MingImageInitializer:
    @staticmethod
    def init(
        model,
        model_config: ModelConfig,
        quantize: int | None,
        text_encoder_quantize: int | None = None,
        model_path: str | None = None,
    ) -> None:
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = None
        model.lora_paths, model.lora_scales = None, None
        model.weights_path = model_path if model_path else model_config.model_name
        model.quantize_args = (quantize, text_encoder_quantize)
        weights = MingImageInitializer._load(model.weights_path)
        model.tokenizers = TokenizerLoader.load_all(
            definitions=MingImageWeightDefinition.get_tokenizers(),
            model_path=model.weights_path,
        )
        model.transformer = MingTransformer()
        model.vae = MingVAE()
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=MingImageWeightDefinition,
            models={"transformer": model.transformer, "vae": model.vae},
        )
        MingImageInitializer._apply_text_side(model, weights)

    @staticmethod
    def load_text_side(model) -> None:
        # Rebuilds only the text encoder, connector and heads (e.g. after they were released to
        # make room for the DiT). The loader maps weights lazily, so the DiT/VAE tensors it also
        # returns are never read.
        MingImageInitializer._apply_text_side(model, MingImageInitializer._load(model.weights_path))

    @staticmethod
    def _load(path: str) -> LoadedWeights:
        weights = WeightLoader.load(weight_definition=MingImageWeightDefinition, model_path=path)
        weights.components["text_encoder"] = LingMoeEncoder.stack_experts(weights.components["text_encoder"])
        return weights

    @staticmethod
    def _apply_text_side(model, weights: LoadedWeights) -> None:
        # The text encoder is quantized on its own so its (dominant, ~16B-parameter) MoE can go
        # lower than the DiT; a saved mixed checkpoint reloads per layer from its stored shapes.
        quantize, text_encoder_quantize = model.quantize_args
        model.text_encoder = LingMoeEncoder()
        model.connector = MingConnector()
        model.heads = MingHeads()
        WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=MingImageWeightDefinition,
            models={"connector": model.connector, "heads": model.heads},
        )
        text_encoder_component = next(c for c in MingImageWeightDefinition.get_components() if c.name == "text_encoder")
        WeightApplier.apply_and_quantize_single(
            weights=weights,
            model=model.text_encoder,
            component=text_encoder_component,
            quantize_arg=text_encoder_quantize if text_encoder_quantize is not None else quantize,
            quantization_predicate=MingImageWeightDefinition.quantization_predicate,
        )
        # Report what the experts actually hold: a saved mixed checkpoint carries one global
        # quantization level in its metadata, while each layer is restored from its shapes.
        experts = model.text_encoder.layers[-1].mlp.switch_mlp.gate_proj
        model.text_encoder_bits = getattr(experts, "bits", None)
