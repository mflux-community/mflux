from pathlib import Path

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.z_image.model.z_image_text_encoder.text_encoder import TextEncoder
from mflux.models.z_image.model.z_image_transformer.transformer import ZImageTransformer
from mflux.models.z_image.model.z_image_vae.vae import VAE
from mflux.models.z_image.variants.controlnet.transformer_controlnet import ZImageControlNet, ZImageControlNetConfig
from mflux.models.z_image.weights.z_image_controlnet_weight_definition import ZImageControlnetWeightDefinition
from mflux.models.z_image.weights.z_image_lora_mapping import ZImageLoRAMapping
from mflux.models.z_image.weights.z_image_weight_definition import ZImageWeightDefinition


class ZImageInitializer:
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
        ZImageInitializer._init_config(model, model_config)
        weights = ZImageInitializer._load_weights(path)
        ZImageInitializer._init_tokenizers(model, path)
        ZImageInitializer._init_models(model)
        ZImageInitializer._apply_weights(model, weights, quantize)
        ZImageInitializer._apply_lora(model, lora_paths, lora_scales, bake_lora)

    @staticmethod
    def _init_config(model, model_config: ModelConfig) -> None:
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = None

    @staticmethod
    def _load_weights(model_path: str) -> LoadedWeights:
        return WeightLoader.load(
            weight_definition=ZImageWeightDefinition,
            model_path=model_path,
        )

    @staticmethod
    def _init_tokenizers(model, model_path: str) -> None:
        model.tokenizers = TokenizerLoader.load_all(
            definitions=ZImageWeightDefinition.get_tokenizers(),
            model_path=model_path,
        )

    @staticmethod
    def _init_models(model) -> None:
        model.vae = VAE()
        model.transformer = ZImageTransformer()
        model.text_encoder = TextEncoder()

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None) -> None:
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=ZImageWeightDefinition,
            models={
                "vae": model.vae,
                "transformer": model.transformer,
                "text_encoder": model.text_encoder,
            },
        )

    @staticmethod
    def init_controlnet(
        model,
        model_config: ModelConfig,
        quantize: int | None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ) -> None:
        """
        Initialize Z-Image Turbo and attach a ControlNet module loaded from `model_config.controlnet_model`.
        """
        if model_config.controlnet_model is None:
            raise ValueError("ModelConfig.controlnet_model must be set for ControlNet variants.")

        # Base init (vae/transformer/text encoder/tokenizer/loras)
        ZImageInitializer.init(
            model=model,
            model_config=model_config,
            quantize=quantize,
            model_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )

        # Load ControlNet config (best-effort) + weights. A model saved with mflux-save carries
        # its controlnet under controlnet/; honor it so offline reloads work and saved weights are
        # not silently swapped for the remote checkpoint.
        local_root = Path(model_path) if model_path is not None else None
        if local_root is not None and any((local_root / "controlnet").glob("*.safetensors")):
            controlnet_cfg = ZImageControlNetConfig.from_pretrained(str(local_root / "controlnet"))
            controlnet_component = next(
                c for c in ZImageControlnetWeightDefinition.get_components() if c.name == "controlnet"
            )
            controlnet_weights = WeightLoader.load_single_local(
                component=controlnet_component,
                root_path=local_root,
            )
        else:
            controlnet_cfg = ZImageControlNetConfig.from_pretrained(model_config.controlnet_model)
            controlnet_component = ZImageControlnetWeightDefinition.get_controlnet_component()
            controlnet_weights = WeightLoader.load_single(
                component=controlnet_component,
                repo_id=model_config.controlnet_model,
                file_pattern=ZImageInitializer._controlnet_file_pattern(model_config.controlnet_model),
            )

        model.controlnet = ZImageControlNet(config=controlnet_cfg)
        model.controlnet = ZImageControlNet.from_transformer(model.controlnet, model.transformer)

        WeightApplier.apply_and_quantize_single(
            weights=controlnet_weights,
            model=model.controlnet,
            component=controlnet_component,
            quantize_arg=quantize,
            quantization_predicate=ZImageControlnetWeightDefinition.quantization_predicate,
        )

    @staticmethod
    def _controlnet_file_pattern(repo_id: str) -> str:
        # Avoid accidentally loading multiple safetensors from repos that ship several variants (e.g. Union + 8steps + Tile).
        if repo_id == "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1":
            return "Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors"
        # Fallback: load any single safetensors. (If the repo contains multiple, this may not be deterministic.)
        return "*.safetensors"

    @staticmethod
    def _apply_lora(
        model,
        lora_paths: list[str] | None,
        lora_scales: list[float] | None,
        bake_lora: bool,
    ) -> None:
        model.lora_paths, model.lora_scales = LoRALoader.load_and_apply_lora(
            lora_mapping=ZImageLoRAMapping.get_mapping(),
            transformer=model.transformer,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            bake_lora=bake_lora,
        )
