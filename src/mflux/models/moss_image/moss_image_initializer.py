import mlx.core as mx

import mflux.models.moss_image.model.moss_image_scheduler  # noqa: F401 - registers the moss_image scheduler
from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.resolution.lora_resolution import LoraResolution
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.common.resolution.prequantized_resolution import PrequantizedResolution
from mflux.models.common.ssd_stream.streamer import Streamer
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.common_models.qwen3_5 import Qwen35TextModel
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.models.moss_image.model.moss_image_transformer.moss_image_transformer import MOSSImageTransformer
from mflux.models.moss_image.moss_image_streamed_lora import MOSSImageStreamedLoRA
from mflux.models.moss_image.weights.moss_image_lora_mapping import MOSSImageLoRAMapping
from mflux.models.moss_image.weights.moss_image_weight_definition import MOSSImageWeightDefinition

# MOSS-Image ships the FLUX.2 VAE with the small decoder (vae/config.json decoder_block_out_channels).
MOSS_IMAGE_DECODER_BLOCK_OUT_CHANNELS = (96, 192, 384, 384)


class MOSSImageInitializer:
    @staticmethod
    def init(
        model,
        model_config: ModelConfig,
        quantize: int | None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        bake_lora: bool = False,
        ssd_stream: bool = False,
        stream_lora: bool = False,
    ) -> None:
        MOSSImageInitializer._check_ssd_stream(ssd_stream, stream_lora, quantize, lora_paths, bake_lora)
        path = PrequantizedResolution.weights_root(
            model_config,
            model_path,
            quantize,
            has_loras=bool(lora_paths),
            bake_lora=bake_lora,
            **PrequantizedResolution.layout(MOSSImageWeightDefinition),
        )
        MOSSImageInitializer._init_config(model, model_config)
        model.ssd_stream = ssd_stream
        weights = MOSSImageInitializer._load_weights(path)
        MOSSImageInitializer._init_tokenizers(model, path)
        MOSSImageInitializer._init_models(model, model_config)
        MOSSImageInitializer._apply_weights(model, weights, quantize)
        streamed_lora = ssd_stream and stream_lora
        if not streamed_lora:
            MOSSImageInitializer._apply_lora(model, lora_paths, lora_scales, bake_lora, weights)
        # The parameters stay lazy: each one is read (and quantized) when a forward pass first needs
        # it. Evaluating them here held the text encoder, transformer and VAE at once, so --low-ram
        # could only free the text encoder after that peak (#948).
        del weights
        if ssd_stream:
            MOSSImageInitializer._apply_ssd_stream(model, path, lora_paths if streamed_lora else None, lora_scales)
        mx.clear_cache()

    @staticmethod
    def _check_ssd_stream(
        ssd_stream: bool, stream_lora: bool, quantize: int | None, lora_paths: list[str] | None, bake_lora: bool
    ) -> None:
        # Each of these would load and then give a wrong image, or ignore an input, without a message. Stop before
        # the weights are read.
        if stream_lora and not ssd_stream:
            raise ValueError("--stream-lora needs --ssd-stream: it keeps the LoRA on the SSD for the streamed blocks.")
        if not ssd_stream:
            return
        if quantize is not None:
            raise ValueError(
                "--ssd-stream cannot be combined with -q/--quantize: the streamer installs the saved weights over the "
                "layers, and layers that -q has quantized need packed weights. Stream a model that is already "
                "quantized (a pre-quantized repo, or a copy saved with `mflux-save -q N`) and do not pass -q."
            )
        if lora_paths and bake_lora:
            raise ValueError(
                "--ssd-stream cannot be combined with --bake-lora: the streamer installs the saved weights over the "
                "layers, which would drop the baked LoRA. Run the LoRA without --bake-lora."
            )
        if stream_lora and len(lora_paths or []) != 1:
            raise ValueError(
                f"--stream-lora streams one LoRA file, and {len(lora_paths or [])} were given. "
                "Pass exactly one --lora, or run without --stream-lora to load several."
            )

    @staticmethod
    def _init_config(model, model_config: ModelConfig) -> None:
        model.prompt_cache = {}
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = None
        model.lora_paths = []
        model.lora_scales = []

    @staticmethod
    def _load_weights(model_path: str) -> LoadedWeights:
        return WeightLoader.load(weight_definition=MOSSImageWeightDefinition, model_path=model_path)

    @staticmethod
    def _init_tokenizers(model, model_path: str) -> None:
        model.tokenizers = TokenizerLoader.load_all(
            definitions=MOSSImageWeightDefinition.get_tokenizers(),
            model_path=model_path,
        )

    @staticmethod
    def _init_models(model, model_config: ModelConfig) -> None:
        model.vae = Flux2VAE(decoder_block_out_channels=MOSS_IMAGE_DECODER_BLOCK_OUT_CHANNELS)
        model.transformer = MOSSImageTransformer(**model_config.transformer_overrides)
        model.text_encoder = Qwen35TextModel(**model_config.text_encoder_overrides)

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None) -> None:
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=MOSSImageWeightDefinition,
            models={
                "vae": model.vae,
                "transformer": model.transformer,
                "text_encoder": model.text_encoder,
            },
        )

    @staticmethod
    def _apply_lora(
        model,
        lora_paths: list[str] | None,
        lora_scales: list[float] | None,
        bake_lora: bool,
        weights: LoadedWeights,
    ) -> None:
        model.lora_paths, model.lora_scales = LoRALoader.load_and_apply_lora(
            lora_mapping=MOSSImageLoRAMapping.get_mapping(),
            transformer=model.transformer,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            bake_lora=bake_lora,
            dense_weights=weights.dense_component("transformer"),
        )

    @staticmethod
    def _apply_ssd_stream(model, path: str, lora_paths: list[str] | None, lora_scales: list[float] | None) -> None:
        layout = PrequantizedResolution.layout(MOSSImageWeightDefinition)
        root = PathResolution.resolve(
            path=path, patterns=layout["upstream_patterns"], saved_subdirs=layout["saved_subdirs"]
        )
        Streamer(cycle=False).wrap(model.text_encoder.layers, root / "text_encoder", "layers.")
        lora = None
        if lora_paths:
            # The check before the load allows one file, so these list the one LoRA that runs.
            model.lora_paths = LoraResolution.resolve_paths(lora_paths)
            model.lora_scales = LoraResolution.resolve_scales(lora_scales, len(model.lora_paths))
            lora = MOSSImageStreamedLoRA(model.lora_paths[0], model.lora_scales[0])
        Streamer(cycle=True).wrap(model.transformer.layers, root / "transformer", "layers.", lora)
        if lora is not None:
            lora.check_applied(len(model.transformer.layers))
        model.vae.decode_mode = Flux2VAE.DECODE_LOW_MEMORY
