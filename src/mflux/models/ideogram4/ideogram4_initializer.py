import json
from pathlib import Path
from typing import Any

import mlx.core as mx

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.models.ideogram4.model.ideogram4_text_encoder import Qwen3TextEncoder
from mflux.models.ideogram4.model.ideogram4_transformer import Ideogram4Config, Ideogram4Transformer
from mflux.models.ideogram4.weights import Ideogram4LoRAMapping, Ideogram4WeightDefinition
from mflux.utils.gpu_stream import with_gpu_stream


class Ideogram4Initializer:
    @staticmethod
    @with_gpu_stream
    def init(
        model,
        model_config: ModelConfig,
        quantize: int | None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ) -> None:
        path = model_path if model_path else model_config.model_name
        root_path = Ideogram4Initializer._resolve_model_path(path)
        Ideogram4Initializer._init_config(model, model_config, root_path)
        weights = Ideogram4Initializer._load_weights(root_path)
        Ideogram4Initializer._init_tokenizers(model, root_path)
        Ideogram4Initializer._init_models(model, root_path)
        Ideogram4Initializer._apply_weights(model, weights, quantize)
        del weights
        mx.eval(model)
        mx.clear_cache()
        Ideogram4Initializer._apply_lora(model, lora_paths, lora_scales)

    @staticmethod
    def _resolve_model_path(path: str) -> Path:
        root_path = PathResolution.resolve(
            path=path,
            patterns=Ideogram4WeightDefinition.get_download_patterns(),
        )
        if root_path is None:
            raise ValueError(f"No model path resolved for {path!r}")
        return Ideogram4WeightDefinition.validate_fp8_checkpoint(root_path)

    @staticmethod
    def _init_config(model, model_config: ModelConfig, model_path: Path) -> None:
        model.prompt_cache = {}
        model.model_config = model_config
        model.model_path = model_path
        model.callbacks = CallbackRegistry()
        model.tiling_config = None
        model.lora_paths = None
        model.lora_scales = None

    @staticmethod
    def _load_weights(model_path: Path) -> LoadedWeights:
        return WeightLoader.load(
            weight_definition=Ideogram4WeightDefinition,
            model_path=str(model_path),
        )

    @staticmethod
    def _init_tokenizers(model, model_path: Path) -> None:
        model.tokenizers = TokenizerLoader.load_all(
            definitions=Ideogram4WeightDefinition.get_tokenizers(),
            model_path=str(model_path),
        )

    @staticmethod
    def _init_models(model, model_path: Path) -> None:
        model.vae = Flux2VAE()
        model.conditional_transformer = Ideogram4Transformer(
            Ideogram4Initializer._transformer_config(model_path / "transformer")
        )
        model.unconditional_transformer = Ideogram4Transformer(
            Ideogram4Initializer._transformer_config(model_path / "unconditional_transformer")
        )
        model.text_encoder = Qwen3TextEncoder(**Ideogram4Initializer._text_encoder_kwargs(model_path / "text_encoder"))

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None) -> None:
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=Ideogram4WeightDefinition,
            models={
                "vae": model.vae,
                "conditional_transformer": model.conditional_transformer,
                "unconditional_transformer": model.unconditional_transformer,
                "text_encoder": model.text_encoder,
            },
        )

    @staticmethod
    def _apply_lora(model, lora_paths: list[str] | None, lora_scales: list[float] | None) -> None:
        lora_mapping = Ideogram4LoRAMapping.get_mapping()
        model.lora_paths, model.lora_scales = LoRALoader.load_and_apply_lora(
            lora_mapping=lora_mapping,
            transformer=model.conditional_transformer,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )
        if not model.lora_paths:
            return
        for lora_file, scale in zip(model.lora_paths, model.lora_scales):
            LoRALoader._apply_single_lora(
                model.unconditional_transformer,
                lora_file,
                scale,
                lora_mapping,
                role=None,
            )

    @staticmethod
    def _text_encoder_kwargs(directory: Path) -> dict[str, Any]:
        config = Ideogram4Initializer._load_json(directory / "config.json")
        text_config = config.get("text_config") if isinstance(config, dict) else None
        if not isinstance(text_config, dict):
            text_config = {}
        rope_parameters = text_config.get("rope_parameters")
        if not isinstance(rope_parameters, dict):
            rope_parameters = {}
        return {
            "vocab_size": int(text_config.get("vocab_size", 151936)),
            "hidden_size": int(text_config.get("hidden_size", 4096)),
            "num_hidden_layers": int(text_config.get("num_hidden_layers", 36)),
            "num_attention_heads": int(text_config.get("num_attention_heads", 32)),
            "num_key_value_heads": int(text_config.get("num_key_value_heads", 8)),
            "intermediate_size": int(text_config.get("intermediate_size", 12288)),
            "max_position_embeddings": int(text_config.get("max_position_embeddings", 262144)),
            "rope_theta": float(rope_parameters.get("rope_theta", text_config.get("rope_theta", 5_000_000.0))),
            "rms_norm_eps": float(text_config.get("rms_norm_eps", 1e-6)),
            "head_dim": int(text_config.get("head_dim", 128)),
        }

    @staticmethod
    def _transformer_config(directory: Path) -> Ideogram4Config:
        config = Ideogram4Initializer._load_json(directory / "config.json")
        num_heads = int(config.get("num_attention_heads", 18))
        head_dim = int(config.get("attention_head_dim", 256))
        return Ideogram4Config(
            emb_dim=num_heads * head_dim,
            num_layers=int(config.get("num_layers", 34)),
            num_heads=num_heads,
            intermediate_size=int(config.get("intermediate_size", 12288)),
            adanln_dim=int(config.get("adaln_dim", 512)),
            in_channels=int(config.get("in_channels", 128)),
            llm_features_dim=int(config.get("llm_features_dim", 53248)),
            rope_theta=int(config.get("rope_theta", 5_000_000)),
            mrope_section=tuple(config.get("mrope_section", (24, 20, 20))),
            norm_eps=float(config.get("norm_eps", 1e-5)),
        )

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
