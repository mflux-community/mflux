import json
from pathlib import Path
from typing import Any

import mlx.core as mx

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.lora.mapping.lora_saver import LoRASaver
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.models.ideogram4.model.ideogram4_text_encoder import Qwen3TextEncoder
from mflux.models.ideogram4.model.ideogram4_transformer import Ideogram4Config, Ideogram4Transformer
from mflux.models.ideogram4.weights import Ideogram4LoRAMapping, Ideogram4WeightDefinition


class Ideogram4Initializer:
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
        root_path = Ideogram4Initializer._resolve_model_path(path)
        Ideogram4Initializer._init_config(model, model_config, root_path)
        weights = Ideogram4Initializer._load_weights(root_path)
        Ideogram4Initializer._init_tokenizers(model, root_path)
        Ideogram4Initializer._init_models(model, root_path)
        Ideogram4Initializer._apply_weights(model, weights, quantize)
        del weights
        mx.eval(model)
        mx.clear_cache()
        Ideogram4Initializer._apply_lora(model, lora_paths, lora_scales, bake_lora)

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
    def _rebuild_q8_folded_layers(module, tree) -> None:
        """A native save can hold layers folded to an MLX quantized form: baking a LoRA
        over an fp8 base (LoRASaver.bake_and_strip_lora) dequantizes and requantizes the
        merged weight, and `mflux-save -q` quantizes every layer outright, so the
        checkpoint stores a packed 'weight' plus 'scales'/'biases'. Fp8Linear cannot hold
        those tensors and update(strict=False) skips them silently, failing at the first
        forward. Rebuild any such layer as QuantizedLinear before the update, so both
        mixed fp8/quantized and fully quantized checkpoints load. Original fp8 checkpoints
        carry 'weight_scale' instead of 'scales'/'biases' and are left untouched.

        Bits and group size are derived from the stored shapes rather than assumed: a
        4-bit save packs twice as many weights into each uint32 as an 8-bit one, so
        hard-coding q8 here would rebuild a q4 checkpoint at the wrong width and the
        update would skip every layer it touched."""
        from mlx import nn as _nn

        if isinstance(tree, list):
            children = list(module) if hasattr(module, "__iter__") else []
            for idx, sub in enumerate(tree):
                if idx < len(children):
                    Ideogram4Initializer._rebuild_q8_folded_layers(children[idx], sub)
            return
        if not isinstance(tree, dict):
            return
        for key, sub in tree.items():
            if not isinstance(sub, (dict, list)):
                continue
            child = getattr(module, key, None)
            if child is None and isinstance(module, dict):
                child = module.get(key)
            if child is None:
                continue
            if (
                isinstance(sub, dict)
                and "scales" in sub
                and "biases" in sub
                and "weight" in sub
                and not isinstance(child, _nn.QuantizedLinear)
            ):
                output_dims, group_size, bits = Ideogram4Initializer._infer_quantization(child, sub)
                input_dims = sub["scales"].shape[1] * group_size
                replacement = _nn.QuantizedLinear(
                    input_dims,
                    output_dims,
                    bias="bias" in sub,
                    group_size=group_size,
                    bits=bits,
                )
                if isinstance(module, dict):
                    module[key] = replacement
                else:
                    setattr(module, key, replacement)
            else:
                Ideogram4Initializer._rebuild_q8_folded_layers(child, sub)

    @staticmethod
    def _infer_quantization(child, sub) -> tuple[int, int, int]:
        """Recover (output_dims, group_size, bits) from a stored quantized layer.

        `scales` is (output_dims, num_groups) and the packed `weight` is
        (output_dims, input_dims * bits // 32), so once input_dims is known both the group
        size and the bit width follow. The layer being replaced is the authority on
        input_dims; fall back to MLX's default group size when it cannot say.
        """
        scales = sub["scales"]
        output_dims, num_groups = scales.shape[0], scales.shape[1]
        input_dims = getattr(child, "in_features", None)
        if not isinstance(input_dims, int) or input_dims <= 0:
            input_dims = num_groups * 64
        group_size = input_dims // num_groups
        bits = (sub["weight"].shape[-1] * 32) // input_dims
        return output_dims, group_size, bits

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None) -> None:
        for name in ("conditional_transformer", "unconditional_transformer"):
            tree = weights.components.get(name)
            if tree:
                Ideogram4Initializer._rebuild_q8_folded_layers(getattr(model, name), tree)
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
    def _apply_lora(
        model,
        lora_paths: list[str] | None,
        lora_scales: list[float] | None,
        bake_lora: bool,
    ) -> None:
        lora_mapping = Ideogram4LoRAMapping.get_mapping()
        model.lora_paths, model.lora_scales = LoRALoader.load_and_apply_lora(
            lora_mapping=lora_mapping,
            transformer=model.conditional_transformer,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            bake_lora=bake_lora,
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
        # load_and_apply_lora baked the conditional transformer; the unconditional one
        # is populated by hand above, so bake it here to match.
        if bake_lora:
            LoRASaver.bake_and_strip_lora(model.unconditional_transformer)
            mx.eval(model.unconditional_transformer.parameters())

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
