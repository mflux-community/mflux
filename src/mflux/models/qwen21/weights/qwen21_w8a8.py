"""Qwen-Image-2.1 DiT from ComfyUI int8_convrot checkpoints (Comfy-Org/Qwen-Image-2.1, qwen_image_2.1[_turbo]_int8_convrot).

MFLUX_QWEN21_W8A8=<file or directory> replaces the projections of every transformer block (to_q/k/v, to_out, gate/up/out
of the feed-forward) by W8A8Linear and loads the DiT from that file instead of the checkpoint's transformer/ folder; the
text encoder and the VAE still come from the checkpoint. The non-block layers stay bfloat16 as the file stores them.
QKV and gate/up are packed after loading, so each runs as one GEMM with one activation quantization.
"""

import json
import os
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from mflux.models.common.quantization.w8a8 import GROUP, W8A8Linear

PROJECTIONS = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "img_mlp.gate_layer",
    "img_mlp.proj",
    "img_mlp.out",
)
# The checkpoint fuses the feed-forward input projections as gate_up = [gate_layer; proj] (rows): undoing the rotation,
# the first half matches gate_layer (cos 0.99996) and the second proj; the per-row int8 weights are within 0.9% of bf16.
FUSED = {"img_mlp.gate_up": ("img_mlp.gate_layer", "img_mlp.proj")}


class Qwen21W8A8:
    """Loading the DiT from an int8_convrot file (MFLUX_QWEN21_W8A8): checks, conversion and weight loading."""

    @staticmethod
    def checkpoint() -> Path | None:
        path = os.environ.get("MFLUX_QWEN21_W8A8")
        return Path(path).expanduser() if path else None

    @staticmethod
    def download_patterns(patterns: list[str]) -> list[str]:
        """The checkpoint files to fetch: without the float DiT shards when the DiT comes from MFLUX_QWEN21_W8A8."""
        if Qwen21W8A8.checkpoint() is None:
            return patterns
        return [p for p in patterns if p != "transformer/*.safetensors"]

    @staticmethod
    def check_savable(model) -> None:
        """save_model would write int8 weights and scales as an ordinary checkpoint that cannot be reloaded."""
        if getattr(model, "_w8a8_checkpoint", None) is not None:
            raise ValueError(
                "A model loaded with MFLUX_QWEN21_W8A8 cannot be saved; its DiT is the int8_convrot file "
                f"({model._w8a8_checkpoint}). Load it without the variable to save the checkpoint."
            )

    @staticmethod
    def validate(quantize: int | None, lora_paths: list[str] | None) -> Path | None:
        """The MFLUX_QWEN21_W8A8 file, checked before anything is downloaded; None when the variable is unset."""
        path = Qwen21W8A8.checkpoint()
        if path is None:
            return None
        if quantize is not None or lora_paths:
            raise ValueError("MFLUX_QWEN21_W8A8 loads an int8 DiT; it cannot be combined with quantize or LoRAs.")
        if not path.exists():
            raise FileNotFoundError(f"MFLUX_QWEN21_W8A8 points to a missing file: {path}")
        return path

    @staticmethod
    def prepare(model, definition, quantize: int | None, lora_paths: list[str] | None):
        """Before loading: with MFLUX_QWEN21_W8A8 set, swap the block projections for W8A8Linear and return a weight
        definition without the transformer (Qwen21W8A8.load() reads it from the int8 file), plus that file; else (definition, None)."""
        path = Qwen21W8A8.validate(quantize, lora_paths)
        if path is None:
            return definition, None
        Qwen21W8A8.convert(model.transformer)
        model._w8a8_checkpoint = path

        class WithoutTransformer(definition):
            @staticmethod
            def get_components():
                return [c for c in definition.get_components() if c.name != "transformer"]

        return WithoutTransformer, path

    @staticmethod
    def convert(transformer) -> None:
        for block in transformer.transformer_blocks:
            for path in PROJECTIONS:
                *parents, name = path.split(".")
                module = block
                for part in parents:
                    module = module[int(part)] if part.isdigit() else getattr(module, part)
                W8A8Linear.replace(module, name)

    @staticmethod
    def _check_quant(weights: dict) -> None:
        quant_keys = [k for k in weights if k.endswith(".comfy_quant")]
        for key in quant_keys:
            config = json.loads(bytes(memoryview(weights.pop(key).astype(mx.uint8))).decode())
            # e.g. {"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}
            if (
                not config.get("convrot")
                or not str(config.get("format", "")).startswith("int8")
                or int(config.get("convrot_groupsize", 0)) != GROUP
            ):
                raise ValueError(f"{key}: unsupported quantization {config}")
        # W8A8Linear rotates the activations, so an int8 weight stored without the rotation would give x H W^T
        # instead of x W^T without any error: every int8 weight needs its convrot entry.
        int8 = [
            key.removesuffix(".weight")
            for key, value in weights.items()
            if key.endswith(".weight") and value.dtype == mx.int8
        ]
        missing = sorted(f"{prefix}.comfy_quant" for prefix in int8 if f"{prefix}.comfy_quant" not in quant_keys)
        if missing:
            raise ValueError(
                f"int8 weights without quantization metadata: missing {missing[:5]} ({len(missing)}); an int8_convrot "
                f'file stores e.g. {{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": {GROUP}}} there'
            )

    @staticmethod
    def _same_kind(expected: mx.Dtype, found: mx.Dtype) -> bool:
        # int8 weights must stay int8; every other parameter (scales, biases, norms) must be floating point,
        # so an integer scale such as int32 is rejected instead of being multiplied in.
        if expected == mx.int8:
            return found == mx.int8
        return mx.issubdtype(found, mx.floating)

    @staticmethod
    def _check_complete(path: Path, transformer, params: dict[str, mx.array]) -> None:
        # update(strict=False) would leave a missing parameter (such as a projection's scale) at its zero
        # initial value and would accept a wrong shape, so the file must match the transformer exactly.
        expected = dict(tree_flatten(transformer.parameters()))
        missing = sorted(set(expected) - set(params))
        unexpected = sorted(set(params) - set(expected))
        mismatched = sorted(
            k
            for k in set(expected) & set(params)
            if expected[k].shape != params[k].shape or not Qwen21W8A8._same_kind(expected[k].dtype, params[k].dtype)
        )
        if missing or unexpected or mismatched:
            raise ValueError(
                f"{path} does not match the transformer: missing {missing[:5]} ({len(missing)}), "
                f"unexpected {unexpected[:5]} ({len(unexpected)}), wrong shape or dtype {mismatched[:5]} ({len(mismatched)})"
            )

    @staticmethod
    def _read(path: Path) -> dict[str, mx.array]:
        if path.is_dir():
            files = sorted(path.glob("*.safetensors"))
            if len(files) != 1:
                raise ValueError(f"{path} must hold exactly one safetensors file")
            path = files[0]
        return mx.load(str(path))

    @staticmethod
    def load(transformer, path: Path) -> None:
        from mflux.models.qwen21.qwen21_initializer import Qwen21Initializer

        weights = Qwen21W8A8._read(path)
        # ComfyUI attention hints (Turbo: {"attention": "comfy_kitchen_int8"} on blocks 1-31) are not used here.
        for key in [k for k in weights if k.endswith(".comfy_attention.config")]:
            del weights[key]
        Qwen21W8A8._check_quant(weights)
        params = {}
        for key, value in weights.items():
            prefix, _, leaf = key.rpartition(".")
            name = "scale" if leaf == "weight_scale" else leaf
            if leaf == "weight_scale":
                value = value.reshape(-1)
            fused = next((n for n in FUSED if prefix.endswith("." + n)), None)
            if fused is None:
                params[f"{prefix}.{name}"] = value
                continue
            block, parts = prefix[: -len(fused)], FUSED[fused]
            for part, half in zip(parts, mx.split(value, len(parts))):
                params[f"{block}{part}.{name}"] = half
        params = Qwen21Initializer._normalize_transformer_weights(params)
        Qwen21W8A8._check_complete(path, transformer, params)
        transformer.update(tree_unflatten(list(params.items())), strict=False)
        for block in transformer.transformer_blocks:
            attn, mlp = block.attn, block.img_mlp
            attn._w8a8_qkv = W8A8Linear.pack((attn.to_q, attn.to_k, attn.to_v))
            mlp._w8a8_gate_proj = W8A8Linear.pack((mlp.gate_layer, mlp.proj))
