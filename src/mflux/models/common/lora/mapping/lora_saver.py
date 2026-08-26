import mlx.core as mx
import mlx.nn as nn

from mflux.models.common.lora.layer.dense_weight import dense_weight, is_fp8_linear
from mflux.models.common.lora.layer.fused_linear_lora_layer import FusedLoRALinear
from mflux.models.common.lora.layer.linear_lokr_layer import LoKrLinear
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear


class LoRASaver:
    @staticmethod
    def bake_and_strip_lora(module: nn.Module) -> nn.Module:
        upgraded: list[str] = []

        def _assign(parent, attr_name, idx, new_child):
            if parent is None:
                return
            if isinstance(parent, list) and idx is not None:
                parent[idx] = new_child
            elif isinstance(parent, dict) and attr_name is not None:
                parent[attr_name] = new_child
            elif attr_name is not None:
                setattr(parent, attr_name, new_child)

        def _child_path(path: str, part: str) -> str:
            return f"{path}.{part}" if path else part

        def _bake_single(lora_layer: LoRALinear, path: str) -> nn.Module:
            return LoRASaver._bake_lora_into_linear(lora_layer.linear, lora_layer, path=path, upgraded=upgraded)

        def _bake_lokr(lokr_layer: LoKrLinear, path: str) -> nn.Module:
            return LoRASaver._bake_lokr_into_linear(lokr_layer.linear, lokr_layer, path=path, upgraded=upgraded)

        def _bake_fused(fused_layer: FusedLoRALinear, path: str) -> nn.Module:
            # Adapters are folded one at a time rather than summed: a LoKr carrying a
            # dora_scale is a non-linear function of the CURRENT base weight, so each
            # delta must see the result of the previous fold.
            current = fused_layer.base_linear
            for lora in fused_layer.loras:
                if isinstance(lora, LoRALinear):
                    current = LoRASaver._bake_lora_into_linear(current, lora, path=path, upgraded=upgraded)
                elif isinstance(lora, LoKrLinear):
                    current = LoRASaver._bake_lokr_into_linear(current, lora, path=path, upgraded=upgraded)
            return current

        def _walk(obj, parent=None, attr_name=None, idx=None, path=""):
            # Replace wrappers first
            if isinstance(obj, FusedLoRALinear):
                new_child = _bake_fused(obj, path)
                _assign(parent, attr_name, idx, new_child)
                obj = new_child
            elif isinstance(obj, LoKrLinear):
                new_child = _bake_lokr(obj, path)
                _assign(parent, attr_name, idx, new_child)
                obj = new_child
            elif isinstance(obj, LoRALinear):
                new_child = _bake_single(obj, path)
                _assign(parent, attr_name, idx, new_child)
                obj = new_child

            # Recurse into containers/modules
            if isinstance(obj, list):
                for i, child in enumerate(list(obj)):
                    _walk(child, obj, None, i, _child_path(path, str(i)))
            elif isinstance(obj, tuple):
                temp_list = list(obj)
                for i, child in enumerate(temp_list):
                    _walk(child, temp_list, None, i, _child_path(path, str(i)))
                if parent is not None:
                    _assign(parent, attr_name, idx, type(obj)(temp_list))
            elif isinstance(obj, dict):
                for key, child in list(obj.items()):
                    _walk(child, obj, key, None, _child_path(path, str(key)))
            elif isinstance(obj, nn.Module):
                for name, child in vars(obj).items():
                    if isinstance(child, (nn.Module, list, tuple, dict)):
                        _walk(child, obj, name, None, _child_path(path, name))

        _walk(module, None, None, None)
        if upgraded:
            print(
                f"🔧 Re-quantized {len(upgraded)} sub-8-bit layers at q8: the folded LoRA delta is below their quantization step"
            )
        return module

    @staticmethod
    def _bake_lora_into_linear(
        base_linear: nn.Linear | nn.QuantizedLinear,
        lora_layer: LoRALinear,
        path: str = "",
        upgraded: list[str] | None = None,
    ) -> nn.Module:
        delta = mx.matmul(lora_layer.lora_A, lora_layer.lora_B)
        delta = mx.transpose(delta)
        delta = lora_layer.scale * delta
        return LoRASaver._bake_delta_into_linear(base_linear, delta, path=path, upgraded=upgraded)

    @staticmethod
    def _bake_lokr_into_linear(
        base_linear: nn.Linear | nn.QuantizedLinear,
        lokr_layer: LoKrLinear,
        path: str = "",
        upgraded: list[str] | None = None,
    ) -> nn.Module:
        base_weight = dense_weight(base_linear)
        delta = lokr_layer.scale * lokr_layer.delta_weight(base_weight=base_weight)
        return LoRASaver._bake_delta_into_linear(base_linear, delta, path=path, upgraded=upgraded)

    @staticmethod
    def _quantize_dense(
        merged: mx.array,
        bias: mx.array | None,
        group_size: int,
        bits: int,
        mode=None,
    ) -> nn.Module:
        dense_linear = nn.Linear(merged.shape[1], merged.shape[0], bias=bias is not None)
        dense_linear.weight = merged
        if bias is not None:
            dense_linear.bias = bias
        kwargs = {"group_size": group_size, "bits": bits}
        if mode is not None:
            kwargs["mode"] = mode
        quantized = nn.QuantizedLinear.from_linear(dense_linear, **kwargs)
        mx.eval(quantized.parameters())
        return quantized

    @staticmethod
    def _bake_delta_into_linear(
        base_linear: nn.Linear | nn.QuantizedLinear,
        delta: mx.array,
        path: str = "",
        upgraded: list[str] | None = None,
    ) -> nn.Module:
        # Every exit from here is either a merged layer or an exception: returning the
        # untouched base instead would hand back a model that silently generates without
        # the adapter, and mflux-save would write that as a "merged" checkpoint.
        at = f" at {path}" if path else ""

        if not hasattr(base_linear, "weight"):
            raise ValueError(f"Cannot bake a LoRA into {type(base_linear).__name__}{at}: the layer has no weight.")

        base_weight = dense_weight(base_linear)
        if base_weight.shape != delta.shape:
            raise ValueError(
                f"LoRA shape mismatch{at}: base weight {base_weight.shape} vs adapter delta {delta.shape}. "
                f"The adapter does not fit this model."
            )

        merged = base_weight + delta.astype(base_weight.dtype)
        bias = getattr(base_linear, "bias", None)

        try:
            if is_fp8_linear(base_linear):
                # The fp8 codes cannot carry the merged delta, so requantize to MLX q8
                # instead: group-64 affine keeps more mantissa than fp8-e4m3, so nothing is
                # lost relative to the base, and the result runs on the fused
                # quantized-matmul kernel rather than materializing the dense weight per
                # forward. Ideogram4Initializer._rebuild_q8_folded_layers handles loading
                # a checkpoint containing these folded layers.
                compute_dtype = getattr(base_linear, "compute_dtype", mx.bfloat16)
                return LoRASaver._quantize_dense(merged.astype(compute_dtype), bias, group_size=64, bits=8)

            if isinstance(base_linear, nn.QuantizedLinear):
                if base_linear.bits < 8:
                    # A rank-r LoRA delta sits far below a sub-8-bit quantization step (a
                    # q4 group step measures ~10x a typical rank-32 delta), so requantizing
                    # at the base precision rounds most of it away and hands back a model
                    # that generates as if the adapter were never applied. Fold at q8
                    # instead, the same escape the fp8 branch takes; the per-layer loader
                    # reconstructs mixed saves from the stored shapes.
                    if upgraded is not None:
                        upgraded.append(path or "?")
                    return LoRASaver._quantize_dense(merged, bias, group_size=64, bits=8)
                return LoRASaver._quantize_dense(
                    merged,
                    bias,
                    group_size=base_linear.group_size,
                    bits=base_linear.bits,
                    mode=base_linear.mode,
                )

            base_linear.weight = merged.astype(base_linear.weight.dtype)
            return base_linear
        except Exception as e:
            raise RuntimeError(f"Failed to bake a LoRA into {type(base_linear).__name__}{at}: {e}") from e
