import mlx.core as mx
import mlx.nn as nn

from mflux.models.common.lora.layer.dense_weight import dense_weight, is_fp8_linear
from mflux.models.common.lora.layer.fused_linear_lora_layer import FusedLoRALinear
from mflux.models.common.lora.layer.linear_lokr_layer import LoKrLinear
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear


class LoRASaver:
    @staticmethod
    def bake_and_strip_lora(module: nn.Module, dense_weights: dict | None = None) -> nn.Module:
        # dense_weights is the tree this module was loaded from, when the checkpoint stores it
        # unquantized and -q quantized it at load. A layer found there is folded before it is
        # quantized (see _fold_before_quantizing); every other layer is folded as it stands.
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

        def _fold_before_quantizing(base_linear, adapters: list, path: str) -> nn.Module | None:
            if dense_weights is None:
                return None
            source = LoRASaver._stored_weight(dense_weights, path)
            return LoRASaver._fold_before_quantizing(base_linear, adapters, source, path=path)

        def _bake_single(lora_layer: LoRALinear, path: str) -> nn.Module:
            folded = _fold_before_quantizing(lora_layer.linear, [lora_layer], path)
            if folded is not None:
                return folded
            return LoRASaver._bake_lora_into_linear(lora_layer.linear, lora_layer, path=path, upgraded=upgraded)

        def _bake_lokr(lokr_layer: LoKrLinear, path: str) -> nn.Module:
            folded = _fold_before_quantizing(lokr_layer.linear, [lokr_layer], path)
            if folded is not None:
                return folded
            return LoRASaver._bake_lokr_into_linear(lokr_layer.linear, lokr_layer, path=path, upgraded=upgraded)

        def _bake_fused(fused_layer: FusedLoRALinear, path: str) -> nn.Module:
            folded = _fold_before_quantizing(fused_layer.base_linear, fused_layer.loras, path)
            if folded is not None:
                return folded
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

        try:
            _walk(module, None, None, None)
        finally:
            # _walk calls itself, so these closures form a cycle that only the garbage collector
            # frees. Let go of the checkpoint here, or its dense weights outlive the load, also
            # when an adapter that does not fit stops the bake halfway.
            dense_weights = None
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
    def _stored_weight(tree, path: str) -> mx.array | None:
        current = tree
        for part in [*path.split("."), "weight"] if path else ["weight"]:
            if isinstance(current, list) and part.isdigit() and int(part) < len(current):
                current = current[int(part)]
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current if isinstance(current, mx.array) else None

    @staticmethod
    def _fold_before_quantizing(
        base_linear: nn.Module,
        adapters: list,
        source: mx.array | None,
        path: str = "",
    ) -> nn.Module | None:
        # A model that -q quantized at load still has the weights each layer was quantized
        # from. Adding the delta to those in float32 and quantizing the sum is what quantizing
        # a merged checkpoint does, and it keeps the whole adapter. Folding onto the decoded
        # grid instead rounds away whatever part of the delta is under half a step: a
        # Z-Image LoRA at scale 0.5 came out at 80% of its strength on q8 (#814).
        # Returns None when that is not this layer's case, and the caller folds onto the grid.
        if not isinstance(base_linear, nn.QuantizedLinear) or source is None:
            return None
        if not all(isinstance(adapter, (LoRALinear, LoKrLinear)) for adapter in adapters):
            return None
        if source.ndim != 2 or not mx.issubdtype(source.dtype, mx.floating):
            return None
        layout = {"group_size": base_linear.group_size, "bits": base_linear.bits, "mode": base_linear.mode}
        if source.shape[-1] % base_linear.group_size != 0:
            return None
        # The stored weight is this layer's origin only if it quantizes to the same codes, scales
        # and biases (the codes alone survive a rescale of the weight). A direct .diff patch, an
        # earlier bake or a loader that rewrites weights breaks that, and then the stored weight no
        # longer describes the layer.
        requantized = mx.quantize(source, **layout)
        stored = (base_linear.weight, base_linear.scales, getattr(base_linear, "biases", None))
        for again, current in zip(requantized, stored):
            if current is None or again.shape != current.shape:
                return None
            if not mx.array_equal(again.astype(current.dtype), current).item():
                return None
        if len(requantized) != 2 + (stored[2] is not None):
            return None

        at = f" at {path}" if path else ""
        merged = source.astype(mx.float32)
        for adapter in adapters:
            if isinstance(adapter, LoKrLinear):
                # A DoRA-scaled LoKr is a function of the weight it lands on, so each adapter
                # sees the sum of the ones before it.
                delta = adapter.scale * adapter.delta_weight(base_weight=merged)
            else:
                # The factors keep the dtype of the file (often float16 or bfloat16); multiplied in
                # that dtype, a small product rounds before it reaches the float32 sum.
                lora_a, lora_b = adapter.lora_A.astype(mx.float32), adapter.lora_B.astype(mx.float32)
                delta = adapter.scale * mx.transpose(mx.matmul(lora_a, lora_b))
            if delta.shape != merged.shape:
                raise ValueError(
                    f"LoRA shape mismatch{at}: base weight {merged.shape} vs adapter delta {delta.shape}. "
                    f"The adapter does not fit this model."
                )
            merged = merged + delta.astype(mx.float32)

        try:
            folded = LoRASaver._quantize_dense(merged, getattr(base_linear, "bias", None), **layout)
        except Exception as e:
            raise RuntimeError(f"Failed to bake a LoRA into {type(base_linear).__name__}{at}: {e}") from e
        # Quantizing a float32 weight gives float32 scales, and those would promote every
        # activation that goes through the layer. Keep the dtype the layer had.
        folded.scales = folded.scales.astype(base_linear.scales.dtype)
        if getattr(base_linear, "biases", None) is not None:
            folded.biases = folded.biases.astype(base_linear.biases.dtype)
        mx.eval(folded.parameters())
        return folded

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
