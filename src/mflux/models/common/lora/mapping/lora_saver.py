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

        def bake(layer: nn.Module, path: str) -> nn.Module:
            fused = isinstance(layer, FusedLoRALinear)
            base = layer.base_linear if fused else layer.linear
            adapters = layer.loras if fused else [layer]
            source = LoRASaver._stored_weight(dense_weights, path) if dense_weights is not None else None
            folded = LoRASaver._fold_before_quantizing(base, adapters, source, path=path)
            if folded is not None:
                return folded
            # Adapters are folded one at a time rather than summed: a LoKr carrying a
            # dora_scale is a non-linear function of the CURRENT base weight, so each
            # delta must see the result of the previous fold.
            current = base
            for adapter in adapters:
                if isinstance(adapter, LoRALinear):
                    current = LoRASaver._bake_lora_into_linear(current, adapter, path=path, upgraded=upgraded)
                elif isinstance(adapter, LoKrLinear):
                    current = LoRASaver._bake_lokr_into_linear(current, adapter, path=path, upgraded=upgraded)
            return current

        # The walk gets bake as an argument. A walker that closes over itself is a reference cycle
        # that only the garbage collector frees, and it kept Krea 2's 26 GB tree alive after the load.
        LoRASaver._replace_adapters(module, bake)
        if upgraded:
            print(
                f"🔧 Re-quantized {len(upgraded)} sub-8-bit layers at q8: the folded LoRA delta is below their quantization step"
            )
        return module

    @staticmethod
    def _replace_adapters(obj, bake, parent=None, attr_name=None, idx=None, path: str = "") -> None:
        # Replace wrappers first
        if isinstance(obj, (FusedLoRALinear, LoKrLinear, LoRALinear)):
            new_child = bake(obj, path)
            LoRASaver._assign(parent, attr_name, idx, new_child)
            obj = new_child

        # Recurse into containers/modules
        if isinstance(obj, list):
            for i, child in enumerate(list(obj)):
                LoRASaver._replace_adapters(child, bake, obj, None, i, LoRASaver._child_path(path, str(i)))
        elif isinstance(obj, tuple):
            temp_list = list(obj)
            for i, child in enumerate(temp_list):
                LoRASaver._replace_adapters(child, bake, temp_list, None, i, LoRASaver._child_path(path, str(i)))
            if parent is not None:
                LoRASaver._assign(parent, attr_name, idx, type(obj)(temp_list))
        elif isinstance(obj, dict):
            for key, child in list(obj.items()):
                LoRASaver._replace_adapters(child, bake, obj, key, None, LoRASaver._child_path(path, str(key)))
        elif isinstance(obj, nn.Module):
            for name, child in vars(obj).items():
                if isinstance(child, (nn.Module, list, tuple, dict)):
                    LoRASaver._replace_adapters(child, bake, obj, name, None, LoRASaver._child_path(path, name))

    @staticmethod
    def _assign(parent, attr_name, idx, new_child) -> None:
        if parent is None:
            return
        if isinstance(parent, list) and idx is not None:
            parent[idx] = new_child
        elif isinstance(parent, dict) and attr_name is not None:
            parent[attr_name] = new_child
        elif attr_name is not None:
            setattr(parent, attr_name, new_child)

    @staticmethod
    def _child_path(path: str, part: str) -> str:
        return f"{path}.{part}" if path else part

    @staticmethod
    def _bake_lora_into_linear(
        base_linear: nn.Linear | nn.QuantizedLinear,
        lora_layer: LoRALinear,
        path: str = "",
        upgraded: list[str] | None = None,
    ) -> nn.Module:
        return LoRASaver._bake_delta_into_linear(
            base_linear, LoRASaver._lora_delta(lora_layer), path=path, upgraded=upgraded
        )

    @staticmethod
    def _lora_delta(lora_layer: LoRALinear, dtype: mx.Dtype | None = None) -> mx.array:
        # The adapter's weight delta with its scale. dtype, when given, is the precision of the
        # product; otherwise it is the factors' own (the dtype of the file).
        lora_a, lora_b = lora_layer.lora_A, lora_layer.lora_B
        if dtype is not None:
            lora_a, lora_b = lora_a.astype(dtype), lora_b.astype(dtype)
        return lora_layer.scale * mx.transpose(mx.matmul(lora_a, lora_b))

    @staticmethod
    def _shape_mismatch(at: str, base_shape: tuple, delta_shape: tuple) -> ValueError:
        return ValueError(
            f"LoRA shape mismatch{at}: base weight {base_shape} vs adapter delta {delta_shape}. "
            f"The adapter does not fit this model."
        )

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
        # Imported here: the weights package imports ModelSaver, which imports this module.
        from mflux.models.common.weights.loading.weight_applier import WeightApplier

        weight = WeightApplier._nested_get(tree, LoRASaver._child_path(path, "weight"))
        return weight if isinstance(weight, mx.array) else None

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
        stored = [
            part
            for part in (base_linear.weight, base_linear.scales, getattr(base_linear, "biases", None))
            if part is not None
        ]
        if len(requantized) != len(stored):
            return None
        if any(again.shape != current.shape for again, current in zip(requantized, stored)):
            return None
        # One read back per layer: the three comparisons are evaluated together.
        matches = [mx.array_equal(again.astype(current.dtype), current) for again, current in zip(requantized, stored)]
        if not mx.all(mx.stack(matches)).item():
            return None

        at = f" at {path}" if path else ""
        merged = source.astype(mx.float32)
        for adapter in adapters:
            if isinstance(adapter, LoKrLinear):
                # A DoRA-scaled LoKr is a function of the weight it lands on, so each adapter
                # sees the sum of the ones before it. The product is taken in float32 for the
                # same reason as the LoRA factors below.
                delta = adapter.scale * adapter.delta_weight(base_weight=merged, dtype=mx.float32)
            else:
                # The factors keep the dtype of the file (often float16 or bfloat16); multiplied in
                # that dtype, a small product rounds before it reaches the float32 sum.
                delta = LoRASaver._lora_delta(adapter, dtype=mx.float32)
            if delta.shape != merged.shape:
                raise LoRASaver._shape_mismatch(at, merged.shape, delta.shape)
            merged = merged + delta.astype(mx.float32)

        try:
            folded = LoRASaver._quantize_dense(merged, getattr(base_linear, "bias", None), **layout)
        except Exception as e:
            raise RuntimeError(f"Failed to bake a LoRA into {type(base_linear).__name__}{at}: {e}") from e
        # Quantizing a float32 weight gives float32 scales, and those would promote every
        # activation that goes through the layer. Keep the dtype the layer had, and round the
        # codes against the scales and biases in that dtype: codes rounded against the float32
        # ones and decoded with the rounded ones move a whole group by up to the delta itself.
        folded.scales = folded.scales.astype(base_linear.scales.dtype)
        if getattr(base_linear, "biases", None) is not None:
            folded.biases = folded.biases.astype(base_linear.biases.dtype)
            folded.weight = LoRASaver._affine_codes(
                merged, folded.scales, folded.biases, group_size=layout["group_size"], bits=layout["bits"]
            )
        mx.eval(folded.parameters())
        return folded

    @staticmethod
    def _affine_codes(weight: mx.array, scales: mx.array, biases: mx.array, group_size: int, bits: int) -> mx.array:
        # The affine codes of `weight` rounded against the given scales and biases, packed as
        # mx.quantize packs them: one bitstream per row, lowest bits first.
        rows, cols = weight.shape
        groups = weight.astype(mx.float32).reshape(rows, cols // group_size, group_size)
        steps = scales.astype(mx.float32)[..., None]
        edges = biases.astype(mx.float32)[..., None]
        codes = mx.clip(mx.round((groups - edges) / steps), 0, (1 << bits) - 1)
        if 32 % bits == 0:
            # 2, 4 and 8 bits: whole codes per 32-bit word.
            per_word = 32 // bits
            words = codes.astype(mx.uint32).reshape(rows, cols // per_word, per_word)
            shifts = (mx.arange(per_word, dtype=mx.uint32) * bits)[None, None, :]
            return mx.sum(mx.left_shift(words, shifts), axis=-1)
        # 3, 5 and 6 bits: eight codes fill exactly `bits` bytes, so they are joined in a uint64
        # and its low bytes kept.
        joined = codes.astype(mx.uint64).reshape(rows, cols // 8, 8)
        shifts = (mx.arange(8, dtype=mx.uint64) * bits)[None, None, :]
        joined = mx.sum(mx.left_shift(joined, shifts), axis=-1)
        packed = joined.view(mx.uint8).reshape(rows, cols // 8, 8)[:, :, :bits]
        return packed.reshape(rows, cols * bits // 8).view(mx.uint32)

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
            raise LoRASaver._shape_mismatch(at, base_weight.shape, delta.shape)

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
