import inspect
from typing import TYPE_CHECKING

import mlx.nn as nn

from mflux.models.common.resolution.quantization_resolution import QuantizationResolution
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition

if TYPE_CHECKING:
    from mflux.models.common.weights.loading.weight_definition import WeightDefinitionType

_INFERABLE_BITS = {2, 3, 4, 5, 6, 8}
_INFERABLE_GROUP_SIZES = {32, 64, 128}


class WeightApplier:
    @staticmethod
    def _predicate_with_bits(predicate, bits: int | None):
        # Definition predicates may take (path, module) or (path, module, bits).
        # The three-argument form lets a family vary per-layer precision with the
        # requested level (e.g. Qwen protects img_mod_linear only at 4-bit).
        if predicate is None:
            return None
        try:
            parameters = inspect.signature(predicate).parameters
        except (TypeError, ValueError):
            return predicate
        # Count only what can receive `bits` positionally: a keyword-only third parameter
        # would raise TypeError on the call below, and *args reports one parameter while
        # accepting three.
        positional = [
            p
            for p in parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if len(positional) < 3:
            return predicate
        return lambda path, module: predicate(path, module, bits)

    @staticmethod
    def _nested_get(weights, path: str):
        current = weights
        for part in path.split("."):
            if isinstance(current, list):
                if not part.isdigit() or int(part) >= len(current):
                    return None
                current = current[int(part)]
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    @staticmethod
    def _stored_layer_predicate(component_weights, base_predicate):
        # Pre-structure each layer at the precision it was actually saved with.
        # The unquantized module still carries the true input dimension, so both
        # bits and group size are recovered exactly from the stored shapes
        # (packed weight is input_dims * bits / 32 wide, scales is
        # input_dims / group_size wide) with no assumption about either.
        # Uniform saves resolve to the global level as before; mixed saves
        # reconstruct correctly instead of failing shape validation on update.
        def predicate(path: str, module):
            base = base_predicate(path, module) if base_predicate else hasattr(module, "to_quantized")
            if not base:
                return False
            scales = WeightApplier._nested_get(component_weights, f"{path}.scales")
            if scales is None:
                return False  # layer was saved unquantized
            stored_weight = WeightApplier._nested_get(component_weights, f"{path}.weight")
            stored_shape = getattr(stored_weight, "shape", None)
            scales_shape = getattr(scales, "shape", None)
            module_shape = getattr(getattr(module, "weight", None), "shape", None)
            if not stored_shape or not scales_shape or not module_shape:
                return base
            input_dims = module_shape[-1]
            if input_dims == 0 or scales_shape[-1] == 0:
                return base
            bits = stored_shape[-1] * 32 / input_dims
            group_size = input_dims / scales_shape[-1]
            if bits in _INFERABLE_BITS and group_size in _INFERABLE_GROUP_SIZES:
                return {"bits": int(bits), "group_size": int(group_size)}
            # Falling back structures the layer at the REQUESTED precision while the file
            # holds another, so update() may reject it later. Say so here, where the shapes
            # that did not add up are still in hand.
            print(f"⚠️  Could not read the stored quantization of {path} (bits={bits}, group_size={group_size})")
            return base

        return predicate

    @staticmethod
    def apply_and_quantize_single(
        weights: LoadedWeights,
        model: nn.Module,
        component: ComponentDefinition,
        quantize_arg: int | None,
        quantization_predicate=None,
    ) -> int | None:
        stored_q = weights.meta_data.quantization_level
        component_weights = weights.components.get(component.name)

        if component_weights is None:
            raise ValueError(f"No weights found for component: {component.name}")

        if quantization_predicate is None:

            def quantization_predicate(path, module):
                return hasattr(module, "to_quantized")

        bits, warning = QuantizationResolution.resolve(stored=stored_q, requested=quantize_arg)

        if warning:
            print(f"⚠️  {warning}")

        quantization_predicate = WeightApplier._predicate_with_bits(quantization_predicate, bits)

        if bits is None:
            model.update(component_weights, strict=False)
        elif stored_q is None:
            model.update(component_weights, strict=False)
            if not component.skip_quantization:
                nn.quantize(model, class_predicate=quantization_predicate, bits=bits)
        else:
            if not component.skip_quantization:
                stored_predicate = WeightApplier._stored_layer_predicate(component_weights, quantization_predicate)
                nn.quantize(model, class_predicate=stored_predicate, bits=bits)
            model.update(component_weights, strict=False)

        return bits

    @staticmethod
    def apply_and_quantize(
        weights: LoadedWeights,
        models: dict[str, nn.Module],
        quantize_arg: int | None,
        weight_definition: "WeightDefinitionType",
    ) -> int | None:
        stored_q = weights.meta_data.quantization_level
        components = {c.name: c for c in weight_definition.get_components()}

        bits, warning = QuantizationResolution.resolve(stored=stored_q, requested=quantize_arg)

        if warning:
            print(f"⚠️  {warning}")

        if bits is None:
            WeightApplier._set_weights(weights, models, components)
        elif stored_q is None:
            WeightApplier._set_weights(weights, models, components)
            WeightApplier._quantize(models, bits, components, weight_definition)
        else:
            WeightApplier._quantize(models, bits, components, weight_definition, weights=weights)
            WeightApplier._set_weights(weights, models, components)

        return bits

    @staticmethod
    def _set_weights(
        weights: LoadedWeights,
        models: dict[str, nn.Module],
        components: dict | None = None,
    ) -> None:
        for name, model in models.items():
            component_weights = weights.components.get(name)
            if component_weights is not None:
                if components is not None:
                    component = components.get(name)
                    if component is not None and component.weight_subkey is not None:
                        component_weights = component_weights.get(component.weight_subkey, component_weights)
                model.update(component_weights, strict=False)

    @staticmethod
    def _quantize(
        models: dict[str, nn.Module],
        bits: int,
        components: dict,
        weight_definition: "WeightDefinitionType",
        weights: LoadedWeights | None = None,
    ) -> None:
        # Models whose dims are not divisible by 64 (e.g. Boogu's 3360 hidden size)
        # can opt into a smaller group size; defaults to MLX's 64 for every other model.
        group_size = getattr(weight_definition, "quantization_group_size", 64)
        predicate = WeightApplier._predicate_with_bits(weight_definition.quantization_predicate, bits)
        for name, model in models.items():
            component = components.get(name)
            if component and component.skip_quantization:
                continue
            model_predicate = predicate
            if weights is not None:
                component_weights = weights.components.get(name)
                if component is not None and component.weight_subkey is not None and component_weights is not None:
                    component_weights = component_weights.get(component.weight_subkey, component_weights)
                if component_weights is not None:
                    model_predicate = WeightApplier._stored_layer_predicate(component_weights, predicate)
            nn.quantize(
                model,
                group_size=group_size,
                class_predicate=model_predicate,
                bits=bits,
            )
