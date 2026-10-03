import mlx.core as mx
from mlx import nn
from mlx.utils import tree_map


class ComputePrecision:
    # Runs only the inside of the attention and feed-forward modules in a lower precision. The residual
    # stream, AdaLN/modulation, norms between the modules, text encoder and VAE keep the model's precision,
    # because the residual stream can exceed the float16 range (FLUX.2 klein reaches ~1e5).
    CHOICES: dict[str, mx.Dtype] = {"float16": mx.float16}

    def __init__(self, dtype: mx.Dtype | None = None):
        if dtype is not None and dtype not in ComputePrecision.CHOICES.values():
            supported = ", ".join(f"mx.{name}" for name in ComputePrecision.CHOICES)
            raise ValueError(f"Unsupported compute precision {dtype}; supported: {supported} (or None to disable).")
        self.dtype = dtype
        self._limit = None if dtype is None else float(mx.finfo(dtype).max)

    @staticmethod
    def dtype_for(name: str | None) -> mx.Dtype | None:
        return None if name is None else ComputePrecision.CHOICES[name]

    def dtype_or(self, default: mx.Dtype) -> mx.Dtype:
        return default if self.dtype is None else self.dtype

    def to_compute(self, x: mx.array) -> mx.array:
        if self.dtype is None or x.dtype == self.dtype or not mx.issubdtype(x.dtype, mx.floating):
            return x
        return x.astype(self.dtype)

    def shrink(self, x: mx.array, headroom: float) -> mx.array:
        # Keeps a value that would overflow float16 in range; to_stream(..., headroom) scales it back.
        return x if self.dtype is None else x * (1.0 / headroom)

    def to_stream(self, x: mx.array, dtype: mx.Dtype, headroom: float = 1.0) -> mx.array:
        if self.dtype is None:
            return x
        # An overflow degrades locally instead of sending inf into the residual stream.
        x = mx.clip(x, -self._limit, self._limit).astype(dtype)
        return x if headroom == 1.0 else x * headroom

    def apply(self, root: nn.Module, module_types: tuple[type[nn.Module], ...]) -> None:
        # Call once the model is final (after quantization and LoRA): float activations against bfloat16
        # quantization scales would promote quantized_matmul to float32. The cast stays lazy, like the weights
        # it reads: it runs when a module is first used, so loading the model costs no extra memory and
        # the transformer is not read in before the text encoder has run.
        if self.dtype is None:
            return
        for _, module in root.named_modules():
            if isinstance(module, module_types):
                module.update(tree_map(self._cast_parameter, module.parameters()))
                module.compute_precision = self

    def _cast_parameter(self, value):
        if isinstance(value, mx.array) and mx.issubdtype(value.dtype, mx.floating) and value.dtype != self.dtype:
            return value.astype(self.dtype)
        return value
