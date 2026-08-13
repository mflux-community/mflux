import mlx.core as mx
from mlx import nn

from mflux.models.common.config import ModelConfig
from mflux.models.common.weights.loading.safetensors_reader import SafetensorsReader

_MAX_EAGER_INIT_ELEMENTS = 1_000_000


class Fp8Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        compute_dtype: mx.Dtype = ModelConfig.precision,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        if out_features * in_features <= _MAX_EAGER_INIT_ELEMENTS:
            self.weight = mx.zeros((out_features, in_features), dtype=mx.uint8)
            self.weight_scale = mx.ones((out_features,), dtype=mx.float32)
        else:
            self.weight = mx.array([], dtype=mx.uint8)
            self.weight_scale = mx.array([], dtype=mx.float32)
        self.bias = mx.zeros((out_features,), dtype=compute_dtype) if bias else None

    def __call__(self, x: mx.array) -> mx.array:
        if self.weight.shape != (self.out_features, self.in_features):
            raise RuntimeError(
                "Fp8Linear weights have not been loaded: "
                f"expected {(self.out_features, self.in_features)}, got {self.weight.shape}"
            )
        if self.weight_scale.shape != (self.out_features,):
            raise RuntimeError(
                f"Fp8Linear scales have not been loaded: expected {(self.out_features,)}, got {self.weight_scale.shape}"
            )
        dtype = x.dtype if x.dtype in (mx.float16, mx.bfloat16, mx.float32) else self.compute_dtype
        weight = mx.from_fp8(self.weight, dtype=dtype)
        weight = weight * self.weight_scale.astype(dtype)[:, None]
        out = mx.matmul(x.astype(dtype), mx.transpose(weight))
        if self.bias is not None:
            out = out + self.bias.astype(dtype)
        return out

    def to_quantized(
        self,
        group_size: int | None = None,
        bits: int | None = None,
        mode: str = "affine",
        quantize_input: bool = False,
    ) -> nn.Module:
        """Convert this layer to an MLX QuantizedLinear, so `nn.quantize` can reach it.

        Ideogram 4 ships every weight-bearing linear as fp8, and fp8 codes cannot carry a
        quantized payload. Without this method the default predicate (`hasattr(module,
        "to_quantized")`) skips the transformers and the text encoder outright, so
        `mflux-save -q` writes a checkpoint that is still fp8 but stamped with a
        quantization level it does not have.

        Both call orders in WeightApplier have to work:

        - weights already applied (a source fp8 checkpoint being quantized for saving):
          dequantize the fp8 codes through their per-row scale and requantize to `bits`.
        - weights not yet applied (an mflux checkpoint that is *already* quantized, where
          the skeleton is quantized first and `model.update()` fills it in afterwards):
          return a correctly shaped, empty QuantizedLinear and let the update land.
        """
        if quantize_input:
            raise ValueError("Fp8Linear does not support quantize_input (affine weight-only quantization).")

        quantized = nn.QuantizedLinear(
            input_dims=self.in_features,
            output_dims=self.out_features,
            bias=self.bias is not None,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        if self.weight.shape != (self.out_features, self.in_features):
            # Skeleton path: nothing loaded yet, so there is no payload to carry over.
            return quantized

        weight = mx.from_fp8(self.weight, dtype=self.compute_dtype)
        weight = weight * self.weight_scale.astype(self.compute_dtype)[:, None]
        quantized.weight, quantized.scales, quantized.biases = mx.quantize(
            weight,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        if self.bias is not None:
            quantized.bias = self.bias
        return quantized

    @staticmethod
    def read_safetensors(path):
        return SafetensorsReader.read_file(path)

    @staticmethod
    def read_safetensors_directory(directory):
        return SafetensorsReader.read_directory(directory)
