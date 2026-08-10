import math

import mlx.core as mx
from mlx import nn

from mflux.models.common.lora.layer.dense_weight import dense_weight


class LoKrLinear(nn.Module):
    @staticmethod
    def from_linear(
        linear: nn.Linear | nn.QuantizedLinear,
        lokr_w1: mx.array,
        lokr_w2: mx.array,
        dora_scale: mx.array | None = None,
        scale: float = 1.0,
    ):
        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            input_dims *= 32 // linear.bits

        lokr_linear = LoKrLinear(
            linear=linear,
            output_dims=output_dims,
            input_dims=input_dims,
            lokr_w1=lokr_w1,
            lokr_w2=lokr_w2,
            dora_scale=dora_scale,
            scale=scale,
        )
        return lokr_linear

    def __init__(
        self,
        linear: nn.Linear | nn.QuantizedLinear,
        output_dims: int,
        input_dims: int,
        lokr_w1: mx.array,
        lokr_w2: mx.array,
        dora_scale: mx.array | None = None,
        scale: float = 1.0,
    ):
        super().__init__()
        self.linear = linear
        self.output_dims = output_dims
        self.input_dims = input_dims
        self.lokr_w1 = lokr_w1
        self.lokr_w2 = lokr_w2
        self.dora_scale = dora_scale
        self.scale = scale

        expected_elements = self.output_dims * self.input_dims
        actual_elements = math.prod(self.lokr_w1.shape) * math.prod(self.lokr_w2.shape)
        if expected_elements != actual_elements:
            raise ValueError(
                "LoKr Kronecker product does not match target linear weight shape: "
                f"target=({self.output_dims}, {self.input_dims}), "
                f"lokr_w1={self.lokr_w1.shape}, lokr_w2={self.lokr_w2.shape}"
            )

        if self.dora_scale is not None and self.dora_scale.size != self.output_dims:
            raise ValueError(
                "LoKr dora_scale does not match target output dimension: "
                f"target_output={self.output_dims}, dora_scale={self.dora_scale.shape}"
            )

    def can_use_factorized_matmul(self) -> bool:
        return self.lokr_w1.ndim == 2 and self.lokr_w2.ndim == 2

    def delta_matmul(self, x: mx.array) -> mx.array:
        if self.dora_scale is None and self.can_use_factorized_matmul():
            return self.lokr_matmul(x)
        return mx.matmul(x, self.delta_weight().T)

    def lokr_matmul(self, x: mx.array) -> mx.array:
        w1 = self.lokr_w1
        w2 = self.lokr_w2
        in_m = w1.shape[1]
        prefix_shape = x.shape[:-1]
        in_n = x.shape[-1] // in_m

        h_in_group = x.reshape(*prefix_shape, in_m, in_n)
        hb = mx.matmul(h_in_group, mx.transpose(w2))
        h_cross = mx.swapaxes(hb, -1, -2)
        hc = mx.matmul(h_cross, mx.transpose(w1))
        hc = mx.swapaxes(hc, -1, -2)
        return hc.reshape(*prefix_shape, -1)

    def delta_weight(self, base_weight: mx.array | None = None) -> mx.array:
        delta = mx.kron(self.lokr_w1, self.lokr_w2).reshape((self.output_dims, self.input_dims))
        if self.dora_scale is None:
            return delta

        if base_weight is None:
            base_weight = self._dense_base_weight()
        merged_weight = base_weight + delta
        weight_norm = mx.linalg.norm(merged_weight, axis=1, keepdims=True) + mx.finfo(merged_weight.dtype).eps
        decomposed_weight = merged_weight * self.dora_scale.reshape((self.output_dims, 1)) / weight_norm
        return decomposed_weight - base_weight

    def _dense_base_weight(self) -> mx.array:
        return dense_weight(self.linear)

    def __call__(self, x):
        base_out = self.linear(x)
        return base_out + self.scale * self.delta_matmul(x)
