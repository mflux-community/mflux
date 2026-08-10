import mlx.core as mx
from mlx import nn

from mflux.models.common.lora.layer.dense_weight import dense_weight
from mflux.models.common.lora.layer.linear_lokr_layer import LoKrLinear
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear


class FusedLoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear | nn.QuantizedLinear, loras: list[LoRALinear | LoKrLinear]):
        super().__init__()
        self.base_linear = base_linear
        self.loras = loras

    def __call__(self, x):
        base_out = self.base_linear(x)

        if any(isinstance(lora, LoKrLinear) and lora.dora_scale is not None for lora in self.loras):
            base_weight = self._dense_base_weight()
            current_weight = base_weight
            for lora in self.loras:
                if isinstance(lora, LoRALinear):
                    delta = mx.transpose(mx.matmul(lora.lora_A, lora.lora_B))
                    current_weight = current_weight + lora.scale * delta.astype(current_weight.dtype)
                elif isinstance(lora, LoKrLinear):
                    delta = lora.delta_weight(base_weight=current_weight)
                    current_weight = current_weight + lora.scale * delta.astype(current_weight.dtype)

            return base_out + mx.matmul(x, (current_weight - base_weight).T)

        lora_out = mx.zeros_like(base_out)
        for lora in self.loras:
            if isinstance(lora, LoRALinear):
                lora_out += lora.scale * mx.matmul(mx.matmul(x, lora.lora_A), lora.lora_B)
            elif isinstance(lora, LoKrLinear):
                lora_out += lora.scale * lora.delta_matmul(x)

        return base_out + lora_out

    def _dense_base_weight(self) -> mx.array:
        return dense_weight(self.base_linear)
