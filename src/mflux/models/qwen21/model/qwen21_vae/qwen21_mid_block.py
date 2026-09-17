import mlx.core as mx
from mlx import nn

from mflux.models.qwen21.model.qwen21_vae.qwen21_attention_block import Qwen21AttentionBlock
from mflux.models.qwen21.model.qwen21_vae.qwen21_res_block import Qwen21ResBlock


class Qwen21MidBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.resnets = [
            Qwen21ResBlock(dim, dim),
            Qwen21ResBlock(dim, dim),
        ]
        self.attentions = [Qwen21AttentionBlock(dim)]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        x = self.resnets[1](x)
        return x
