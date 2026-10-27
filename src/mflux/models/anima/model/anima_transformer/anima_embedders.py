import math

import mlx.core as mx
from mlx import nn

from mflux.models.anima.model.anima_transformer.anima_block import AnimaBlock


class AnimaPatchEmbed(nn.Module):
    # 2x2 patches of the latent plus an all-zero padding-mask channel, flattened channel-first as (c, py, px).
    def __init__(self, latent_channels: int, dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        # Index 0 stands for the reference's patchify Rearrange, so the weight keeps its key proj.1.weight.
        self.proj = [nn.Identity(), nn.Linear((latent_channels + 1) * patch_size**2, dim, bias=False)]

    def __call__(self, latents: mx.array) -> mx.array:
        batch, _, height, width = latents.shape
        p = self.patch_size
        x = mx.concatenate([latents, mx.zeros((batch, 1, height, width), dtype=latents.dtype)], axis=1)
        x = x.reshape(batch, -1, height // p, p, width // p, p).transpose(0, 2, 4, 1, 3, 5)
        return self.proj[1](x.reshape(batch, (height // p) * (width // p), -1))


class AnimaTimesteps(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def __call__(self, timesteps: mx.array) -> mx.array:
        half = self.dim // 2
        freqs = mx.exp(-math.log(10000) * mx.arange(half, dtype=mx.float32) / half)
        phase = timesteps.astype(mx.float32)[:, None] * freqs[None]
        return mx.concatenate([mx.cos(phase), mx.sin(phase)], axis=-1)


class AnimaTimestepEmbedding(nn.Module):
    # With AdaLN-LoRA the embedding passed on is the sinusoid itself; the MLP output is the shared
    # adaln_lora term that every modulation adds.
    def __init__(self, dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(dim, dim, bias=False)
        self.linear_2 = nn.Linear(dim, 3 * dim, bias=False)

    def __call__(self, sample: mx.array) -> mx.array:
        return self.linear_2(nn.silu(self.linear_1(sample)))


class AnimaFinalLayer(nn.Module):
    def __init__(self, dim: int, out_dim: int, adaln_lora_dim: int):
        super().__init__()
        self.adaln_modulation = AnimaBlock.modulation(dim, adaln_lora_dim, 2)
        self.linear = nn.Linear(dim, out_dim, bias=False)

    def __call__(self, x: mx.array, emb: mx.array, adaln_lora: mx.array) -> mx.array:
        shift, scale = AnimaBlock.modulate(self.adaln_modulation, emb, adaln_lora)
        return self.linear(AnimaBlock.norm(x) * (1 + scale) + shift)
