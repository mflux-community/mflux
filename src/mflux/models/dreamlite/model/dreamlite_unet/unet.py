import mlx.core as mx
from mlx import nn

from mflux.models.dreamlite.model.dreamlite_unet.attention import DreamLiteRMSNorm
from mflux.models.dreamlite.model.dreamlite_unet.timestep_embedding import DreamLiteTimestepEmbedding
from mflux.models.dreamlite.model.dreamlite_unet.unet_blocks import (
    DreamLiteDownBlock,
    DreamLiteMidBlock,
    DreamLiteUpBlock,
)


class DreamLiteUNet(nn.Module):
    # The pruned SDXL-style UNet (0.39B). The first two down stages and their up twin run cross-attention only
    LATENT_CHANNELS = 4
    BLOCK_CHANNELS = (256, 512, 896)
    NUM_HEADS = (4, 8, 14)
    TRANSFORMER_LAYERS = (1, 2, 4)
    SELF_ATTENTION = (False, False, True)
    TIME_PROJ_DIM = 256
    TIME_EMBED_DIM = 1024
    TEXT_DIM = 2048
    CONTEXT_DIM = 2304
    GROUPS = 32
    EPS = 1e-5

    def __init__(
        self,
        block_channels: tuple[int, ...] = BLOCK_CHANNELS,
        num_heads: tuple[int, ...] = NUM_HEADS,
        transformer_layers: tuple[int, ...] = TRANSFORMER_LAYERS,
        time_proj_dim: int = TIME_PROJ_DIM,
        time_embed_dim: int = TIME_EMBED_DIM,
        text_dim: int = TEXT_DIM,
        context_dim: int = CONTEXT_DIM,
    ):
        super().__init__()
        cls = DreamLiteUNet
        channels, heads, layers, self_attention = block_channels, num_heads, transformer_layers, cls.SELF_ATTENTION
        time_dim = time_embed_dim
        self.time_proj_dim = time_proj_dim
        self.conv_in = nn.Conv2d(cls.LATENT_CHANNELS, channels[0], kernel_size=3, padding=1)
        self.time_embedding = DreamLiteTimestepEmbedding(time_proj_dim, time_dim)
        # The size conditioning: the sinusoidal embeddings of the image's width and height, concatenated
        self.add_embedding = DreamLiteTimestepEmbedding(2 * time_proj_dim, time_dim)
        self.encoder_hid_proj = [nn.Linear(text_dim, context_dim), DreamLiteRMSNorm(context_dim)]
        self.down_blocks = [
            DreamLiteDownBlock(
                in_channels=channels[max(index - 1, 0)],
                out_channels=channels[index],
                time_embed_dim=time_dim,
                num_heads=heads[index],
                transformer_layers=layers[index],
                context_dim=context_dim,
                self_attention=self_attention[index],
                downsample=index < len(channels) - 1,
            )
            for index in range(len(channels))
        ]
        self.mid_block = DreamLiteMidBlock(channels[-1], time_dim, heads[-1], layers[-1], context_dim)
        self.up_blocks = [
            DreamLiteUpBlock(
                in_channels=channels[max(index - 1, 0)],
                out_channels=channels[index],
                prev_channels=channels[min(index + 1, len(channels) - 1)],
                time_embed_dim=time_dim,
                num_heads=heads[index] if index > 0 else None,
                transformer_layers=layers[index],
                context_dim=context_dim,
                self_attention=self_attention[index],
                upsample=index > 0,
            )
            for index in reversed(range(len(channels)))
        ]
        self.conv_norm_out = nn.GroupNorm(cls.GROUPS, channels[0], eps=cls.EPS, pytorch_compatible=True)
        self.conv_out = nn.Conv2d(channels[0], cls.LATENT_CHANNELS, kernel_size=3, padding=1)

    def __call__(
        self,
        latents: mx.array,
        timestep: mx.array,
        prompt_embeds: mx.array,
        size: tuple[int, int],
    ) -> mx.array:
        # latents: (batch, channels, height, width); size: the (width, height) the model is conditioned on
        dtype = self.conv_in.weight.dtype
        context = self.encoder_hid_proj[1](self.encoder_hid_proj[0](prompt_embeds.astype(dtype)))
        time_embedding = self._time_embedding(timestep, size, latents.shape[0], dtype)

        hidden = self.conv_in(latents.astype(dtype).transpose(0, 2, 3, 1))
        skips = [hidden]
        for block in self.down_blocks:
            hidden, block_skips = block(hidden, time_embedding, context)
            skips.extend(block_skips)
        hidden = self.mid_block(hidden, time_embedding, context)
        for block in self.up_blocks:
            hidden = block(hidden, skips, time_embedding, context)
        hidden = self.conv_out(nn.silu(self.conv_norm_out(hidden)))
        return hidden.transpose(0, 3, 1, 2)

    def _time_embedding(self, timestep: mx.array, size: tuple[int, int], batch: int, dtype: mx.Dtype) -> mx.array:
        proj_dim = self.time_proj_dim
        timesteps = mx.broadcast_to(timestep.reshape(-1), (batch,))
        time = DreamLiteTimestepEmbedding.sinusoidal(timesteps, proj_dim).astype(dtype)
        size_ids = mx.array(size, dtype=mx.float32)
        size_proj = DreamLiteTimestepEmbedding.sinusoidal(size_ids, proj_dim).reshape(1, -1).astype(dtype)
        return self.time_embedding(time) + self.add_embedding(size_proj)
