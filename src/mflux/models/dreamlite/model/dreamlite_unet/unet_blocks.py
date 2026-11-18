import mlx.core as mx
from mlx import nn

from mflux.models.dreamlite.model.dreamlite_unet.resnet_block import DreamLiteResnetBlock
from mflux.models.dreamlite.model.dreamlite_unet.transformer_2d import DreamLiteTransformer2D


class DreamLiteDownsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def __call__(self, hidden: mx.array) -> mx.array:
        return self.conv(hidden)


class DreamLiteUpsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def __call__(self, hidden: mx.array) -> mx.array:
        hidden = mx.repeat(mx.repeat(hidden, 2, axis=1), 2, axis=2)
        return self.conv(hidden)


class DreamLiteDownBlock(nn.Module):
    NUM_LAYERS = 2

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        num_heads: int,
        transformer_layers: int,
        context_dim: int,
        self_attention: bool,
        downsample: bool,
    ):
        super().__init__()
        layers = range(DreamLiteDownBlock.NUM_LAYERS)
        self.resnets = [
            DreamLiteResnetBlock(in_channels if index == 0 else out_channels, out_channels, time_embed_dim)
            for index in layers
        ]
        self.attentions = [
            DreamLiteTransformer2D(out_channels, num_heads, transformer_layers, context_dim, self_attention)
            for _ in layers
        ]
        if downsample:
            self.downsamplers = [DreamLiteDownsample(out_channels)]

    def __call__(
        self,
        hidden: mx.array,
        time_embedding: mx.array,
        context: mx.array,
    ) -> tuple[mx.array, list[mx.array]]:
        skips = []
        for resnet, attention in zip(self.resnets, self.attentions):
            hidden = attention(resnet(hidden, time_embedding), context)
            skips.append(hidden)
        if "downsamplers" in self:
            hidden = self.downsamplers[0](hidden)
            skips.append(hidden)
        return hidden, skips


class DreamLiteMidBlock(nn.Module):
    def __init__(self, channels: int, time_embed_dim: int, num_heads: int, transformer_layers: int, context_dim: int):
        super().__init__()
        self.resnets = [DreamLiteResnetBlock(channels, channels, time_embed_dim) for _ in range(2)]
        self.attentions = [DreamLiteTransformer2D(channels, num_heads, transformer_layers, context_dim, True)]

    def __call__(self, hidden: mx.array, time_embedding: mx.array, context: mx.array) -> mx.array:
        hidden = self.resnets[0](hidden, time_embedding)
        hidden = self.attentions[0](hidden, context)
        return self.resnets[1](hidden, time_embedding)


class DreamLiteUpBlock(nn.Module):
    NUM_LAYERS = 3

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_channels: int,
        time_embed_dim: int,
        num_heads: int | None,
        transformer_layers: int,
        context_dim: int,
        self_attention: bool,
        upsample: bool,
    ):
        super().__init__()
        last = DreamLiteUpBlock.NUM_LAYERS - 1
        self.resnets = [
            DreamLiteResnetBlock(
                (prev_channels if index == 0 else out_channels) + (in_channels if index == last else out_channels),
                out_channels,
                time_embed_dim,
            )
            for index in range(DreamLiteUpBlock.NUM_LAYERS)
        ]
        # The last, full-resolution up block has no attention
        if num_heads is not None:
            self.attentions = [
                DreamLiteTransformer2D(out_channels, num_heads, transformer_layers, context_dim, self_attention)
                for _ in range(DreamLiteUpBlock.NUM_LAYERS)
            ]
        if upsample:
            self.upsamplers = [DreamLiteUpsample(out_channels)]

    def __call__(
        self,
        hidden: mx.array,
        skips: list[mx.array],
        time_embedding: mx.array,
        context: mx.array,
    ) -> mx.array:
        for index, resnet in enumerate(self.resnets):
            hidden = resnet(mx.concatenate([hidden, skips.pop()], axis=-1), time_embedding)
            if "attentions" in self:
                hidden = self.attentions[index](hidden, context)
        if "upsamplers" in self:
            hidden = self.upsamplers[0](hidden)
        return hidden
