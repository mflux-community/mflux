import mlx.core as mx
from mlx import nn


class DreamLiteTinyBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        # The odd indices are the reference's ReLUs: they hold no weights and keep the checkpoint's conv.0/2/4 names
        self.conv = [
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        ]

    def __call__(self, hidden: mx.array) -> mx.array:
        residual = hidden
        for layer in self.conv:
            hidden = layer(hidden)
        return nn.relu(hidden + residual)
