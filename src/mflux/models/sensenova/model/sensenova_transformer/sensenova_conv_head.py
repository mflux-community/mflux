import mlx.core as mx
from mlx import nn


class SenseNovaConvHead(nn.Module):
    def __init__(self, input_dim: int = 4096, hidden_dim: int = 1024, out_channels: int = 3, upscale: int = 8):
        super().__init__()
        self.upscale = upscale
        self.conv1 = nn.Conv2d(input_dim // 4, hidden_dim, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(hidden_dim // 4, out_channels * upscale**2, 3, padding=1, bias=True)

    @staticmethod
    def pixel_shuffle(x: mx.array, r: int) -> mx.array:
        # NHWC torch.nn.PixelShuffle: out[b, h*r+i, w*r+j, c] = in[b, h, w, c*r*r + i*r + j]
        batch, height, width, channels = x.shape
        x = x.reshape(batch, height, width, channels // (r * r), r, r)
        return x.transpose(0, 1, 4, 2, 5, 3).reshape(batch, height * r, width * r, channels // (r * r))

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, H/32, W/32, 4096] (NHWC) -> [B, 3, H, W]
        x = nn.gelu(self.conv1(self.pixel_shuffle(x, 2)))
        x = self.pixel_shuffle(self.conv2(self.pixel_shuffle(x, 2)), self.upscale)
        return x.transpose(0, 3, 1, 2)
