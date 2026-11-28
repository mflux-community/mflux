import mlx.core as mx
from mlx import nn


class CausalConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple = 3,
        stride: int | tuple = 1,
        padding: int | tuple = 1,
        causal_temporal: bool = True,
        use_padding_causal: bool = False,
    ):
        super().__init__()
        self.causal_temporal = causal_temporal
        self.use_padding_causal = use_padding_causal

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        kt, kh, kw = kernel_size
        self.weight = mx.zeros((out_channels, kt, kh, kw, in_channels))
        self.bias = mx.zeros((out_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        _, _, T, _, _ = x.shape
        kt, _, _ = self.kernel_size
        pt, ph, pw = self.padding
        _, sh, sw = self.stride

        if T == 1 and self._can_collapse_temporal(kt, pt):
            # Single-frame causal conv repeats x[:, :, 0] across all kt taps,
            # which is algebraically identical to a 2D conv with the temporal taps summed.
            w2d = (
                self.weight[:, 0]
                if kt == 1
                else mx.sum(self.weight.astype(mx.float32), axis=1).astype(mx.result_type(x, self.weight))
            )
            x2d = x[:, :, 0, :, :].transpose(0, 2, 3, 1)
            out = mx.conv2d(x2d, w2d, stride=(sh, sw), padding=(ph, pw)) + self.bias
            return out.transpose(0, 3, 1, 2)[:, :, None, :, :]

        if self.causal_temporal and kt > 1:
            causal_pad = (2 * pt) if self.use_padding_causal else kt - 1
            if causal_pad > 0:
                first_frame = x[:, :, :1, :, :]
                pad_frames = mx.repeat(first_frame, causal_pad, axis=2)
                x = mx.concatenate([pad_frames, x], axis=2)
            temporal_padding = 0
        else:
            temporal_padding = pt

        x = x.transpose(0, 2, 3, 4, 1)
        out = mx.conv_general(x, self.weight, stride=self.stride, padding=(temporal_padding, ph, pw))
        out = out + self.bias
        out = out.transpose(0, 4, 1, 2, 3)
        return out

    def _can_collapse_temporal(self, kt: int, pt: int) -> bool:
        if kt == 1:
            return pt == 0
        if not self.causal_temporal:
            return False
        causal_pad = (2 * pt) if self.use_padding_causal else kt - 1
        return causal_pad == kt - 1
