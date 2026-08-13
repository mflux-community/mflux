from mlx import nn

from mflux.models.common.pid_decoder.pixdit.pixdit_rms_norm import PixDiTRMSNorm


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm = PixDiTRMSNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def __call__(self, x):
        return self.linear(self.norm(x))
