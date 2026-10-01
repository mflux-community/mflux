from mflux.models.qwen21.model.qwen21_vae.blocks import ChannelNorm


class Qwen21RMSNorm(ChannelNorm):
    def __init__(self, num_channels: int):
        super().__init__(num_channels, text_mode=True)
