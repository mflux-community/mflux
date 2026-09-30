from mflux.models.qwen21.model.qwen21_vae.blocks import DownBlock
from mflux.models.qwen21.model.qwen21_vae.qwen21_vae_config import Qwen21VAEConfig
from mflux.models.qwen21.model.qwen21_vae.vae import Encoder


class Qwen21ResidualDownBlock(DownBlock):
    def __init__(
        self, in_dim: int, out_dim: int, num_res_blocks: int, temporal_downsample: bool, down_flag: bool = True
    ):
        super().__init__(in_dim, out_dim, num_res_blocks, down_flag, temporal_downsample, text_mode=True)


class Qwen21Encoder(Encoder):
    def __init__(
        self,
        in_channels: int = 4,
        dim: int = 96,
        z_dim: int = 128,
        dim_mult: tuple[int, ...] = (1, 2, 4, 8, 8),
        num_res_blocks: int = 2,
        temporal_downsample: tuple[bool, ...] = (False, True, True, True),
    ):
        config = Qwen21VAEConfig.resolve(
            dict(
                in_channels=in_channels,
                base_dim=dim,
                z_dim=z_dim // 2,
                dim_mult=dim_mult,
                num_res_blocks=num_res_blocks,
                temporal_downsample=temporal_downsample,
            )
        )
        super().__init__(config, text_mode=True)
