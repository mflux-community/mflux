from mflux.models.qwen21.model.qwen21_vae.blocks import UpBlock
from mflux.models.qwen21.model.qwen21_vae.qwen21_vae_config import Qwen21VAEConfig
from mflux.models.qwen21.model.qwen21_vae.vae import Decoder


class Qwen21ResidualUpBlock(UpBlock):
    def __init__(self, in_dim: int, out_dim: int, num_res_blocks: int, temporal_upsample: bool, up_flag: bool = True):
        super().__init__(in_dim, out_dim, num_res_blocks, up_flag, temporal_upsample, text_mode=True)


class Qwen21Decoder(Decoder):
    def __init__(
        self,
        dim: int = 144,
        z_dim: int = 64,
        dim_mult: tuple[int, ...] = (1, 2, 4, 8, 8),
        num_res_blocks: int = 2,
        temporal_upsample: tuple[bool, ...] = (True, True, True, False),
        out_channels: int = 4,
    ):
        config = Qwen21VAEConfig.resolve(
            dict(
                decoder_base_dim=dim,
                z_dim=z_dim,
                dim_mult=dim_mult,
                num_res_blocks=num_res_blocks,
                temporal_downsample=tuple(reversed(temporal_upsample)),
                out_channels=out_channels,
            )
        )
        super().__init__(config, text_mode=True)
