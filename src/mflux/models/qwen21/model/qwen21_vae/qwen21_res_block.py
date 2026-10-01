from mflux.models.qwen21.model.qwen21_vae.blocks import ResidualBlock


class Qwen21ResBlock(ResidualBlock):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__(in_dim, out_dim, text_mode=True)
