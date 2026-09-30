from mflux.models.qwen21.model.qwen21_vae.blocks import AttentionBlock


class Qwen21AttentionBlock(AttentionBlock):
    def __init__(self, dim: int):
        super().__init__(dim, text_mode=True)
