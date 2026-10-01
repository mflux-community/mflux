from mflux.models.qwen21.model.qwen21_vae.blocks import Resample


class Qwen21Resample(Resample):
    def __init__(self, dim: int, out_dim: int, mode: str):
        if mode not in {"downsample", "upsample"}:
            raise ValueError(f"Unsupported resample mode: {mode}")
        super().__init__(
            dim, up=mode == "upsample", temporal=False, text_mode=True, out_dim=out_dim if mode == "upsample" else dim
        )
