from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TilingConfig:
    vae_decode_tiles_per_dim: int | None = 8
    vae_decode_tile_size: int = 512
    vae_decode_overlap: int = 8
    vae_encode_tiled: bool = True
    vae_encode_tile_size: int = 512
    vae_encode_tile_overlap: int = 64

    @staticmethod
    def may_tile_implicitly(model: object) -> bool:
        # Memory-saving modes choose tiled decoding on the user's behalf, so they only do so on
        # decoders where tiling costs nothing visually. A decoder that opts out still tiles
        # whenever the user asks for it with --vae-tiling.
        return bool(getattr(getattr(model, "vae", None), "supports_implicit_tiling", True))
