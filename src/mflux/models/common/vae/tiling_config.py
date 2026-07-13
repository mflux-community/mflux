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
