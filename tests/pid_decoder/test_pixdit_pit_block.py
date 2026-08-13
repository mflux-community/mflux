import mlx.core as mx

from mflux.models.common.pid_decoder.pixdit.pixdit_embedders import PixelTokenEmbedder
from mflux.models.common.pid_decoder.pixdit.pixdit_pit_block import PiTBlock
from mflux.models.common.pid_decoder.pixdit.pixdit_sincos_pos_embed import get_2d_sincos_pos_embed


def test_get_2d_sincos_pos_embed_shape():
    embed_dim, H, W = 8, 4, 6
    pos = get_2d_sincos_pos_embed(embed_dim, H, W)
    assert pos.shape == (H * W, embed_dim)
    # Row 0 (h=0, w=0) is the origin: sin(0)=0, cos(0)=1 for every frequency,
    # laid out as [sin_x, cos_x, sin_y, cos_y] (quarters of embed_dim each).
    quarter = embed_dim // 4
    expected = mx.concatenate([mx.zeros(quarter), mx.ones(quarter), mx.zeros(quarter), mx.ones(quarter)])
    assert mx.allclose(pos[0], expected).item()


def test_pixel_token_embedder_output_shape_and_ordering():
    # attn_hidden_size-style distinction isn't relevant here, but in_channels !=
    # hidden_size_output exercises the same "don't assume dims are equal" risk.
    patch_size = 4
    embedder = PixelTokenEmbedder(in_channels=3, hidden_size_output=8)
    B, H, W = 2, 8, 12  # Hs=2, Ws=3 patches
    x = mx.random.normal((B, 3, H, W))
    out = embedder(x, img_height=H, img_width=W, patch_size=patch_size)
    Hs, Ws = H // patch_size, W // patch_size
    P2 = patch_size * patch_size
    assert out.shape == (B * Hs * Ws, P2, 8)


def test_pit_block_output_shape():
    patch_size = 4
    block = PiTBlock(
        pixel_hidden_size=8,
        patch_hidden_size=16,
        patch_size=patch_size,
        attn_hidden_size=16,  # != pixel_hidden_size -- exercises compress/expand round-trip
        attn_num_heads=2,
        rope_ref_grid_h=2,
        rope_ref_grid_w=2,
    )
    Hs, Ws = 2, 2  # 2x2 patch grid
    BL = Hs * Ws
    x = mx.random.normal((BL, patch_size * patch_size, 8))
    s_cond = mx.random.normal((BL, 16))
    out = block(x, s_cond, image_height=Hs * patch_size, image_width=Ws * patch_size, patch_size=patch_size)
    assert out.shape == x.shape


def test_pit_block_handles_attn_dim_smaller_than_pixel_stream():
    # Real checkpoint has attn_hidden_size=1152, pixel_hidden_size=16 (attn >> pixel).
    # Also check the opposite direction (attn < p2*pixel_dim) round-trips cleanly.
    patch_size = 4
    p2 = patch_size * patch_size  # 16
    block = PiTBlock(
        pixel_hidden_size=16,
        patch_hidden_size=8,
        patch_size=patch_size,
        attn_hidden_size=32,  # << p2 * pixel_hidden_size (=256)
        attn_num_heads=4,
        rope_ref_grid_h=2,
        rope_ref_grid_w=2,
    )
    Hs, Ws = 2, 2
    BL = Hs * Ws
    x = mx.random.normal((BL, p2, 16))
    s_cond = mx.random.normal((BL, 8))
    out = block(x, s_cond, image_height=Hs * patch_size, image_width=Ws * patch_size, patch_size=patch_size)
    assert out.shape == x.shape
    assert not mx.allclose(out, x).item()
