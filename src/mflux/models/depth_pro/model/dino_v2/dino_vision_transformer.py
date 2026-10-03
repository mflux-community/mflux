import mlx.core as mx
import mlx.nn as nn

from mflux.models.depth_pro.model.dino_v2.patch_embed import PatchEmbed
from mflux.models.depth_pro.model.dino_v2.transformer_block import TransformerBlock


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 1024,
        num_heads: int = 16,
        mlp_hidden_dim: int = 4096,
        num_blocks: int = 24,
        hook_block_ids: tuple[int, int] = (5, 11),
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hook_block_ids = hook_block_ids
        self.cls_token = mx.random.normal(shape=(1, 1, embed_dim))
        # 577 = a 24x24 patch grid plus the cls token. DepthProUtil.split always gives 384 px patches.
        self.pos_embed = mx.random.normal(shape=(1, 577, embed_dim))
        self.patch_embed = PatchEmbed(embed_dim=embed_dim)
        self.blocks = [TransformerBlock(dim=embed_dim, num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim) for i in range(num_blocks)]  # fmt: off
        self.norm = nn.LayerNorm(dims=embed_dim, eps=1e-6, bias=True)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        backbone_highres_hook0 = None
        backbone_highres_hook1 = None

        x = self.patch_embed(x)
        x = self._pos_embed(x)
        for i, block in enumerate(self.blocks):
            x = block(x)

            # Save intermediary results for later
            if i == self.hook_block_ids[0]:
                backbone_highres_hook0 = x
            if i == self.hook_block_ids[1]:
                backbone_highres_hook1 = x

        x = self.norm(x)
        return x, backbone_highres_hook0, backbone_highres_hook1

    def _pos_embed(self, x: mx.array) -> mx.array:
        B, H, W, C = x.shape
        x = x.reshape((B, -1, self.embed_dim))
        to_cat = [mx.broadcast_to(self.cls_token, (B,) + self.cls_token.shape[1:])]
        x = mx.concatenate(to_cat + [x], axis=1)
        x = x + self.pos_embed
        return x
