import mlx.core as mx
import mlx.nn as nn

from mflux.models.depth_pro.model.decoder.multires_conv_decoder import MultiresConvDecoder
from mflux.models.depth_pro.model.encoder.depth_pro_encoder import DepthProEncoder
from mflux.models.depth_pro.model.head.fov_head import FOVHead


class DepthProModel(nn.Module):
    def __init__(
        self,
        embed_dim: int = 1024,
        num_heads: int = 16,
        mlp_hidden_dim: int = 4096,
        num_blocks: int = 24,
        hook_block_ids: tuple[int, int] = (5, 11),
        encoder_feature_dims: tuple[int, int, int, int] = (256, 512, 1024, 1024),
        decoder_features: int = 256,
    ):
        super().__init__()
        self.encoder = DepthProEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_hidden_dim=mlp_hidden_dim,
            num_blocks=num_blocks,
            hook_block_ids=hook_block_ids,
            encoder_feature_dims=encoder_feature_dims,
            decoder_features=decoder_features,
        )
        self.decoder = MultiresConvDecoder(encoder_feature_dims=encoder_feature_dims, decoder_features=decoder_features)
        self.head = FOVHead(dim_decoder=decoder_features)

    def __call__(self, x0: mx.array, x1: mx.array, x2: mx.array) -> tuple[mx.array, mx.array]:
        x0_lat, x1_lat, x0_feat, x1_feat, x_global = self.encoder(x0, x1, x2)
        decoded = self.decoder(x0_lat, x1_lat, x0_feat, x1_feat, x_global)
        return self.head(decoded)
