import mlx.core as mx
import mlx.nn as nn

from mflux.models.depth_pro.model.decoder.feature_fusion_block_2d import FeatureFusionBlock2d
from mflux.models.depth_pro.model.depth_pro_util import DepthProUtil


class MultiresConvDecoder(nn.Module):
    def __init__(
        self,
        encoder_feature_dims: tuple[int, int, int, int] = (256, 512, 1024, 1024),
        decoder_features: int = 256,
    ):
        super().__init__()
        # Level 0 is upsample_latent0, which the encoder already projects to decoder_features.
        self.convs = [nn.Identity()] + [
            nn.Conv2d(in_channels=dim, out_channels=decoder_features, kernel_size=3, stride=1, padding=1, bias=False)
            for dim in encoder_feature_dims
        ]
        self.fusions = [FeatureFusionBlock2d(num_features=decoder_features, deconv=i > 0) for i in range(len(encoder_feature_dims) + 1)]  # fmt: off

    def __call__(
        self,
        x0_latent: mx.array,
        x1_latent: mx.array,
        x0_features: mx.array,
        x1_features: mx.array,
        x_global_features: mx.array,
    ) -> mx.array:
        # Process global features:
        features = DepthProUtil.apply_conv(x_global_features, self.convs[4])
        features = self.fusions[4](features)

        # Process remaining levels with skip connections:
        x1_skip_features = DepthProUtil.apply_conv(x1_features, self.convs[3])
        features = self.fusions[3](features, x1_skip_features)

        x0_skip_features = DepthProUtil.apply_conv(x0_features, self.convs[2])
        features = self.fusions[2](features, x0_skip_features)

        x1_skip_latents = DepthProUtil.apply_conv(x1_latent, self.convs[1])
        features = self.fusions[1](features, x1_skip_latents)

        x0_skip_latents = DepthProUtil.apply_conv(x0_latent, self.convs[0])
        features = self.fusions[0](features, x0_skip_latents)

        return features
