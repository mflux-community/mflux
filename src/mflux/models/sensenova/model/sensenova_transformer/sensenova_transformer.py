import mlx.core as mx
from mlx import nn

from mflux.models.sensenova.model.sensenova_transformer.sensenova_conv_head import SenseNovaConvHead
from mflux.models.sensenova.model.sensenova_transformer.sensenova_decoder_layer import SenseNovaDecoderLayer
from mflux.models.sensenova.model.sensenova_transformer.sensenova_patch_embedder import SenseNovaPatchEmbedder
from mflux.models.sensenova.model.sensenova_transformer.sensenova_rms_norm import SenseNovaRMSNorm
from mflux.models.sensenova.model.sensenova_transformer.sensenova_rope import SenseNovaRope
from mflux.models.sensenova.model.sensenova_transformer.sensenova_timestep_embedder import SenseNovaTimestepEmbedder


class SenseNovaTransformer(nn.Module):
    # The gen stream of the MoT backbone. Pixel space in, clean-image prediction out (no VAE).
    NOISE_SCALE_MAX = 16.0
    T_EPS = 0.02
    TOKEN_SIZE = 32

    def __init__(
        self,
        num_layers: int = 42,
        hidden_size: int = 4096,
        intermediate_size: int = 12288,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.patch_embedder = SenseNovaPatchEmbedder(hidden_size=hidden_size)
        self.timestep_embedder = SenseNovaTimestepEmbedder(hidden_size)
        self.noise_scale_embedder = SenseNovaTimestepEmbedder(hidden_size)
        self.layers = [SenseNovaDecoderLayer(hidden_size, intermediate_size, eps) for _ in range(num_layers)]
        self.norm = SenseNovaRMSNorm(hidden_size, eps)
        self.head = SenseNovaConvHead(hidden_size)

    def __call__(
        self,
        image: mx.array,
        t: mx.array | float,
        noise_scale: float,
        prefix_kv: list[tuple[mx.array, mx.array]],
        text_len: int,
    ) -> mx.array:
        # image: [B, 3, H, W] (the noisy x), prefix_kv: per layer (k, v) [B, 8, text_len, 128]; returns x_pred [B, 3, H, W]
        embeds = self.embed(image, t, noise_scale)
        hidden = self.denoise(embeds, prefix_kv, text_len, image.shape[2] // self.TOKEN_SIZE, image.shape[3] // self.TOKEN_SIZE)  # fmt: skip
        return self.predict_image(hidden, image.shape[2] // self.TOKEN_SIZE, image.shape[3] // self.TOKEN_SIZE)

    def embed(self, image: mx.array, t: mx.array | float, noise_scale: float) -> mx.array:
        t = mx.broadcast_to(mx.array(t, dtype=mx.float32).reshape(-1), (image.shape[0],))
        scale = mx.full(t.shape, noise_scale / self.NOISE_SCALE_MAX, dtype=mx.float32)
        conditioning = self.timestep_embedder(t) + self.noise_scale_embedder(scale)
        return self.patch_embedder(image) + conditioning[:, None, :]

    def denoise(
        self,
        embeds: mx.array,
        prefix_kv: list[tuple[mx.array, mx.array]],
        text_len: int,
        grid_h: int,
        grid_w: int,
    ) -> mx.array:
        # The image tokens attend to the prefix and to each other without a mask; their own K/V are not cached
        rope = SenseNovaRope.cos_sin(SenseNovaRope.indexes(text_len, grid_h, grid_w), embeds.dtype)
        for layer, kv in zip(self.layers, prefix_kv):
            embeds = layer(embeds, rope, kv)
        return self.norm(embeds)

    def predict_image(self, hidden: mx.array, grid_h: int, grid_w: int) -> mx.array:
        return self.head(hidden.reshape(hidden.shape[0], grid_h, grid_w, -1))

    @staticmethod
    def velocity(x_pred: mx.array, image: mx.array, t: mx.array | float) -> mx.array:
        return ((x_pred - image) / mx.maximum(1.0 - mx.array(t, dtype=mx.float32), SenseNovaTransformer.T_EPS)).astype(
            x_pred.dtype
        )
