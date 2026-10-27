import mlx.core as mx
from mlx import nn

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.fibo2.model.fibo2_transformer.fibo2_perceiver import Fibo2Perceiver
from mflux.models.fibo2.model.fibo2_transformer.fibo2_transformer_block import Fibo2TransformerBlock
from mflux.models.z_image.model.z_image_transformer.final_layer import FinalLayer
from mflux.models.z_image.model.z_image_transformer.rope_embedder import RopeEmbedder
from mflux.models.z_image.model.z_image_transformer.timestep_embedder import TimestepEmbedder


class Fibo2Transformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 32,
        patch_size: int = 2,
        dim: int = 3584,
        n_layers: int = 44,
        n_refiner_layers: int = 2,
        n_heads: int = 28,
        norm_eps: float = 1e-5,
        cap_feat_dim: int = 7680,
        injection_layer_ids: tuple[int, ...] = (6, 14, 22, 30, 38),
        perceiver_num_layers: int = 2,
        max_num_gist_tokens: int = 512,
        rope_theta: float = 256.0,
        t_scale: float = 1000.0,
        axes_dims: tuple[int, ...] = (32, 48, 48),
        axes_lens: tuple[int, ...] = (2048, 512, 512),
    ):
        super().__init__()
        self.patch_size = patch_size
        self.t_scale = t_scale
        self.injection_layer_ids = injection_layer_ids
        self.max_num_gist_tokens = max_num_gist_tokens
        self.x_embedder = nn.Linear(patch_size * patch_size * in_channels, dim, bias=True)
        self.t_embedder = TimestepEmbedder(out_size=min(dim, 256), mid_size=1024)
        self.rope_embedder = RopeEmbedder(theta=rope_theta, axes_dims=list(axes_dims), axes_lens=list(axes_lens))
        self.noise_refiner = [Fibo2TransformerBlock(dim, n_heads, norm_eps) for _ in range(n_refiner_layers)]
        self.perceiver = Fibo2Perceiver(dim, n_heads, cap_feat_dim, perceiver_num_layers, norm_eps, max_gist=max_num_gist_tokens)  # fmt: off
        self.layers = [
            Fibo2TransformerBlock(dim, n_heads, norm_eps, cap_feat_dim if i in injection_layer_ids else None)
            for i in range(n_layers)
        ]
        self.final_layer = FinalLayer(dim, patch_size * patch_size * in_channels)

    def __call__(
        self,
        latents: mx.array,
        sigma: mx.array,
        encoder_hidden_states: mx.array,
        context_latents: list[mx.array] | None = None,
    ) -> mx.array:
        batch_size, channels, height, width = latents.shape
        height_tokens, width_tokens = height // self.patch_size, width // self.patch_size
        num_image_tokens = height_tokens * width_tokens

        # Image tokens get the timestep's modulation, clean tokens (gist and context images) the t = 0 one
        t_noisy = self.t_embedder(sigma * self.t_scale).astype(ModelConfig.precision)
        t_clean = self.t_embedder(mx.zeros_like(sigma)).astype(ModelConfig.precision)

        tokens = [self._patchify(latents), *(self._patchify(context) for context in context_latents or [])]
        hidden_states = self.x_embedder(mx.concatenate(tokens, axis=1).astype(ModelConfig.precision))
        image_freqs = self.rope_embedder(self.image_position_ids(latents, context_latents))

        refiner_noise_mask = None
        if context_latents:
            refiner_noise_mask = Fibo2Transformer._noise_mask(batch_size, hidden_states.shape[1], 0, num_image_tokens)
        for block in self.noise_refiner:
            hidden_states = block(hidden_states, image_freqs, t_noisy, t_clean, refiner_noise_mask)

        # Gist tokens from the first text bundle, then one stream: gist, image, context
        num_gist_tokens = self.perceiver.gist_count(encoder_hidden_states.shape[2])
        gist = self.perceiver(encoder_hidden_states[:, 0], num_gist_tokens)
        gist_freqs = self.rope_embedder(Fibo2Transformer._coordinate_grid((num_gist_tokens, 1, 1), (1, 0, 0)))
        hidden_states = mx.concatenate([gist, hidden_states], axis=1)
        freqs_cis = mx.concatenate([gist_freqs, image_freqs], axis=0)
        noise_mask = Fibo2Transformer._noise_mask(batch_size, hidden_states.shape[1], num_gist_tokens, num_image_tokens)  # fmt: off

        bundle_index = {layer_id: index + 1 for index, layer_id in enumerate(self.injection_layer_ids)}
        for layer_id, block in enumerate(self.layers):
            text = encoder_hidden_states[:, bundle_index[layer_id]] if layer_id in bundle_index else None
            hidden_states = block(
                hidden_states,
                freqs_cis,
                t_noisy,
                t_clean,
                noise_mask,
                encoder_hidden_states=text,
            )

        hidden_states = self.final_layer(hidden_states[:, num_gist_tokens : num_gist_tokens + num_image_tokens], t_noisy)  # fmt: off
        hidden_states = hidden_states.reshape(batch_size, height_tokens, width_tokens, channels, self.patch_size, self.patch_size)  # fmt: off
        return hidden_states.transpose(0, 3, 1, 4, 2, 5).reshape(batch_size, channels, height, width)

    def image_position_ids(self, latents: mx.array, context_latents: list[mx.array] | None = None) -> mx.array:
        # The image sits on RoPE plane max_num_gist_tokens + 1 whatever the gist count, context image k on the k-th
        # plane after it, each with its own rows and columns
        image_plane = self.max_num_gist_tokens + 1
        p = self.patch_size
        sizes = [latents.shape[2:], *(context.shape[2:] for context in context_latents or [])]
        ids = [
            Fibo2Transformer._coordinate_grid((1, height // p, width // p), (image_plane + k, 0, 0))
            for k, (height, width) in enumerate(sizes)
        ]
        return mx.concatenate(ids, axis=0)

    def _patchify(self, latents: mx.array) -> mx.array:
        # 2x2 latent patches become tokens, features ordered (channel, dy, dx) as in training
        batch_size, channels, height, width = latents.shape
        p = self.patch_size
        latents = latents.reshape(batch_size, channels, height // p, p, width // p, p)
        return latents.transpose(0, 2, 4, 1, 3, 5).reshape(batch_size, (height // p) * (width // p), channels * p * p)

    @staticmethod
    def _noise_mask(batch_size: int, seq_len: int, start: int, length: int) -> mx.array:
        positions = mx.arange(seq_len)
        mask = (positions >= start) & (positions < start + length)
        return mx.broadcast_to(mask[None], (batch_size, seq_len))

    @staticmethod
    def _coordinate_grid(size: tuple[int, int, int], start: tuple[int, int, int]) -> mx.array:
        axes = [mx.arange(x0, x0 + span, dtype=mx.int32) for x0, span in zip(start, size)]
        return mx.stack(mx.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
