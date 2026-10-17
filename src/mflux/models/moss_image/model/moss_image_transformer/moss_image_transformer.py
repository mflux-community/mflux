import mlx.core as mx
import mlx.nn as nn

from mflux.models.moss_image.model.moss_image_transformer.moss_image_block import (
    MOSSImageBlock,
    RMSNorm,
)
from mflux.models.moss_image.model.moss_image_transformer.moss_image_rope import MOSSImageRoPE
from mflux.models.moss_image.model.moss_image_transformer.moss_image_text_embedder import (
    MOSSImageTextProjection,
)


class MOSSImageTransformer(nn.Module):
    def __init__(
        self,
        dim: int = 3072,
        ffn_dim: int = 8192,
        num_heads: int = 24,
        num_layers: int = 42,
        text_dim: int = 4096,
        freq_dim: int = 256,
        latent_channels: int = 128,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.latent_channels = latent_channels
        self.freq_dim = freq_dim
        self.num_layers = num_layers

        self.video_embedder = nn.Linear(latent_channels, dim, bias=True)
        self.text_embedder = MOSSImageTextProjection(text_dim, dim, ffn_dim, num_heads)
        self.rope = MOSSImageRoPE(dim // num_heads, rope_base)

        self.time_embedding_0 = nn.Linear(freq_dim, dim, bias=True)
        self.time_embedding_2 = nn.Linear(dim, dim, bias=True)

        self.shared_block_modulation = nn.Linear(dim, 6 * dim, bias=True)
        self.video_head_modulation = nn.Linear(dim, 2 * dim, bias=True)

        self.layers = [MOSSImageBlock(dim, ffn_dim, num_heads) for _ in range(num_layers)]

        self.final_norm_video = RMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)

    def time_embed(self, timestep: float | mx.array) -> tuple[mx.array, mx.array]:
        # Timestep sinusoidal embedding in float32 / float64
        t = mx.atleast_1d(mx.array(timestep, dtype=mx.float32))
        half = self.freq_dim // 2
        freq = 10000.0 ** (-mx.arange(half, dtype=mx.float32) / half)
        phase = t[:, None] * freq[None, :]
        sin_part = mx.sin(phase)
        cos_part = mx.cos(phase)
        embedding = mx.concatenate([cos_part, sin_part], axis=-1)

        h = nn.silu(self.time_embedding_0(embedding))
        time_feat = self.time_embedding_2(h)

        mod = mx.reshape(self.shared_block_modulation(time_feat), (t.shape[0], 6, self.dim))
        head = mx.reshape(self.video_head_modulation(time_feat), (t.shape[0], 2, self.dim))
        return mod, head

    def __call__(
        self,
        hidden_states: mx.array,
        timestep: float | mx.array,
        encoder_hidden_states: mx.array,
        skip_layers: list[int] | tuple[int, ...] | None = None,
    ) -> mx.array:
        # hidden_states: (B, C, H, W), C = 128
        B, C, H, W = hidden_states.shape
        skip_set = set(skip_layers or ())

        mod, head = self.time_embed(timestep)

        # video embed: (B, H*W, dim)
        # flatten(2).transpose(1, 2) in PyTorch -> (B, H*W, C)
        video = mx.reshape(hidden_states, (B, C, H * W))
        video = mx.transpose(video, (0, 2, 1))
        video = self.video_embedder(video)

        # text embed
        text_len = encoder_hidden_states.shape[1]
        rope = self.rope(H, W, text_len)

        text_rope = rope[:, H * W :]
        text = self.text_embedder(encoder_hidden_states, text_rope)

        x = mx.concatenate([video, text], axis=1)

        for layer_idx, layer in enumerate(self.layers):
            if layer_idx not in skip_set:
                x = layer(x, rope, mod)

        v_out = x[:, : H * W, :]
        v_out = self.final_norm_video(v_out)

        shift_h = head[:, 0:1, :]
        scale_h = head[:, 1:2, :]
        v_out = v_out * (1.0 + scale_h) + shift_h
        v_out = self.final_linear_video(v_out)

        # reshape back to (B, C, H, W)
        v_out = mx.transpose(v_out, (0, 2, 1))
        out = mx.reshape(v_out, (B, C, H, W))
        return out
