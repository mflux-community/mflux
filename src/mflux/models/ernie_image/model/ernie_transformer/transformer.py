import mlx.core as mx
from mlx import nn

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.ernie_image.model.ernie_transformer.rope_embedder import ErnieRopeEmbedder
from mflux.models.ernie_image.model.ernie_transformer.timestep_embedder import (
    ErnieTimestepEmbedder,
    get_timestep_embedding,
)
from mflux.models.ernie_image.model.ernie_transformer.transformer_block import ErnieTransformerBlock


class ErniePatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.proj(x)
        B, H, W, C = x.shape
        return x.reshape(B, H * W, C)


class ErnieAdaLNContinuous(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=eps, affine=False)
        self.linear = nn.Linear(hidden_size, hidden_size * 2)

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        scale, shift = mx.split(self.linear(c), 2, axis=-1)
        x = self.norm(x)
        return x * (1 + scale[:, None, :]) + shift[:, None, :]


class ErnieTransformer(nn.Module):
    def __init__(
        self,
        hidden_size: int = 4096,
        num_attention_heads: int = 32,
        num_layers: int = 36,
        ffn_hidden_size: int = 12288,
        in_channels: int = 128,
        out_channels: int = 128,
        patch_size: int = 1,
        text_in_dim: int = 3072,
        rope_theta: int = 256,
        rope_axes_dim: list[int] | None = None,
        eps: float = 1e-6,
        qk_layernorm: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        rope_axes_dim = rope_axes_dim or [32, 48, 48]

        self.x_embedder = ErniePatchEmbed(in_channels, hidden_size, patch_size)
        self.text_proj = nn.Linear(text_in_dim, hidden_size, bias=False)
        self.time_embedding = ErnieTimestepEmbedder(hidden_size)
        self.adaln_modulation = nn.Linear(hidden_size, 6 * hidden_size)
        self.pos_embed = ErnieRopeEmbedder(dim=self.head_dim, theta=rope_theta, axes_dim=rope_axes_dim)
        self.layers = [
            ErnieTransformerBlock(hidden_size, num_attention_heads, ffn_hidden_size, eps, qk_layernorm)
            for _ in range(num_layers)
        ]

        self._pos_cache: dict = {}

        self.final_norm = ErnieAdaLNContinuous(hidden_size, eps)
        self.final_linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

    def get_pos_encoding(
        self,
        B: int,
        H: int,
        W: int,
        T: int,
        text_lens: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        N_img = H * W
        cache_key = (H, W, T, tuple(text_lens.tolist()))
        if cache_key not in self._pos_cache:
            grid_y, grid_x = mx.meshgrid(
                mx.arange(H, dtype=mx.float32),
                mx.arange(W, dtype=mx.float32),
                indexing="ij",
            )
            grid_yx = mx.stack([grid_y.reshape(-1), grid_x.reshape(-1)], axis=-1)

            t_coord = text_lens.astype(mx.float32)[:, None, None]
            image_ids = mx.concatenate(
                [mx.broadcast_to(t_coord, (B, N_img, 1)), mx.broadcast_to(grid_yx[None, :, :], (B, N_img, 2))],
                axis=-1,
            )

            text_pos = mx.arange(T, dtype=mx.float32)[None, :, None]
            text_ids = mx.concatenate(
                [mx.broadcast_to(text_pos, (B, T, 1)), mx.zeros((B, T, 2), dtype=mx.float32)],
                axis=-1,
            )

            all_ids = mx.concatenate([image_ids, text_ids], axis=1)
            freqs_cis = self.pos_embed(all_ids)

            cos = mx.cos(freqs_cis).astype(ModelConfig.precision)
            sin = mx.sin(freqs_cis).astype(ModelConfig.precision)

            valid_text = mx.arange(T)[None, :] < text_lens[:, None]
            img_mask = mx.ones((B, N_img), dtype=mx.bool_)
            bool_mask = mx.concatenate([img_mask, valid_text], axis=1)
            float_mask = mx.where(bool_mask, 0.0, -float("inf")).astype(mx.bfloat16)
            attn_mask = float_mask[:, None, None, :]

            if len(self._pos_cache) >= 64:
                self._pos_cache.pop(next(iter(self._pos_cache)))
            self._pos_cache[cache_key] = (cos, sin, attn_mask)

        return self._pos_cache[cache_key]

    def __call__(
        self,
        hidden_states: mx.array,
        timestep: mx.array,
        text_bth: mx.array,
        text_lens: mx.array,
        *,
        cos: mx.array | None = None,
        sin: mx.array | None = None,
        attn_mask: mx.array | None = None,
    ) -> mx.array:
        B, C, H, W = hidden_states.shape
        N_img = H * W
        T = text_bth.shape[1]

        img_emb = self.x_embedder(hidden_states.transpose(0, 2, 3, 1))
        text_emb = self.text_proj(text_bth)
        x = mx.concatenate([img_emb, text_emb], axis=1)

        if cos is None:
            cos, sin, attn_mask = self.get_pos_encoding(B, H, W, T, text_lens)

        t_emb = get_timestep_embedding(timestep.astype(mx.float32), self.hidden_size)
        c = self.time_embedding(t_emb)
        adaln_out = self.adaln_modulation(nn.silu(c))
        pieces = mx.split(adaln_out, 6, axis=-1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [p[:, None, :] for p in pieces]
        temb = (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)

        for layer in self.layers:
            x = layer(x, cos, sin, temb, attn_mask)

        img_tokens = x[:, :N_img, :]
        img_tokens = self.final_norm(img_tokens, c).astype(hidden_states.dtype)
        patches = self.final_linear(img_tokens)

        output = patches.reshape(B, H, W, self.patch_size, self.patch_size, self.out_channels)
        output = output.transpose(0, 5, 1, 3, 2, 4)
        output = output.reshape(B, self.out_channels, H * self.patch_size, W * self.patch_size)
        return output
