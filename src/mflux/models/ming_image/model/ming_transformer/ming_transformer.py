import mlx.core as mx
from mlx import nn

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.z_image.model.z_image_transformer.feed_forward import FeedForward
from mflux.models.z_image.model.z_image_transformer.final_layer import FinalLayer
from mflux.models.z_image.model.z_image_transformer.rope_embedder import RopeEmbedder
from mflux.models.z_image.model.z_image_transformer.timestep_embedder import TimestepEmbedder

SEQ_MULTI_OF = 32


class MingAttention(nn.Module):
    # Z-Image's attention with upstream Ming's dtype handling: RoPE is applied in float32 and
    # cast back, so the residual stream stays in the weights' dtype instead of promoting to float32.
    def __init__(self, dim: int, n_heads: int, eps: float = 1e-5):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = [nn.Linear(dim, dim, bias=False)]
        self.norm_q = nn.RMSNorm(self.head_dim, eps=eps)
        self.norm_k = nn.RMSNorm(self.head_dim, eps=eps)

    def __call__(self, x: mx.array, freqs_cis: mx.array) -> mx.array:
        B, L, D = x.shape
        q = self.norm_q(self.to_q(x).reshape(B, L, self.n_heads, self.head_dim))
        k = self.norm_k(self.to_k(x).reshape(B, L, self.n_heads, self.head_dim))
        v = self.to_v(x).reshape(B, L, self.n_heads, self.head_dim)
        q = MingAttention._rope(q, freqs_cis).transpose(0, 2, 1, 3)
        k = MingAttention._rope(k, freqs_cis).transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(q, k, v.transpose(0, 2, 1, 3), scale=self.head_dim**-0.5)
        return self.to_out[0](out.transpose(0, 2, 1, 3).reshape(B, L, D))

    @staticmethod
    def _rope(x: mx.array, freqs_cis: mx.array) -> mx.array:
        B, L, H, Dh = x.shape
        xf = x.astype(mx.float32).reshape(B, L, H, Dh // 2, 2)
        cos = freqs_cis[None, :, None, :, 0]
        sin = freqs_cis[None, :, None, :, 1]
        real = xf[..., 0] * cos - xf[..., 1] * sin
        imag = xf[..., 0] * sin + xf[..., 1] * cos
        return mx.stack([real, imag], axis=-1).reshape(B, L, H, Dh).astype(x.dtype)


class MingBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, norm_eps: float, modulation: bool):
        super().__init__()
        self.attention = MingAttention(dim, n_heads)
        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8))
        self.attention_norm1 = nn.RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = nn.RMSNorm(dim, eps=norm_eps)
        if modulation:
            self.adaLN_modulation = [nn.Linear(min(dim, 256), 4 * dim, bias=True)]

    def __call__(self, x: mx.array, freqs_cis: mx.array, t_emb: mx.array | None = None) -> mx.array:
        if t_emb is None:
            x = x + self.attention_norm2(self.attention(self.attention_norm1(x), freqs_cis))
            return x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))
        scale_msa, gate_msa, scale_mlp, gate_mlp = mx.split(self.adaLN_modulation[0](t_emb)[:, None], 4, axis=2)
        x = x + mx.tanh(gate_msa) * self.attention_norm2(self.attention(self.attention_norm1(x) * (1.0 + scale_msa), freqs_cis))  # fmt: off
        return x + mx.tanh(gate_mlp) * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * (1.0 + scale_mlp)))


class MingTransformer(nn.Module):
    """Ming-Image's DiT: the Z-Image single-stream architecture (same weights layout) with two
    conditioning streams. The caption sequence is cap_embedder(cap_feats) followed by the
    already-projected cap_feats_2, and alignment padding is zero-masked upstream, so for a single
    image the pad tokens are simply dropped while the image RoPE offset still counts them."""

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        dim: int = 3840,
        n_layers: int = 30,
        n_refiner_layers: int = 2,
        n_heads: int = 30,
        norm_eps: float = 1e-5,
        cap_feat_dim: int = 2560,
        rope_theta: float = 256.0,
        t_scale: float = 1000.0,
        axes_dims: tuple[int, ...] = (32, 48, 48),
        axes_lens: tuple[int, ...] = (20480, 512, 512),
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.t_scale = t_scale
        key = f"{patch_size}-1"
        embed_dim = patch_size * patch_size * in_channels
        self.all_x_embedder = {key: nn.Linear(embed_dim, dim, bias=True)}
        self.all_final_layer = {key: FinalLayer(dim, embed_dim)}
        self.t_embedder = TimestepEmbedder(out_size=min(dim, 256), mid_size=1024)
        self.cap_embedder = [nn.RMSNorm(cap_feat_dim, eps=norm_eps), nn.Linear(cap_feat_dim, dim, bias=True)]
        self.noise_refiner = [MingBlock(dim, n_heads, norm_eps, modulation=True) for _ in range(n_refiner_layers)]
        self.context_refiner = [MingBlock(dim, n_heads, norm_eps, modulation=False) for _ in range(n_refiner_layers)]
        self.layers = [MingBlock(dim, n_heads, norm_eps, modulation=True) for _ in range(n_layers)]
        self.rope_embedder = RopeEmbedder(theta=rope_theta, axes_dims=list(axes_dims), axes_lens=list(axes_lens))

    def __call__(self, x: mx.array, timestep: mx.array, cap_feats: mx.array, cap_feats_2: mx.array) -> mx.array:
        """x: (C, 1, H, W) latents; timestep: (1,) = 1 - sigma; cap_feats: (N1, 2560);
        cap_feats_2: (N2, dim). Returns the flow prediction (C, 1, H, W) with upstream's sign."""
        key = f"{self.patch_size}-1"
        dtype = ModelConfig.precision
        t_emb = self.t_embedder(timestep.astype(mx.float32) * self.t_scale).astype(dtype)

        tokens, (F, H, W) = MingTransformer._patchify(x.astype(dtype), self.patch_size)
        caption = mx.concatenate([self.cap_embedder[1](self.cap_embedder[0](cap_feats.astype(dtype))), cap_feats_2.astype(dtype)], axis=0)  # fmt: off
        x_pos, cap_pos = MingTransformer._position_ids(
            caption.shape[0], (F, H // self.patch_size, W // self.patch_size)
        )
        x_freqs = self.rope_embedder(x_pos)
        cap_freqs = self.rope_embedder(cap_pos)

        img = self.all_x_embedder[key](tokens)[None]
        for layer in self.noise_refiner:
            img = layer(img, x_freqs, t_emb)
        cap = caption[None]
        for layer in self.context_refiner:
            cap = layer(cap, cap_freqs)

        n_img = img.shape[1]
        unified = mx.concatenate([img, cap], axis=1)
        freqs = mx.concatenate([x_freqs, cap_freqs], axis=0)
        for layer in self.layers:
            unified = layer(unified, freqs, t_emb)
        out = self.all_final_layer[key](unified[:, :n_img], t_emb)[0]
        return -MingTransformer._unpatchify(out, (F, H, W), self.patch_size, self.in_channels)

    @staticmethod
    def _position_ids(cap_len: int, grid: tuple[int, int, int]) -> tuple[mx.array, mx.array]:
        # Caption tokens sit on the t axis starting at 1; the image grid starts after the
        # 32-aligned caption length, exactly where upstream places it with the padding present.
        cap_pos = mx.stack([mx.arange(1, cap_len + 1), mx.zeros(cap_len, mx.int32), mx.zeros(cap_len, mx.int32)], axis=-1)  # fmt: off
        cap_padded = cap_len + (-cap_len) % SEQ_MULTI_OF
        f, h, w = grid
        ft, ht, wt = mx.meshgrid(mx.arange(f), mx.arange(h), mx.arange(w), indexing="ij")
        x_pos = mx.stack([ft.flatten() + cap_padded + 1, ht.flatten(), wt.flatten()], axis=-1)
        return x_pos.astype(mx.int32), cap_pos.astype(mx.int32)

    @staticmethod
    def _patchify(x: mx.array, p: int) -> tuple[mx.array, tuple[int, int, int]]:
        C, F, H, W = x.shape
        x = x.reshape(C, F, 1, H // p, p, W // p, p)
        x = x.transpose(1, 3, 5, 2, 4, 6, 0)
        return x.reshape(F * (H // p) * (W // p), p * p * C), (F, H, W)

    @staticmethod
    def _unpatchify(x: mx.array, size: tuple[int, int, int], p: int, c: int) -> mx.array:
        F, H, W = size
        x = x.reshape(F, H // p, W // p, 1, p, p, c)
        x = x.transpose(6, 0, 3, 1, 4, 2, 5)
        return x.reshape(c, F, H, W)
