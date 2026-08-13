from __future__ import annotations

import math

import mlx.core as mx
from mlx import nn


def get_timestep_embedding(
    timesteps: mx.array,
    embedding_dim: int,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 0.0,
    scale: float = 1.0,
    max_period: int = 10000,
) -> mx.array:
    """Sinusoidal timestep embedding (port of diffusers ``get_timestep_embedding``).

    Args:
        timesteps: 1D array of timesteps, shape ``(B,)``.
        embedding_dim: Output embedding dimension.
        flip_sin_to_cos: If True, concatenate ``[cos, sin]`` (diffusers default).
        downscale_freq_shift: Shift applied to the frequency denominator.
        scale: Multiplier applied to ``timestep * freqs`` before sin/cos.
        max_period: Maximum sinusoid period.

    Returns:
        ``(B, embedding_dim)`` embedding.
    """
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * mx.arange(half_dim, dtype=mx.float32)
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = mx.exp(exponent)
    emb = timesteps.astype(mx.float32)[:, None] * emb[None, :]
    emb = scale * emb
    if flip_sin_to_cos:
        emb = mx.concatenate([mx.cos(emb), mx.sin(emb)], axis=-1)
    else:
        emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
    if embedding_dim % 2 == 1:
        emb = mx.pad(emb, [(0, 0), (0, 1)])
    return emb


class TimestepEmbedding(nn.Module):
    """Two-layer MLP over the sinusoidal timestep embedding (SiLU between)."""

    def __init__(self, in_channels: int, time_embed_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def __call__(self, sample: mx.array) -> mx.array:
        return self.linear_2(self.act(self.linear_1(sample)))


class Lumina2CombinedTimestepCaptionEmbedding(nn.Module):
    """Combined timestep + caption (instruction) embedding (Lumina2).

    Produces the modulation conditioning ``temb`` from the timestep and projects
    the Qwen3-VL instruction features into model dimension.

    Args:
        hidden_size: Transformer model dimension.
        instruction_feat_dim: Qwen3-VL hidden size feeding the caption embedder.
        frequency_embedding_size: Sinusoidal timestep dimension.
        norm_eps: RMSNorm epsilon for the caption embedder.
        timestep_scale: Scale applied inside the sinusoidal timestep projection.
    """

    def __init__(
        self,
        hidden_size: int,
        instruction_feat_dim: int,
        frequency_embedding_size: int = 256,
        norm_eps: float = 1e-5,
        timestep_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.timestep_scale = timestep_scale
        self.timestep_embedder = TimestepEmbedding(frequency_embedding_size, min(hidden_size, 1024))
        # Sequential(RMSNorm, Linear) — kept as a list to mirror the reference keys
        # (caption_embedder.0 = norm, caption_embedder.1 = linear).
        self.caption_embedder = [
            nn.RMSNorm(instruction_feat_dim, eps=norm_eps),
            nn.Linear(instruction_feat_dim, hidden_size, bias=True),
        ]

    def __call__(self, timestep: mx.array, instruction_hidden_states: mx.array) -> tuple[mx.array, mx.array]:
        time_proj = get_timestep_embedding(
            timestep,
            self.frequency_embedding_size,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.0,
            scale=self.timestep_scale,
        )
        time_embed = self.timestep_embedder(time_proj.astype(instruction_hidden_states.dtype))
        caption_embed = self.caption_embedder[1](self.caption_embedder[0](instruction_hidden_states))
        return time_embed, caption_embed


class LuminaRMSNormZero(nn.Module):
    """Adaptive RMSNorm-zero modulation (Lumina2).

    Applies ``RMSNorm(x) * (1 + scale_msa)`` and returns the remaining
    ``(gate_msa, scale_mlp, gate_mlp)`` modulation terms derived from ``temb``.

    Args:
        embedding_dim: Model dimension.
        norm_eps: RMSNorm epsilon.
    """

    def __init__(self, embedding_dim: int, norm_eps: float) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(min(embedding_dim, 1024), 4 * embedding_dim, bias=True)
        self.norm = nn.RMSNorm(embedding_dim, eps=norm_eps)

    def __call__(self, x: mx.array, emb: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        emb = self.linear(self.silu(emb))
        scale_msa, gate_msa, scale_mlp, gate_mlp = mx.split(emb, 4, axis=-1)
        x = self.norm(x) * (1 + scale_msa[:, None])
        return x, gate_msa, scale_mlp, gate_mlp


class LuminaLayerNormContinuous(nn.Module):
    """Continuous-conditioning LayerNorm with output projection (Lumina2 norm_out).

    Args:
        embedding_dim: Model dimension.
        conditioning_embedding_dim: Conditioning (``temb``) dimension.
        eps: LayerNorm epsilon.
        out_dim: Output projection dimension (``patch**2 * out_channels``).
    """

    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        eps: float = 1e-6,
        out_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear_1 = nn.Linear(conditioning_embedding_dim, embedding_dim, bias=True)
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, affine=False)
        self.linear_2 = nn.Linear(embedding_dim, out_dim, bias=True) if out_dim is not None else None

    def __call__(self, x: mx.array, conditioning_embedding: mx.array) -> mx.array:
        emb = self.linear_1(self.silu(conditioning_embedding).astype(x.dtype))
        x = self.norm(x) * (1 + emb)[:, None, :]
        if self.linear_2 is not None:
            x = self.linear_2(x)
        return x


class LuminaFeedForward(nn.Module):
    """SwiGLU feed-forward (Lumina2): ``linear_2(silu(linear_1(x)) * linear_3(x))``.

    Args:
        dim: Model dimension.
        inner_dim: Pre-rounding intermediate dimension (rounded up to ``multiple_of``).
        multiple_of: Round the intermediate dimension up to this multiple.
        ffn_dim_multiplier: Optional custom multiplier applied to ``inner_dim``.
    """

    def __init__(
        self,
        dim: int,
        inner_dim: int,
        multiple_of: int = 256,
        ffn_dim_multiplier: float | None = None,
    ) -> None:
        super().__init__()
        if ffn_dim_multiplier is not None:
            inner_dim = int(ffn_dim_multiplier * inner_dim)
        inner_dim = multiple_of * ((inner_dim + multiple_of - 1) // multiple_of)
        self.linear_1 = nn.Linear(dim, inner_dim, bias=False)
        self.linear_2 = nn.Linear(inner_dim, dim, bias=False)
        self.linear_3 = nn.Linear(dim, inner_dim, bias=False)
        self.silu = nn.SiLU()

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_2(self.silu(self.linear_1(x)) * self.linear_3(x))
