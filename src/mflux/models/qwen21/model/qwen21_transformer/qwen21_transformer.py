from __future__ import annotations

import mlx.core as mx
import numpy as np
from mlx import nn

from mflux.models.common.config import ModelConfig
from mflux.models.common.config.config import Config
from mflux.models.qwen21.model.qwen21_transformer.qwen21_norm_out import Qwen21AdaLayerNormContinuous
from mflux.models.qwen21.model.qwen21_transformer.qwen21_rope import Qwen21Rope
from mflux.models.qwen21.model.qwen21_transformer.qwen21_text_projection import Qwen21TextProjection
from mflux.models.qwen21.model.qwen21_transformer.qwen21_time_text_embed import Qwen21TimeTextEmbed
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer_block import Qwen21TransformerBlock


class Qwen21Transformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 64,
        num_layers: int = 32,
        attention_head_dim: int = 128,
        num_attention_heads: int = 32,
        context_in_dim: int = 4096,
        mlp_ratio: int = 3,
        axes_dims_rope: tuple[int, int, int] = (16, 56, 56),
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.inner_dim = num_attention_heads * attention_head_dim
        self.pos_embed = Qwen21Rope(theta=10000, axes_dim=list(axes_dims_rope))
        self.time_text_embed = Qwen21TimeTextEmbed(embedding_dim=self.inner_dim)
        self.txt_in = Qwen21TextProjection(context_in_dim, self.inner_dim, eps=eps)
        self.img_in = nn.Linear(in_channels, self.inner_dim, bias=False)
        # One shared modulation for every block: [mod1.scale, mod1.gate, mod2.scale, mod2.gate].
        # Sequential so the checkpoint key modulation.1.weight lands unchanged.
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(self.inner_dim, 4 * self.inner_dim, bias=False))
        self.transformer_blocks = [
            Qwen21TransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                mlp_ratio=mlp_ratio,
                eps=eps,
            )
            for _ in range(num_layers)
        ]
        self.norm_out = Qwen21AdaLayerNormContinuous(embedding_dim=self.inner_dim, eps=eps)
        self._step_fn = None
        self.pdd_proj_out_weights = None
        self.pdd_sigmas = None
        self.proj_out = nn.Linear(self.inner_dim, out_channels, bias=False)
        self._geometry_cache: dict[tuple[int, int, int], tuple[mx.array, mx.array, mx.array]] = {}

    def __call__(
        self,
        t: int | float,
        config: Config,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        encoder_hidden_states_mask: mx.array | None = None,
        pdd_step: int | None = None,
    ) -> mx.array:
        if self.has_pdd and pdd_step is None:
            raise ValueError("A PDD transformer requires a denoising step index.")
        timestep = Qwen21Transformer._compute_timestep(t, config)
        timestep_rows = mx.concatenate([timestep, mx.zeros((1,), dtype=timestep.dtype)])
        rope_cos, rope_sin, attn_mask = self._geometry(
            text_len=encoder_hidden_states.shape[1],
            latent_height=config.height // 16,
            latent_width=config.width // 16,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
        )
        if self._step_fn is None:
            # compile after the shapes of one denoise step are known; mx.compile retraces
            # per shape, so prompt-length and resolution changes just compile again
            self._step_fn = mx.compile(self._forward)
        step = mx.array(0 if pdd_step is None else pdd_step)
        return self._step_fn(hidden_states, encoder_hidden_states, timestep_rows, rope_cos, rope_sin, attn_mask, step)

    @property
    def has_pdd(self) -> bool:
        return self.pdd_proj_out_weights is not None

    def enable_pdd(self, proj_out_weights: mx.array, sigmas: mx.array) -> None:
        expected_head_shape = (self.proj_out.weight.shape[0], self.inner_dim)
        if proj_out_weights.ndim != 3 or proj_out_weights.shape[1:] != expected_head_shape:
            raise ValueError(
                f"PDD output heads must have shape (steps, {expected_head_shape[0]}, {expected_head_shape[1]}), "
                f"got {proj_out_weights.shape}."
            )
        if sigmas.shape != (proj_out_weights.shape[0] + 1,):
            raise ValueError(f"PDD sigma grid must have {proj_out_weights.shape[0] + 1} values, got {sigmas.shape}.")
        self.pdd_proj_out_weights = proj_out_weights
        self.pdd_sigmas = sigmas.astype(mx.float32)

    def _forward(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        timestep_rows: mx.array,
        rope_cos: mx.array,
        rope_sin: mx.array,
        attn_mask: mx.array | None,
        pdd_step: mx.array,
    ) -> mx.array:
        text_len = encoder_hidden_states.shape[1]
        num_image_tokens = hidden_states.shape[1]

        temb = self.time_text_embed(timestep_rows)
        modulation = self.modulation(temb)
        mod1, mod2 = mx.split(modulation, 2, axis=-1)
        # causal_condition: text tokens read the t=0 row, target tokens the sampled-t row
        mod1 = Qwen21Transformer._select_modulation_rows(mod1, text_len, num_image_tokens)
        mod2 = Qwen21Transformer._select_modulation_rows(mod2, text_len, num_image_tokens)

        hidden_states = mx.concatenate([self.txt_in(encoder_hidden_states), self.img_in(hidden_states)], axis=1)

        for block in self.transformer_blocks:
            # mask None means padding-free: the segmented attention path (faster, same math)
            hidden_states = block(
                hidden_states, mod1, mod2, rope_cos, rope_sin, attn_mask, text_len if attn_mask is None else None
            )

        scale = self.norm_out.linear(nn.silu(temb))
        scale = Qwen21Transformer._select_modulation_rows(scale, text_len, num_image_tokens)
        hidden_states = self.norm_out(hidden_states, scale)
        if self.has_pdd:
            hidden_states = hidden_states @ self.pdd_proj_out_weights[pdd_step].T
        else:
            hidden_states = self.proj_out(hidden_states)
        return hidden_states[:, text_len:]

    def _geometry(
        self,
        text_len: int,
        latent_height: int,
        latent_width: int,
        encoder_hidden_states_mask: mx.array | None,
    ) -> tuple[mx.array, mx.array, mx.array | None]:
        cache_key = (text_len, latent_height, latent_width)
        if cache_key not in self._geometry_cache:
            rope_cos, rope_sin = self.pos_embed(text_len, latent_height, latent_width)
            has_padding = (
                encoder_hidden_states_mask is not None and int(mx.sum(encoder_hidden_states_mask).item()) < text_len
            )
            if not has_padding:
                # padding-free: no mask tensor; blocks use the segmented attention path
                self._geometry_cache[cache_key] = (rope_cos, rope_sin, None)
                return self._geometry_cache[cache_key]
            seq_len = text_len + latent_height * latent_width
            idx = mx.arange(seq_len)
            # block-causal: causal over the joint sequence, target rows see everything
            allowed = (idx[None, :] <= idx[:, None]) | (idx[:, None] >= text_len)
            text_valid = mx.concatenate(
                [
                    encoder_hidden_states_mask[0].astype(mx.bool_),
                    mx.ones((latent_height * latent_width,), dtype=mx.bool_),
                ]
            )
            allowed = allowed & text_valid[None, :]
            # the mask dtype must match the bf16 activation stream (fast SDPA requires it to promote)
            mask = mx.where(allowed, 0.0, -1e9).astype(ModelConfig.precision)[None, None, :, :]
            self._geometry_cache[cache_key] = (rope_cos, rope_sin, mask)
        return self._geometry_cache[cache_key]

    @staticmethod
    def _select_modulation_rows(params: mx.array, text_len: int, num_image_tokens: int) -> mx.array:
        # params holds [sampled-t, t=0] rows; the joint sequence is [text | target]
        text_row = mx.broadcast_to(params[1][None, None, :], (1, text_len, params.shape[-1]))
        image_row = mx.broadcast_to(params[0][None, None, :], (1, num_image_tokens, params.shape[-1]))
        return mx.concatenate([text_row, image_row], axis=1)

    @staticmethod
    def _compute_timestep(t: int | float, config: Config) -> mx.array:
        if isinstance(t, int) and t < len(config.scheduler.sigmas):
            time_step = config.scheduler.sigmas[t]
        elif isinstance(t, (int, float)):
            time_step = t / 1000.0 if t > 1.0 else t
        else:
            time_step = t
        return mx.array(np.full((1,), time_step, dtype=np.float32))
