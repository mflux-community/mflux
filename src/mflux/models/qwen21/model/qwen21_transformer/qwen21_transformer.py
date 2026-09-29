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
        self._image_step_fn = None
        # Per-prompt text-prefix K/V caches, bounded to the two embeddings of the most
        # recent generation (positive and negative CFG prompts), keyed by the
        # encoder-hidden-states array identity (the caller caches embeddings per prompt,
        # so the object persists across steps).
        self.use_text_cache = True
        self._text_caches: dict[int, dict] = {}
        self.proj_out = nn.Linear(self.inner_dim, out_channels, bias=False)
        self._geometry_cache: dict[tuple[int, int, int], tuple[mx.array, mx.array, mx.array]] = {}

    def __call__(
        self,
        t: int | float,
        config: Config,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        encoder_hidden_states_mask: mx.array | None = None,
    ) -> mx.array:
        timestep = Qwen21Transformer._compute_timestep(t, config)
        rope_cos, rope_sin, attn_mask = self._geometry(
            text_len=encoder_hidden_states.shape[1],
            latent_height=config.height // 16,
            latent_width=config.width // 16,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
        )
        text_len = encoder_hidden_states.shape[1]
        if attn_mask is not None or not self.use_text_cache:
            # padded prompts (and cache-disabled runs) keep the full joint recompute path
            timestep_rows = mx.concatenate([timestep, mx.zeros((1,), dtype=timestep.dtype)])
            if self._step_fn is None:
                # compile after the shapes of one denoise step are known; mx.compile retraces
                # per shape, so prompt-length and resolution changes just compile again
                self._step_fn = mx.compile(self._forward)
            return self._step_fn(hidden_states, encoder_hidden_states, timestep_rows, rope_cos, rope_sin, attn_mask)

        geometry = (text_len, config.height // 16, config.width // 16)
        cache = self._text_caches.get(id(encoder_hidden_states))
        if cache is None or cache["geometry"] != geometry or cache["embedding"] is not encoder_hidden_states:
            cache = self._build_text_cache(encoder_hidden_states, rope_cos, rope_sin, geometry)
            # Evict oldest entries first so the cache holds at most the two
            # embeddings of one generation (the positive and negative CFG prompts);
            # a single long prompt's per-block K/V is already O(100 MB).
            while len(self._text_caches) >= 2:
                self._text_caches.pop(next(iter(self._text_caches)))
            self._text_caches[id(encoder_hidden_states)] = cache
        if self._image_step_fn is None:
            # like _step_fn: retraces per (image-token-count, text-length) shape change
            self._image_step_fn = mx.compile(self._image_forward)
        return self._image_step_fn(
            hidden_states,
            timestep,
            cache["text_kvs"],
            rope_cos[text_len:],
            rope_sin[text_len:],
        )

    def _build_text_cache(
        self,
        encoder_hidden_states: mx.array,
        rope_cos: mx.array,
        rope_sin: mx.array,
        geometry: tuple[int, int, int],
    ) -> dict:
        """Compute the text hidden stream once per (prompt, geometry).

        Text tokens read the t=0 modulation row (both timestep rows are zero here)
        and attend only causally to each other, so this stream — and each block's
        post-rope K/V over the text positions — is identical on every denoise step.
        """
        text_len = encoder_hidden_states.shape[1]
        timestep_rows = mx.zeros((2,), dtype=mx.float32)
        temb = self.time_text_embed(timestep_rows)
        modulation = self.modulation(temb)
        mod1, mod2 = mx.split(modulation, 2, axis=-1)
        mod1 = mx.broadcast_to(mod1[1][None, None, :], (1, text_len, mod1.shape[-1]))
        mod2 = mx.broadcast_to(mod2[1][None, None, :], (1, text_len, mod2.shape[-1]))

        hidden_states = self.txt_in(encoder_hidden_states)
        text_kvs = []
        for block in self.transformer_blocks:
            hidden_states, key_text, value_text = block.text_forward(
                hidden_states, mod1, mod2, rope_cos[:text_len], rope_sin[:text_len]
            )
            text_kvs.append((key_text, value_text))
        arrays = [hidden_states]
        for key_text, value_text in text_kvs:
            arrays.extend((key_text, value_text))
        mx.eval(*arrays)
        return {"embedding": encoder_hidden_states, "geometry": geometry, "text_kvs": text_kvs}

    def _image_forward(
        self,
        hidden_states: mx.array,
        timestep: mx.array,
        text_kvs: list[tuple[mx.array, mx.array]],
        rope_cos: mx.array,
        rope_sin: mx.array,
    ) -> mx.array:
        # image tokens only: the text prefix contributes through the cached per-block K/V
        num_image_tokens = hidden_states.shape[1]
        timestep_rows = mx.concatenate([timestep, mx.zeros((1,), dtype=timestep.dtype)])
        temb = self.time_text_embed(timestep_rows)
        modulation = self.modulation(temb)
        mod1, mod2 = mx.split(modulation, 2, axis=-1)
        # image tokens read the sampled-t row (row 0)
        mod1 = mx.broadcast_to(mod1[0][None, None, :], (1, num_image_tokens, mod1.shape[-1]))
        mod2 = mx.broadcast_to(mod2[0][None, None, :], (1, num_image_tokens, mod2.shape[-1]))

        hidden_states = self.img_in(hidden_states)
        for block, (key_text, value_text) in zip(self.transformer_blocks, text_kvs):
            hidden_states = block.image_forward(
                hidden_states, key_text, value_text, mod1, mod2, rope_cos, rope_sin
            )

        scale = self.norm_out.linear(nn.silu(temb))
        scale = mx.broadcast_to(scale[0][None, None, :], (1, num_image_tokens, scale.shape[-1]))
        hidden_states = self.proj_out(self.norm_out(hidden_states, scale))
        return hidden_states

    def _forward(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        timestep_rows: mx.array,
        rope_cos: mx.array,
        rope_sin: mx.array,
        attn_mask: mx.array | None,
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
        hidden_states = self.proj_out(self.norm_out(hidden_states, scale))
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
