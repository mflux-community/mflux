from __future__ import annotations

import mlx.core as mx
import numpy as np
from mlx import nn

from mflux.models.common.config.config import Config
from mflux.models.flux.model.flux_transformer.ada_layer_norm_continuous import AdaLayerNormContinuous
from mflux.models.qwen.model.qwen_transformer.qwen_rope import QwenEmbedRopeMLX
from mflux.models.qwen.model.qwen_transformer.qwen_time_text_embed import QwenTimeTextEmbed
from mflux.models.qwen.model.qwen_transformer.qwen_transformer_block import QwenTransformerBlock
from mflux.models.qwen.model.qwen_transformer.qwen_transformer_rms_norm import QwenTransformerRMSNorm


class QwenTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        patch_size: int = 2,
        zero_cond_t: bool = False,
    ) -> None:
        super().__init__()
        # Qwen-Image-Edit-2511 (transformer config zero_cond_t): the tokens of the reference images are
        # modulated with the embedding of timestep 0, and only the noisy latent with the current one.
        self.zero_cond_t = zero_cond_t
        self.inner_dim = num_attention_heads * attention_head_dim
        self.img_in = nn.Linear(in_channels, self.inner_dim)
        self.txt_norm = QwenTransformerRMSNorm(joint_attention_dim, eps=1e-6)
        self.txt_in = nn.Linear(joint_attention_dim, self.inner_dim)
        self.time_text_embed = QwenTimeTextEmbed(timestep_proj_dim=256, inner_dim=self.inner_dim)
        self.pos_embed = QwenEmbedRopeMLX(theta=10000, axes_dim=[16, 56, 56], scale_rope=True)
        self.transformer_blocks = [QwenTransformerBlock(dim=self.inner_dim, num_heads=num_attention_heads, head_dim=attention_head_dim) for i in range(num_layers)]  # fmt: off
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * out_channels)

    def __call__(
        self,
        t: int,
        config: Config,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        encoder_hidden_states_mask: mx.array,
        qwen_image_ids: mx.array | None = None,
        cond_image_grid: tuple[int, int, int] | None = None,
    ) -> mx.array:
        hidden_states = self.img_in(hidden_states)
        batch_size = hidden_states.shape[0]
        timestep = QwenTransformer._compute_timestep(t, config)
        timestep = mx.broadcast_to(timestep, (batch_size,)).astype(hidden_states.dtype)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)
        text_embeddings = self.time_text_embed(timestep, hidden_states)
        cond_embeddings, is_cond_token = None, None
        if self.zero_cond_t and cond_image_grid is not None:
            cond_embeddings = self.time_text_embed(mx.zeros_like(timestep), hidden_states)
            target_tokens = (config.height // 16) * (config.width // 16)
            is_cond_token = (mx.arange(hidden_states.shape[1]) >= target_tokens)[None, :, None]
        image_rotary_embeddings = QwenTransformer._compute_rotary_embeddings(
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            pos_embed=self.pos_embed,
            config=config,
            cond_image_grid=cond_image_grid,
        )
        for idx, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = QwenTransformer._apply_transformer_block(
                idx=idx,
                block=block,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                text_embeddings=text_embeddings,
                image_rotary_embeddings=image_rotary_embeddings,
                cond_embeddings=cond_embeddings,
                is_cond_token=is_cond_token,
            )
        hidden_states = self.norm_out(hidden_states, text_embeddings)
        hidden_states = self.proj_out(hidden_states)
        return hidden_states

    @staticmethod
    def _apply_transformer_block(
        idx: int,
        block: QwenTransformerBlock,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        encoder_hidden_states_mask: mx.array,
        text_embeddings: mx.array,
        image_rotary_embeddings: tuple[mx.array, mx.array],
        cond_embeddings: mx.array | None = None,
        is_cond_token: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        return block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            text_embeddings=text_embeddings,
            image_rotary_emb=image_rotary_embeddings,
            block_idx=idx,
            cond_embeddings=cond_embeddings,
            is_cond_token=is_cond_token,
        )

    @staticmethod
    def _compute_timestep(
        t: int | float,
        config: Config,
    ) -> mx.array:
        if isinstance(t, int):
            if t < len(config.scheduler.sigmas):
                timestep_idx = t
                time_step = config.scheduler.sigmas[timestep_idx]
            else:
                timestep_idx = None
                for idx, ts in enumerate(config.scheduler.timesteps):
                    if abs(int(ts.item()) - t) < 1:
                        timestep_idx = idx
                        break
                if timestep_idx is None:
                    time_step = t / 1000.0
                else:
                    time_step = config.scheduler.sigmas[timestep_idx]
        else:
            timestep_idx = None
            time_step = t

        timestep = mx.array(np.full((1,), time_step, dtype=np.float32))
        return timestep

    @staticmethod
    def _compute_rotary_embeddings(
        encoder_hidden_states_mask: mx.array,
        pos_embed: QwenEmbedRopeMLX,
        config: Config,
        cond_image_grid: tuple[int, int, int] | list[tuple[int, int, int]] | None = None,
    ) -> tuple[mx.array, mx.array]:
        latent_height = config.height // 16
        latent_width = config.width // 16

        if cond_image_grid is None:
            img_shapes = [(1, latent_height, latent_width)]
        else:
            if isinstance(cond_image_grid, list):
                img_shapes = [(1, latent_height, latent_width)] + cond_image_grid
            else:
                img_shapes = [(1, latent_height, latent_width), cond_image_grid]

        txt_seq_lens = [int(mx.sum(encoder_hidden_states_mask[i]).item()) for i in range(encoder_hidden_states_mask.shape[0])]  # fmt: off
        img_rotary_emb, txt_rotary_emb = pos_embed(video_fhw=img_shapes, txt_seq_lens=txt_seq_lens)
        return img_rotary_emb, txt_rotary_emb
