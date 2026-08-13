import mlx.core as mx
from mlx import nn

from mflux.models.lens.model.transformer.lens_transformer_block import LensTransformerBlock
from mflux.models.qwen.model.qwen_transformer.qwen_rope import QwenEmbedRopeMLX
from mflux.models.qwen.model.qwen_transformer.qwen_timestep_embedding import QwenTimestepEmbedding
from mflux.models.qwen.model.qwen_transformer.qwen_timesteps import QwenTimesteps


class _LensTimeTextEmbed(nn.Module):
    def __init__(self, inner_dim: int):
        super().__init__()
        self.time_proj = QwenTimesteps(proj_dim=256, scale=1000.0)
        self.timestep_embedder = QwenTimestepEmbedding(proj_dim=256, inner_dim=inner_dim)

    def __call__(self, timestep: mx.array, dtype) -> mx.array:
        return self.timestep_embedder(self.time_proj(timestep).astype(dtype))


class LensTransformer(nn.Module):
    """Lens dual-stream MMDiT: 48 blocks, inner_dim 1536, multi-layer GPT-OSS text.

    Same rope semantics as Qwen-Image (centered h/w halves, text offset at
    max(h//2, w//2)) with axes (8, 28, 28); the joint sequence puts the image
    stream first. norm_out is a standard AdaLayerNormContinuous: scale first,
    the opposite of Flux's LastLayer.
    """

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 128,
        out_channels: int = 32,
        num_layers: int = 48,
        attention_head_dim: int = 64,
        num_attention_heads: int = 24,
        enc_hidden_dim: int = 2880,
        num_selected_layers: int = 4,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = QwenEmbedRopeMLX(theta=10000, axes_dim=[8, 28, 28], scale_rope=True)
        self.time_text_embed = _LensTimeTextEmbed(self.inner_dim)

        self.txt_norm = [nn.RMSNorm(enc_hidden_dim, eps=1e-5) for _ in range(num_selected_layers)]
        self.txt_in = nn.Linear(enc_hidden_dim * num_selected_layers, self.inner_dim, bias=True)
        self.img_in = nn.Linear(in_channels, self.inner_dim, bias=True)

        self.transformer_blocks = [
            LensTransformerBlock(dim=self.inner_dim, num_heads=num_attention_heads, head_dim=attention_head_dim)
            for _ in range(num_layers)
        ]

        self.norm_out = _LensAdaLayerNormContinuous(self.inner_dim)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * out_channels, bias=True)

    def __call__(
        self,
        hidden_states: mx.array,  # [B, h*w, in_channels] packed latents
        encoder_layers: mx.array,  # [B, S, L, enc_hidden_dim] from the encoder
        timestep: mx.array,  # [B], sigma in [0, 1]
        latent_height: int,  # patch-grid height (pixels / 16)
        latent_width: int,
    ) -> mx.array:
        text_seq_len = encoder_layers.shape[1]

        hidden_states = self.img_in(hidden_states)

        normed = [self.txt_norm[i](encoder_layers[:, :, i]) for i in range(len(self.txt_norm))]
        encoder_hidden_states = self.txt_in(mx.concatenate(normed, axis=-1))
        encoder_hidden_states = encoder_hidden_states.astype(hidden_states.dtype)

        temb = self.time_text_embed(timestep, hidden_states.dtype)

        image_rotary_emb, text_rotary_emb = self.pos_embed((1, latent_height, latent_width), [text_seq_len])

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                text_rotary_emb=text_rotary_emb,
            )

        hidden_states = self.norm_out(hidden_states, temb)
        return self.proj_out(hidden_states)  # [B, h*w, patch^2 * out_channels]


class _LensAdaLayerNormContinuous(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, 2 * dim, bias=True)
        self.norm = nn.LayerNorm(dim, eps=1e-6, affine=False)

    def __call__(self, x: mx.array, conditioning: mx.array) -> mx.array:
        emb = self.linear(nn.silu(conditioning))
        scale, shift = mx.split(emb, 2, axis=-1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]
