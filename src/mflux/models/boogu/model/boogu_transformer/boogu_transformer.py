from __future__ import annotations

import mlx.core as mx
from mlx import nn

from mflux.models.boogu.model.boogu_transformer.boogu_blocks import (
    BooguImageDoubleStreamTransformerBlock,
    BooguImageTransformerBlock,
)
from mflux.models.boogu.model.boogu_transformer.boogu_embeddings import (
    Lumina2CombinedTimestepCaptionEmbedding,
    LuminaLayerNormContinuous,
)
from mflux.models.boogu.model.boogu_transformer.boogu_rope import BooguImageRoPE


class BooguImageTransformer(nn.Module):
    """Boogu-Image mixed single/double-stream diffusion transformer (NextDiT lineage).

    Forward topology (Turbo T2I, batch size 1, no reference images):
    ``patchify -> context refiner (instruction) + noise refiner (image) ->
    double-stream blocks -> fuse [instruction, image] -> single-stream blocks ->
    norm_out -> unpatchify`` producing a velocity prediction with the latent shape.

    Args:
        patch_size: Latent patch size.
        in_channels: Latent channel count (16, FLUX VAE).
        hidden_size: Model dimension.
        num_layers: Total transformer layers (double + single stream).
        num_double_stream_layers: Number of leading double-stream layers.
        num_refiner_layers: Number of refiner layers (per refiner family).
        num_attention_heads: Query head count.
        num_kv_heads: Key/value head count (GQA).
        multiple_of: FFN intermediate rounding.
        ffn_dim_multiplier: Optional FFN multiplier.
        norm_eps: RMSNorm epsilon.
        axes_dim_rope: Per-axis RoPE dimension (sums to head_dim).
        axes_lens: Per-axis RoPE table length.
        instruction_feat_dim: Qwen3-VL hidden size feeding the caption embedder.
        timestep_scale: Scale applied inside the sinusoidal timestep projection.
    """

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        hidden_size: int = 3360,
        num_layers: int = 40,
        num_double_stream_layers: int = 8,
        num_refiner_layers: int = 2,
        num_attention_heads: int = 28,
        num_kv_heads: int = 7,
        multiple_of: int = 256,
        ffn_dim_multiplier: float | None = None,
        norm_eps: float = 1e-5,
        axes_dim_rope: tuple[int, int, int] = (40, 40, 40),
        axes_lens: tuple[int, int, int] = (2048, 1664, 1664),
        instruction_feat_dim: int = 4096,
        timestep_scale: float = 1000.0,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_single_stream_layers = num_layers - num_double_stream_layers

        self.rope_embedder = BooguImageRoPE(
            theta=10000,
            axes_dim=axes_dim_rope,
            axes_lens=axes_lens,
            patch_size=patch_size,
        )

        self.x_embedder = nn.Linear(patch_size * patch_size * in_channels, hidden_size)
        self.ref_image_patch_embedder = nn.Linear(patch_size * patch_size * in_channels, hidden_size)

        self.time_caption_embed = Lumina2CombinedTimestepCaptionEmbedding(
            hidden_size=hidden_size,
            instruction_feat_dim=instruction_feat_dim,
            norm_eps=norm_eps,
            timestep_scale=timestep_scale,
        )

        def _base_block(modulation: bool) -> BooguImageTransformerBlock:
            return BooguImageTransformerBlock(
                hidden_size, num_attention_heads, num_kv_heads, multiple_of, ffn_dim_multiplier, norm_eps, modulation
            )

        self.noise_refiner = [_base_block(True) for _ in range(num_refiner_layers)]
        self.ref_image_refiner = [_base_block(True) for _ in range(num_refiner_layers)]
        self.context_refiner = [_base_block(False) for _ in range(num_refiner_layers)]

        self.double_stream_layers = [
            BooguImageDoubleStreamTransformerBlock(
                hidden_size, num_attention_heads, num_kv_heads, multiple_of, ffn_dim_multiplier, norm_eps
            )
            for _ in range(num_double_stream_layers)
        ]
        self.single_stream_layers = [_base_block(True) for _ in range(self.num_single_stream_layers)]

        self.norm_out = LuminaLayerNormContinuous(
            embedding_dim=hidden_size,
            conditioning_embedding_dim=min(hidden_size, 1024),
            eps=1e-6,
            out_dim=patch_size * patch_size * self.out_channels,
        )

        # Reference-image index embedding (kept for weight-loading parity; unused for T2I).
        self.image_index_embedding = mx.zeros((5, hidden_size))

    def _patchify(self, latent: mx.array) -> mx.array:
        """``(B, C, H, W)`` -> ``(B, (H/p)(W/p), p*p*C)``."""
        p = self.patch_size
        b, c, h, w = latent.shape
        x = mx.reshape(latent, (b, c, h // p, p, w // p, p))
        x = mx.transpose(x, (0, 2, 4, 3, 5, 1))
        return mx.reshape(x, (b, (h // p) * (w // p), p * p * c))

    def _unpatchify(self, tokens: mx.array, h_tokens: int, w_tokens: int) -> mx.array:
        """``(B, (H/p)(W/p), p*p*C)`` -> ``(B, C, H, W)``."""
        p = self.patch_size
        b = tokens.shape[0]
        c = self.out_channels
        x = mx.reshape(tokens, (b, h_tokens, w_tokens, p, p, c))
        x = mx.transpose(x, (0, 5, 1, 3, 2, 4))
        return mx.reshape(x, (b, c, h_tokens * p, w_tokens * p))

    def __call__(
        self,
        hidden_states: mx.array,
        timestep: mx.array,
        instruction_hidden_states: mx.array,
    ) -> mx.array:
        """Predict the flow-matching velocity.

        Args:
            hidden_states: Latent ``(B, C, H_lat, W_lat)``.
            timestep: ``(B,)`` sigma values.
            instruction_hidden_states: Qwen3-VL features ``(B, L_instruct, instruction_feat_dim)``.

        Returns:
            Velocity prediction ``(B, C, H_lat, W_lat)``.
        """
        p = self.patch_size
        _, _, h_lat, w_lat = hidden_states.shape
        h_tokens, w_tokens = h_lat // p, w_lat // p
        cap_len = instruction_hidden_states.shape[1]

        # Timestep + caption conditioning.
        temb, instruction_hidden_states = self.time_caption_embed(timestep, instruction_hidden_states)

        # Patchify image latent and build rotaries.
        image_tokens = self._patchify(hidden_states)
        rope = self.rope_embedder(cap_len, h_tokens, w_tokens)

        # Context refinement (instruction stream, no modulation).
        for layer in self.context_refiner:
            instruction_hidden_states = layer(instruction_hidden_states, rope.context)

        # Image patch embed + noise refinement.
        image_hidden_states = self.x_embedder(image_tokens)
        for layer in self.noise_refiner:
            image_hidden_states = layer(image_hidden_states, rope.image, temb)

        # Double-stream stage.
        instruct_hidden_states = instruction_hidden_states
        img_hidden_states = image_hidden_states
        for layer in self.double_stream_layers:
            img_hidden_states, instruct_hidden_states = layer(
                img_hidden_states, instruct_hidden_states, rope.image, rope.joint, temb
            )

        # Fuse to the joint sequence [instruction, image] and run single-stream blocks.
        hidden_states = mx.concatenate([instruct_hidden_states, img_hidden_states], axis=1)
        for layer in self.single_stream_layers:
            hidden_states = layer(hidden_states, rope.joint, temb)

        # Output projection + unpatchify (image tokens are the trailing segment).
        hidden_states = self.norm_out(hidden_states, temb)
        image_out = hidden_states[:, cap_len:]
        return self._unpatchify(image_out, h_tokens, w_tokens)
