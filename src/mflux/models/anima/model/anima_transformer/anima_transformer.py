import mlx.core as mx
from mlx import nn

from mflux.models.anima.model.anima_transformer.anima_attention import AnimaAttention
from mflux.models.anima.model.anima_transformer.anima_block import AnimaBlock, AnimaFeedForward
from mflux.models.anima.model.anima_transformer.anima_embedders import (
    AnimaFinalLayer,
    AnimaPatchEmbed,
    AnimaTimestepEmbedding,
    AnimaTimesteps,
)
from mflux.models.anima.model.anima_transformer.anima_llm_adapter import AnimaLLMAdapter
from mflux.models.anima.model.anima_transformer.anima_rope import AnimaRoPE
from mflux.models.common.compute_precision import ComputePrecision


class AnimaTransformer(nn.Module):
    # The Cosmos-Predict2 2B DiT (MiniTrainDIT) with Anima's LLM adapter. Module names follow the ComfyUI
    # checkpoint (circlestone-labs/Anima, split_files/diffusion_models), so the weight mapping is the identity
    # once the file's key prefix is dropped.
    MIN_CONTEXT_LENGTH = 512

    def __init__(
        self,
        dim: int = 2048,
        num_blocks: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        latent_channels: int = 16,
        context_dim: int = 1024,
        patch_size: int = 2,
        adaln_lora_dim: int = 256,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.latent_channels = latent_channels
        self.x_embedder = AnimaPatchEmbed(latent_channels, dim, patch_size)
        self.t_embedder = [AnimaTimesteps(dim), AnimaTimestepEmbedding(dim)]
        self.t_embedding_norm = nn.RMSNorm(dim, eps=1e-6)
        self.blocks = [AnimaBlock(dim, context_dim, num_heads, mlp_ratio, adaln_lora_dim) for _ in range(num_blocks)]
        self.final_layer = AnimaFinalLayer(dim, patch_size**2 * latent_channels, adaln_lora_dim)
        self.llm_adapter = AnimaLLMAdapter(dim=context_dim)
        self.rope = AnimaRoPE(dim // num_heads)

    def __call__(self, latents: mx.array, sigma: mx.array, context: mx.array) -> mx.array:
        # latents: (B, 16, H, W); sigma: (B,) flow time in [0, 1]; context: (B, S, 1024) from encode_text.
        # Returns the flow velocity, shaped like the latents.
        batch, channels, height, width = latents.shape
        dtype = context.dtype
        x = self.x_embedder(latents.astype(dtype))
        emb = self.t_embedder[0](sigma).astype(dtype)
        adaln_lora = self.t_embedder[1](emb)[:, None]
        emb = self.t_embedding_norm(emb)[:, None]
        rope = self.rope(height // self.patch_size, width // self.patch_size)
        for block in self.blocks:
            x = block(x, emb, adaln_lora, context, rope)
        x = self.final_layer(x, emb, adaln_lora)
        return self._unpatchify(x, channels, height, width)

    def apply_compute_precision(self, precision: ComputePrecision) -> None:
        # Only the blocks: the LLM adapter has its own attention class and keeps the model's precision.
        precision.apply(self, (AnimaAttention, AnimaFeedForward))

    def encode_text(self, hidden_states: mx.array, t5_ids: mx.array) -> mx.array:
        # The context the blocks attend to: the adapter output, zero-padded to 512 tokens like the reference.
        # The padding is part of the conditioning: the zero keys still take a share of every softmax.
        # The adapter runs in the dtype of the hidden states; the context comes back in the blocks' dtype.
        context = self.llm_adapter(hidden_states, t5_ids)
        missing = AnimaTransformer.MIN_CONTEXT_LENGTH - context.shape[1]
        if missing > 0:
            context = mx.pad(context, [(0, 0), (0, missing), (0, 0)])
        return context.astype(self.t_embedding_norm.weight.dtype)

    def _unpatchify(self, x: mx.array, channels: int, height: int, width: int) -> mx.array:
        # Output features are ordered (py, px, c), unlike the (c, py, px) input patches.
        p = self.patch_size
        x = x.reshape(x.shape[0], height // p, width // p, p, p, channels).transpose(0, 5, 1, 3, 2, 4)
        return x.reshape(x.shape[0], channels, height, width)
