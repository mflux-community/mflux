import mlx.core as mx
from mlx import nn

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux.model.flux_transformer.common.attention_utils import AttentionUtils
from mflux.models.flux2.model.flux2_transformer.feed_forward import Flux2SwiGLU
from mflux.models.flux2.model.flux2_transformer.flux2_kv_cache import Flux2KVCache


class Flux2ParallelSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, mlp_ratio: float = 3.0):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.to_qkv_mlp_proj = nn.Linear(dim, self.inner_dim * 3 + self.mlp_hidden_dim * 2, bias=False)
        self.norm_q = nn.RMSNorm(dim_head, eps=1e-5)
        self.norm_k = nn.RMSNorm(dim_head, eps=1e-5)
        self.mlp_act = Flux2SwiGLU()
        self.to_out = nn.Linear(self.inner_dim + self.mlp_hidden_dim, dim, bias=False)

    def __call__(
        self,
        hidden_states: mx.array,
        image_rotary_emb,
        kv_cache: Flux2KVCache | None = None,
        kv_cache_layer_idx: int | None = None,
    ):
        proj = self.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden = mx.split(proj, [self.inner_dim * 3], axis=-1)
        query, key, value = mx.split(qkv, 3, axis=-1)

        batch, seq_len, _ = query.shape
        query = mx.transpose(mx.reshape(query, (batch, seq_len, self.heads, self.dim_head)), (0, 2, 1, 3))
        key = mx.transpose(mx.reshape(key, (batch, seq_len, self.heads, self.dim_head)), (0, 2, 1, 3))
        value = mx.transpose(mx.reshape(value, (batch, seq_len, self.heads, self.dim_head)), (0, 2, 1, 3))

        query = self.norm_q(query.astype(mx.float32)).astype(ModelConfig.precision)
        key = self.norm_k(key.astype(mx.float32)).astype(ModelConfig.precision)

        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            query, key = AttentionUtils.apply_rope_bshd(query, key, cos, sin)

        if kv_cache is not None:
            kv_cache.store_reference("single", kv_cache_layer_idx, key, value)
            key, value = kv_cache.append_reference("single", kv_cache_layer_idx, key, value)
            hidden_states = kv_cache.compute_extract_attention(
                query=query,
                key=key,
                value=value,
                batch_size=batch,
                num_heads=self.heads,
                head_dim=self.dim_head,
            )
        else:
            hidden_states = AttentionUtils.compute_attention(
                query=query,
                key=key,
                value=value,
                batch_size=batch,
                num_heads=self.heads,
                head_dim=self.dim_head,
            )

        mlp_hidden = self.mlp_act(mlp_hidden)
        hidden_states = mx.concatenate([hidden_states, mlp_hidden], axis=-1)
        hidden_states = self.to_out(hidden_states)
        return hidden_states
