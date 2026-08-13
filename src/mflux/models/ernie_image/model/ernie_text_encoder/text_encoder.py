import mlx.core as mx
from mlx import nn
from mlx.core.fast import rms_norm, scaled_dot_product_attention


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return rms_norm(x, self.weight, self.eps)


class _MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class _Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
    ):
        super().__init__()
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.num_kv_groups = num_attention_heads // num_key_value_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)

    def __call__(self, x: mx.array, mask: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        B, T, _ = x.shape

        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        q, k = _apply_rotary(q, k, cos, sin)

        if self.num_kv_groups > 1:
            k = mx.repeat(k, self.num_kv_groups, axis=1)
            v = mx.repeat(v, self.num_kv_groups, axis=1)

        out = scaled_dot_product_attention(
            q.astype(mx.float32),
            k.astype(mx.float32),
            v.astype(mx.float32),
            scale=self.scale,
            mask=mask,
        ).astype(x.dtype)

        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.o_proj(out)


def _apply_rotary(q, k, cos, sin):
    def rotate_half(x):
        half = x.shape[-1] // 2
        return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)

    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


class _DecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
    ):
        super().__init__()
        self.input_layernorm = _RMSNorm(hidden_size, rms_norm_eps)
        self.self_attn = _Attention(hidden_size, num_attention_heads, num_key_value_heads, head_dim)
        self.post_attention_layernorm = _RMSNorm(hidden_size, rms_norm_eps)
        self.mlp = _MLP(hidden_size, intermediate_size)

    def __call__(self, x: mx.array, mask: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask, cos, sin)
        return h + self.mlp(self.post_attention_layernorm(h))


class _TextModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        intermediate_size: int,
        max_position_embeddings: int,
        rope_theta: float,
        rms_norm_eps: float,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [
            _DecoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                intermediate_size=intermediate_size,
                rms_norm_eps=rms_norm_eps,
            )
            for _ in range(num_hidden_layers)
        ]
        self.norm = _RMSNorm(hidden_size, rms_norm_eps)

        self._head_dim = head_dim
        self._rope_theta = rope_theta
        self._max_pos = max_position_embeddings
        inv_freq = 1.0 / (rope_theta ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
        self._inv_freq = inv_freq

    def _get_rotary(self, seq_len: int, dtype) -> tuple[mx.array, mx.array]:
        positions = mx.arange(seq_len, dtype=mx.float32)
        freqs = positions[:, None] * self._inv_freq[None, :]
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb)[None, None, :, :].astype(dtype)
        sin = mx.sin(emb)[None, None, :, :].astype(dtype)
        return cos, sin


class _LanguageModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = _TextModel(**kwargs)


class ErnieMistralTextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 131072,
        hidden_size: int = 3072,
        num_hidden_layers: int = 26,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        intermediate_size: int = 9216,
        max_position_embeddings: int = 262144,
        rope_theta: float = 1000000.0,
        rms_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.language_model = _LanguageModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
        )
        self._num_hidden_layers = num_hidden_layers

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        model = self.language_model.model
        B, T = input_ids.shape

        hidden_states = model.embed_tokens(input_ids)

        if attention_mask is None:
            attention_mask = mx.ones((B, T), dtype=mx.int32)

        dtype = hidden_states.dtype
        pad_mask = mx.where(
            attention_mask == 1,
            mx.zeros(attention_mask.shape, dtype=mx.float32),
            mx.full(attention_mask.shape, -1e9, dtype=mx.float32),
        )[:, None, None, :]

        idx = mx.arange(T, dtype=mx.int32)
        causal_mask = mx.where(
            idx[None, :] > idx[:, None],
            mx.full((T, T), -1e9, dtype=mx.float32),
            mx.zeros((T, T), dtype=mx.float32),
        )[None, None, :, :]

        combined_mask = (causal_mask + pad_mask).astype(dtype)

        cos, sin = model._get_rotary(T, dtype)

        second_to_last = hidden_states
        for i, layer in enumerate(model.layers):
            hidden_states = layer(hidden_states, combined_mask, cos, sin)
            if i == self._num_hidden_layers - 2:
                second_to_last = hidden_states

        return second_to_last
