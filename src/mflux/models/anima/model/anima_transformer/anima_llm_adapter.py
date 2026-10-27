import mlx.core as mx
from mlx import nn


class AnimaAdapterAttention(nn.Module):
    # Both the query and the key get a 1D rotary embedding over their own positions, also in cross-attention.
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)

    def __call__(self, x: mx.array, context: mx.array) -> mx.array:
        batch, seq_len, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(batch, seq_len, self.num_heads, -1)).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(context).reshape(batch, context.shape[1], self.num_heads, -1)).transpose(0, 2, 1, 3)
        v = self.v_proj(context).reshape(batch, context.shape[1], self.num_heads, -1).transpose(0, 2, 1, 3)
        q = mx.fast.rope(q, q.shape[-1], traditional=False, base=10000.0, scale=1.0, offset=0)
        k = mx.fast.rope(k, k.shape[-1], traditional=False, base=10000.0, scale=1.0, offset=0)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1))


class AnimaAdapterBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm_self_attn = nn.RMSNorm(dim, eps=1e-6)
        self.self_attn = AnimaAdapterAttention(dim, num_heads)
        self.norm_cross_attn = nn.RMSNorm(dim, eps=1e-6)
        self.cross_attn = AnimaAdapterAttention(dim, num_heads)
        self.norm_mlp = nn.RMSNorm(dim, eps=1e-6)
        self.mlp = [nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim)]

    def __call__(self, x: mx.array, source: mx.array) -> mx.array:
        h = self.norm_self_attn(x)
        x = x + self.self_attn(h, h)
        x = x + self.cross_attn(self.norm_cross_attn(x), source)
        h = self.norm_mlp(x)
        for layer in self.mlp:
            h = layer(h)
        return x + h


class AnimaLLMAdapter(nn.Module):
    # Turns the Qwen3 hidden states into the transformer's text context: embeddings of the prompt's T5 token ids
    # attend to the Qwen3 states through six blocks. The output has one vector per T5 token.
    def __init__(self, vocab_size: int = 32128, dim: int = 1024, num_layers: int = 6, num_heads: int = 16, mlp_ratio: float = 4.0):  # fmt: off
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = [AnimaAdapterBlock(dim, num_heads, mlp_ratio) for _ in range(num_layers)]
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.RMSNorm(dim, eps=1e-6)

    def __call__(self, source: mx.array, target_ids: mx.array) -> mx.array:
        # source: (B, S, dim) Qwen3 hidden states; target_ids: (B, T) T5 token ids
        x = self.embed(target_ids).astype(source.dtype)
        for block in self.blocks:
            x = block(x, source)
        return self.norm(self.out_proj(x))
