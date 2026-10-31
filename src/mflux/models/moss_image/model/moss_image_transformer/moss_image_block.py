import mlx.core as mx
from mlx import nn

from mflux.models.moss_image.model.moss_image_transformer.moss_image_rope import MOSSImageRoPE


class MOSSImageAttention:
    @staticmethod
    def attend(q: mx.array, k: mx.array, v: mx.array, rope: mx.array, heads: int) -> mx.array:
        # q, k: (B, L, heads, head_dim) after the QK norm; v: (B, L, dim)
        batch, seq_len, _ = v.shape
        q = MOSSImageRoPE.rotate(q, rope).transpose(0, 2, 1, 3)
        k = MOSSImageRoPE.rotate(k, rope).transpose(0, 2, 1, 3)
        v = v.reshape(batch, seq_len, heads, -1).transpose(0, 2, 1, 3)
        scale = q.shape[-1] ** -0.5
        out = mx.fast.scaled_dot_product_attention(q.astype(v.dtype), k.astype(v.dtype), v, scale=scale)
        return out.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)

    @staticmethod
    @mx.compile  # one kernel instead of a chain of elementwise passes over the (B, L, ffn_dim) activations
    def situ(hidden: mx.array, gate: mx.array) -> mx.array:
        return (4.0 * mx.tanh(gate / 4.0) * mx.sigmoid(gate)) * (25.0 * mx.tanh(hidden / 25.0))


class MOSSImageBlock(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, heads: int):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.proj = {
            "q": nn.Linear(dim, dim, bias=False),
            "k": nn.Linear(dim, dim, bias=False),
            "v": nn.Linear(dim, dim, bias=False),
            "o": nn.Linear(dim, dim, bias=False),
            "ffn_up_gate": nn.Linear(dim, 2 * ffn_dim, bias=False),
            "ffn_down": nn.Linear(ffn_dim, dim, bias=False),
        }

    def __call__(self, x: mx.array, rope: mx.array, modulation: mx.array) -> mx.array:
        # modulation: (B, 6, dim) as shift, scale, gate for attention, then for the FFN
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = (modulation[:, i : i + 1] for i in range(6))
        batch, seq_len, _ = x.shape

        y = self.norm1(x) * (1 + scale_a) + shift_a
        q = self.q_norm(self.proj["q"](y).reshape(batch, seq_len, self.heads, -1))
        k = self.k_norm(self.proj["k"](y).reshape(batch, seq_len, self.heads, -1))
        attn = MOSSImageAttention.attend(q, k, self.proj["v"](y), rope, self.heads)
        x = x + self.proj["o"](attn) * gate_a

        y = self.norm2(x) * (1 + scale_f) + shift_f
        hidden, gate = mx.split(self.proj["ffn_up_gate"](y), 2, axis=-1)
        return x + self.proj["ffn_down"](MOSSImageAttention.situ(hidden, gate)) * gate_f


class MOSSImageTextBlock(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, heads: int):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate = nn.Linear(dim, ffn_dim)
        self.ffn_up = nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def __call__(self, x: mx.array, rope: mx.array) -> mx.array:
        batch, seq_len, _ = x.shape
        q, k, v = mx.split(self.qkv(self.norm1(x)), 3, axis=-1)
        q = self.q_norm(q.reshape(batch, seq_len, self.heads, -1))
        k = self.k_norm(k.reshape(batch, seq_len, self.heads, -1))
        x = x + self.proj(MOSSImageAttention.attend(q, k, v, rope, self.heads))
        y = self.norm2(x)
        return x + self.ffn_down(MOSSImageAttention.situ(self.ffn_up(y), self.ffn_gate(y)))
