import mlx.core as mx
from mlx import nn

# Ming's connector/ folder: a Qwen2-1.5B decoder stack used as a bidirectional encoder over the
# 256 query-token states (upstream flips every self_attn.is_causal to False). No embeddings or
# lm_head are used; the output is the final-norm hidden state.
HIDDEN_SIZE = 1536
NUM_LAYERS = 28
NUM_HEADS = 12
NUM_KV_HEADS = 2
HEAD_DIM = 128
INTERMEDIATE = 8960
ROPE_THETA = 1000000.0
RMS_EPS = 1e-6


class ConnectorAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN_SIZE, NUM_HEADS * HEAD_DIM, bias=True)
        self.k_proj = nn.Linear(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, bias=True)
        self.v_proj = nn.Linear(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, bias=True)
        self.o_proj = nn.Linear(NUM_HEADS * HEAD_DIM, HIDDEN_SIZE, bias=False)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        q = ConnectorAttention._rope(q, cos, sin)
        k = ConnectorAttention._rope(k, cos, sin)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=HEAD_DIM**-0.5)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, NUM_HEADS * HEAD_DIM))

    @staticmethod
    def _rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        cos = cos.astype(x.dtype)
        sin = sin.astype(x.dtype)
        x1, x2 = x[..., : HEAD_DIM // 2], x[..., HEAD_DIM // 2 :]
        return x * cos + mx.concatenate([-x2, x1], axis=-1) * sin


class ConnectorMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN_SIZE, INTERMEDIATE, bias=False)
        self.up_proj = nn.Linear(HIDDEN_SIZE, INTERMEDIATE, bias=False)
        self.down_proj = nn.Linear(INTERMEDIATE, HIDDEN_SIZE, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class ConnectorLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = ConnectorAttention()
        self.mlp = ConnectorMLP()
        self.input_layernorm = nn.RMSNorm(HIDDEN_SIZE, eps=RMS_EPS)
        self.post_attention_layernorm = nn.RMSNorm(HIDDEN_SIZE, eps=RMS_EPS)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))


class MingConnector(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [ConnectorLayer() for _ in range(NUM_LAYERS)]
        self.norm = nn.RMSNorm(HIDDEN_SIZE, eps=RMS_EPS)

    def __call__(self, x: mx.array) -> mx.array:
        inv_freq = 1.0 / (ROPE_THETA ** (mx.arange(0, HEAD_DIM, 2, dtype=mx.float32) / HEAD_DIM))
        freqs = mx.arange(x.shape[1], dtype=mx.float32)[:, None] * inv_freq[None, :]
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos, sin = mx.cos(emb), mx.sin(emb)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.norm(x)
