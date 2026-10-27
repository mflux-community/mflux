import mlx.core as mx
from mlx import nn

from mflux.models.common_models.qwen3_5.qwen3_5_rms_norm import Qwen35GatedRMSNorm


class Qwen35CausalConv1d(nn.Module):
    # Depthwise causal convolution. The weight keeps the checkpoint layout (channels, 1, kernel).
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = mx.zeros((channels, 1, kernel_size))

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, L, C)
        seq_len = x.shape[1]
        padded = mx.pad(x, [(0, 0), (self.kernel_size - 1, 0), (0, 0)])
        weight = self.weight[:, 0, :].astype(mx.float32)
        out = mx.zeros(x.shape, dtype=mx.float32)
        for tap in range(self.kernel_size):
            out = out + padded[:, tap : tap + seq_len, :].astype(mx.float32) * weight[:, tap]
        return out.astype(x.dtype)


class Qwen35GatedDeltaNet(nn.Module):
    CHUNK_SIZE = 64

    def __init__(
        self,
        hidden_size: int,
        num_key_heads: int,
        num_value_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        conv_kernel_size: int,
        rms_norm_eps: float,
    ):
        super().__init__()
        self.num_key_heads = num_key_heads
        self.num_value_heads = num_value_heads
        self.key_head_dim = key_head_dim
        self.value_head_dim = value_head_dim
        self.key_dim = key_head_dim * num_key_heads
        self.value_dim = value_head_dim * num_value_heads

        conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = Qwen35CausalConv1d(conv_dim, conv_kernel_size)
        self.dt_bias = mx.ones((num_value_heads,))
        self.A_log = mx.zeros((num_value_heads,))
        self.norm = Qwen35GatedRMSNorm(value_head_dim, eps=rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        self.in_proj_qkv = nn.Linear(hidden_size, conv_dim, bias=False)
        self.in_proj_z = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(hidden_size, num_value_heads, bias=False)
        self.in_proj_a = nn.Linear(hidden_size, num_value_heads, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        batch, seq_len, _ = hidden_states.shape

        mixed_qkv = nn.silu(self.conv1d(self.in_proj_qkv(hidden_states)))
        query, key, value = mx.split(mixed_qkv, [self.key_dim, 2 * self.key_dim], axis=-1)
        query = query.reshape(batch, seq_len, self.num_key_heads, self.key_head_dim)
        key = key.reshape(batch, seq_len, self.num_key_heads, self.key_head_dim)
        value = value.reshape(batch, seq_len, self.num_value_heads, self.value_head_dim)

        z = self.in_proj_z(hidden_states).reshape(batch, seq_len, self.num_value_heads, self.value_head_dim)
        beta = mx.sigmoid(self.in_proj_b(hidden_states).astype(mx.float32))
        a = self.in_proj_a(hidden_states).astype(mx.float32)
        g = -mx.exp(self.A_log.astype(mx.float32)) * nn.softplus(a + self.dt_bias.astype(mx.float32))

        repeats = self.num_value_heads // self.num_key_heads
        if repeats > 1:
            query = mx.repeat(query, repeats, axis=2)
            key = mx.repeat(key, repeats, axis=2)

        core = Qwen35GatedDeltaNet._gated_delta_rule(query, key, value, g, beta).astype(hidden_states.dtype)
        core = self.norm(core, z).reshape(batch, seq_len, self.value_dim)
        return self.out_proj(core)

    @staticmethod
    def _gated_delta_rule(query: mx.array, key: mx.array, value: mx.array, g: mx.array, beta: mx.array) -> mx.array:
        # Chunked form of the gated delta rule, in float32, as in the reference. Inside a chunk the
        # update is a few matrix products. Only the state between chunks is serial, so a 4096-token
        # prompt takes 64 serial steps for each layer, not 4096.
        chunk = Qwen35GatedDeltaNet.CHUNK_SIZE
        query = Qwen35GatedDeltaNet._l2_normalize(query.astype(mx.float32))
        key = Qwen35GatedDeltaNet._l2_normalize(key.astype(mx.float32))
        # (B, L, H, D) -> (B, H, L, D) and (B, L, H) -> (B, H, L)
        query, key, value = (x.astype(mx.float32).transpose(0, 2, 1, 3) for x in (query, key, value))
        g, beta = (x.astype(mx.float32).transpose(0, 2, 1) for x in (g, beta))

        batch, heads, seq_len, key_dim = key.shape
        value_dim = value.shape[-1]
        pad = (chunk - seq_len % chunk) % chunk
        query, key, value = (mx.pad(x, [(0, 0), (0, 0), (0, pad), (0, 0)]) for x in (query, key, value))
        g, beta = (mx.pad(x, [(0, 0), (0, 0), (0, pad)]) for x in (g, beta))
        chunks = (seq_len + pad) // chunk

        query = query * (key_dim**-0.5)
        value_beta = value * beta[..., None]
        key_beta = key * beta[..., None]
        query, key, value_beta, key_beta = (
            x.reshape(batch, heads, chunks, chunk, -1) for x in (query, key, value_beta, key_beta)
        )
        g = mx.cumsum(g.reshape(batch, heads, chunks, chunk), axis=-1)

        lower = mx.tril(mx.ones((chunk, chunk), dtype=mx.bool_))
        strictly_lower = mx.tril(mx.ones((chunk, chunk), dtype=mx.bool_), k=-1)
        # Mask before exp: above the diagonal the difference is positive and can overflow.
        decay = mx.where(lower, mx.exp(mx.where(lower, g[..., :, None] - g[..., None, :], 0.0)), 0.0)

        # Solve (I - A) T = I row by row (forward substitution), as the reference does.
        # Do not use the product (I + A)(I + A^2)(I + A^4)... here. When the keys in a chunk
        # are similar, the powers of A become very large, float32 cannot cancel them and the
        # result becomes inf or NaN (issue #920).
        a = mx.where(strictly_lower, -((key_beta @ key.swapaxes(-1, -2)) * decay), 0.0)
        rows = [a[..., :1, :]]
        for i in range(1, chunk):
            row = a[..., i : i + 1, :]
            rows.append(row + row[..., :i] @ mx.concatenate(rows, axis=-2))
        inverse = mx.concatenate(rows, axis=-2) + mx.eye(chunk, dtype=mx.float32)

        value_chunks = inverse @ value_beta
        key_cumulative_decay = inverse @ (key_beta * mx.exp(g)[..., None])

        state = mx.zeros((batch, heads, key_dim, value_dim), dtype=mx.float32)
        outputs = []
        for i in range(chunks):
            q_i, k_i, g_i = query[:, :, i], key[:, :, i], g[:, :, i]
            attn = (q_i @ k_i.swapaxes(-1, -2)) * decay[:, :, i]
            v_new = value_chunks[:, :, i] - key_cumulative_decay[:, :, i] @ state
            outputs.append((q_i * mx.exp(g_i)[..., None]) @ state + attn @ v_new)
            g_last = g_i[..., -1:]
            state = state * mx.exp(g_last)[..., None] + (k_i * mx.exp(g_last - g_i)[..., None]).swapaxes(-1, -2) @ v_new
        out = mx.stack(outputs, axis=2).reshape(batch, heads, chunks * chunk, value_dim)[:, :, :seq_len]
        return out.transpose(0, 2, 1, 3)

    @staticmethod
    def _l2_normalize(x: mx.array, eps: float = 1e-6) -> mx.array:
        return x * mx.rsqrt(mx.sum(x * x, axis=-1, keepdims=True) + eps)
