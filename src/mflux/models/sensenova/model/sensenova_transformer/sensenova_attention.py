import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention

from mflux.models.sensenova.model.sensenova_transformer.sensenova_rms_norm import SenseNovaRMSNorm
from mflux.models.sensenova.model.sensenova_transformer.sensenova_rope import SenseNovaRope


class SenseNovaAttention(nn.Module):
    # One class for both weight streams; the checkpoint's *_mot_gen tensors and the und tensors fill the same names
    def __init__(self, hidden_size: int = 4096, num_heads: int = 32, num_kv_heads: int = 8, head_dim: int = 128, eps: float = 1e-6):  # fmt: skip
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = SenseNovaRMSNorm(head_dim // 2, eps)
        self.q_norm_hw = SenseNovaRMSNorm(head_dim // 2, eps)
        self.k_norm = SenseNovaRMSNorm(head_dim // 2, eps)
        self.k_norm_hw = SenseNovaRMSNorm(head_dim // 2, eps)

    def project(self, x: mx.array, rope: list[tuple[mx.array, mx.array]]) -> tuple[mx.array, mx.array, mx.array]:
        # q [B, heads, S, D], k and v [B, kv_heads, S, D]; k is post-norm and post-RoPE, as the prefix cache stores it
        batch, seq, _ = x.shape
        half = self.head_dim // 2
        q = self.q_proj(x).reshape(batch, seq, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, seq, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, seq, self.num_kv_heads, self.head_dim)
        q = mx.concatenate([self.q_norm(q[..., :half]), self.q_norm_hw(q[..., half:])], axis=-1)
        k = mx.concatenate([self.k_norm(k[..., :half]), self.k_norm_hw(k[..., half:])], axis=-1)
        q = SenseNovaRope.apply(q, rope)
        k = SenseNovaRope.apply(k, rope)
        return q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)

    def __call__(
        self,
        x: mx.array,
        rope: list[tuple[mx.array, mx.array]],
        prefix_kv: tuple[mx.array, mx.array] | None = None,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        batch, seq, _ = x.shape
        q, k, v = self.project(x, rope)
        if prefix_kv is not None:
            k = mx.concatenate([prefix_kv[0].astype(k.dtype), k], axis=2)
            v = mx.concatenate([prefix_kv[1].astype(v.dtype), v], axis=2)
        out = scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(batch, seq, -1))
