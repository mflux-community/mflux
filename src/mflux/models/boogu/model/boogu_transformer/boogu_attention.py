from __future__ import annotations

import math

import mlx.core as mx
from mlx import nn

from mflux.models.boogu.model.boogu_transformer.boogu_rope import RotaryEmb, apply_rotary_emb


def _sdpa_gqa(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> mx.array:
    """Grouped-query scaled-dot-product attention with all-valid (no) mask.

    For Turbo T2I with batch size 1 every token is valid, so the reference's
    padding/causal masks reduce to no mask. Key/value heads are repeat-expanded
    to the query head count (matching the reference's explicit ``repeat_interleave``
    to avoid the MATH sdpa backend).

    Args:
        query: ``(B, S, num_heads, head_dim)``.
        key: ``(B, S, num_kv_heads, head_dim)``.
        value: ``(B, S, num_kv_heads, head_dim)``.
        num_heads: Query head count.
        num_kv_heads: Key/value head count.
        head_dim: Per-head dimension.

    Returns:
        ``(B, S, num_heads * head_dim)``.
    """
    batch_size, seq_len = query.shape[0], query.shape[1]

    # (B, S, H, D) -> (B, H, S, D)
    q = mx.transpose(query, (0, 2, 1, 3))
    k = mx.transpose(key, (0, 2, 1, 3))
    v = mx.transpose(value, (0, 2, 1, 3))

    if num_kv_heads < num_heads:
        repeats = num_heads // num_kv_heads
        k = mx.repeat(k, repeats, axis=1)
        v = mx.repeat(v, repeats, axis=1)

    scale = 1.0 / math.sqrt(head_dim)
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)

    out = mx.transpose(out, (0, 2, 1, 3))
    return mx.reshape(out, (batch_size, seq_len, num_heads * head_dim))


class BooguAttention(nn.Module):
    """Standard GQA self-attention with qk-RMSNorm and RoPE.

    Used by the base/refiner/single-stream blocks and by the image self-attention
    inside the double-stream block. Mirrors a diffusers ``Attention`` with
    ``qk_norm="rms_norm"``, GQA (``kv_heads < heads``), no biases, and a single
    ``to_out`` projection.

    Args:
        dim: Model dimension.
        num_attention_heads: Query head count.
        num_kv_heads: Key/value head count (GQA).
        eps: RMSNorm epsilon (diffusers ``Attention`` uses ``1e-5`` here).
    """

    def __init__(self, dim: int, num_attention_heads: int, num_kv_heads: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.heads = num_attention_heads
        self.kv_heads = num_kv_heads
        self.head_dim = dim // num_attention_heads

        self.to_q = nn.Linear(dim, num_attention_heads * self.head_dim, bias=False)
        self.to_k = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.to_v = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.norm_q = nn.RMSNorm(self.head_dim, eps=eps)
        self.norm_k = nn.RMSNorm(self.head_dim, eps=eps)
        self.to_out = [nn.Linear(num_attention_heads * self.head_dim, dim, bias=False)]

    def __call__(self, hidden_states: mx.array, rotary_emb: RotaryEmb) -> mx.array:
        batch_size, seq_len, _ = hidden_states.shape

        query = mx.reshape(self.to_q(hidden_states), (batch_size, seq_len, self.heads, self.head_dim))
        key = mx.reshape(self.to_k(hidden_states), (batch_size, seq_len, self.kv_heads, self.head_dim))
        value = mx.reshape(self.to_v(hidden_states), (batch_size, seq_len, self.kv_heads, self.head_dim))

        query = self.norm_q(query)
        key = self.norm_k(key)
        query = apply_rotary_emb(query, rotary_emb)
        key = apply_rotary_emb(key, rotary_emb)

        out = _sdpa_gqa(query, key, value, self.heads, self.kv_heads, self.head_dim)
        return self.to_out[0](out)


class BooguDoubleStreamProcessor(nn.Module):
    """Processor-owned q/k/v and output projections for the joint attention.

    In the reference, the host ``Attention`` module's ``to_q/to_k/to_v`` are
    deleted and these projections live on the attention *processor*. We keep the
    same nesting (``...img_instruct_attn.processor.img_to_q`` etc.) so the weight
    mapping stays close to 1:1.

    Args:
        head_dim: Per-head dimension.
        num_attention_heads: Query head count.
        num_kv_heads: Key/value head count.
    """

    def __init__(self, head_dim: int, num_attention_heads: int, num_kv_heads: int) -> None:
        super().__init__()
        query_dim = head_dim * num_attention_heads
        kv_dim = head_dim * num_kv_heads
        self.img_to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.img_to_k = nn.Linear(query_dim, kv_dim, bias=False)
        self.img_to_v = nn.Linear(query_dim, kv_dim, bias=False)
        self.instruct_to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.instruct_to_k = nn.Linear(query_dim, kv_dim, bias=False)
        self.instruct_to_v = nn.Linear(query_dim, kv_dim, bias=False)
        self.instruct_out = nn.Linear(query_dim, query_dim, bias=False)
        self.img_out = nn.Linear(query_dim, query_dim, bias=False)


class BooguDoubleStreamJointAttention(nn.Module):
    """Joint image<->instruction self-attention for the double-stream block.

    Instruction and image streams generate separate q/k/v (via the processor),
    which are concatenated ``[instruction, image]``, attended jointly with
    qk-RMSNorm + RoPE, split back, given separate output projections, re-merged,
    and passed through a final shared ``to_out`` projection.

    Returns the full joint ``(B, L_instruct + L_img, dim)`` tensor; the block
    splits it into the two streams (matching the reference).

    Args:
        dim: Model dimension.
        num_attention_heads: Query head count.
        num_kv_heads: Key/value head count.
        eps: RMSNorm epsilon.
    """

    def __init__(self, dim: int, num_attention_heads: int, num_kv_heads: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.heads = num_attention_heads
        self.kv_heads = num_kv_heads
        self.head_dim = dim // num_attention_heads

        self.norm_q = nn.RMSNorm(self.head_dim, eps=eps)
        self.norm_k = nn.RMSNorm(self.head_dim, eps=eps)
        self.to_out = [nn.Linear(num_attention_heads * self.head_dim, dim, bias=False)]
        self.processor = BooguDoubleStreamProcessor(self.head_dim, num_attention_heads, num_kv_heads)

    def __call__(
        self,
        img_hidden_states: mx.array,
        instruct_hidden_states: mx.array,
        rotary_emb: RotaryEmb,
    ) -> mx.array:
        proc = self.processor
        batch_size = img_hidden_states.shape[0]
        l_instruct = instruct_hidden_states.shape[1]

        img_q = proc.img_to_q(img_hidden_states)
        img_k = proc.img_to_k(img_hidden_states)
        img_v = proc.img_to_v(img_hidden_states)
        ins_q = proc.instruct_to_q(instruct_hidden_states)
        ins_k = proc.instruct_to_k(instruct_hidden_states)
        ins_v = proc.instruct_to_v(instruct_hidden_states)

        # Instruction first, then image (matches reference concat order).
        query = mx.concatenate([ins_q, img_q], axis=1)
        key = mx.concatenate([ins_k, img_k], axis=1)
        value = mx.concatenate([ins_v, img_v], axis=1)
        seq_len = query.shape[1]

        query = mx.reshape(query, (batch_size, seq_len, self.heads, self.head_dim))
        key = mx.reshape(key, (batch_size, seq_len, self.kv_heads, self.head_dim))
        value = mx.reshape(value, (batch_size, seq_len, self.kv_heads, self.head_dim))

        query = self.norm_q(query)
        key = self.norm_k(key)
        query = apply_rotary_emb(query, rotary_emb)
        key = apply_rotary_emb(key, rotary_emb)

        out = _sdpa_gqa(query, key, value, self.heads, self.kv_heads, self.head_dim)

        # Separate output projections per stream, then re-merge + shared to_out.
        instruct_out = proc.instruct_out(out[:, :l_instruct])
        img_out = proc.img_out(out[:, l_instruct:])
        merged = mx.concatenate([instruct_out, img_out], axis=1)
        return self.to_out[0](merged)
