import mlx.core as mx
from mlx import nn

from mflux.models.common_models.gpt_oss.switch_layers import SwitchGLU

# Ling-mini-2.0 (bailing_moe_v2) as shipped in Ming-Image's mllm/ folder. Only the pieces a
# text-to-image prompt pass touches are built: the Qwen2.5 ViT, lm_head and audio router are
# never loaded.
HIDDEN_SIZE = 2048
NUM_LAYERS = 20
NUM_HEADS = 16
NUM_KV_HEADS = 4
HEAD_DIM = 128
ROPE_DIM = 64  # partial_rotary_factor 0.5
ROPE_THETA = 600000.0
RMS_EPS = 1e-6
VOCAB_SIZE = 157184
DENSE_INTERMEDIATE = 5120
FIRST_K_DENSE = 1
NUM_EXPERTS = 256
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
MOE_INTERMEDIATE = 512
ROUTED_SCALING = 2.5


class LingRope:
    # video_rope from Ming's modeling_bailing_moe_v2: 3D (t, h, w) positions over the first 64 of
    # each head's 128 dims (rotate-half layout). Of the 32 frequencies, the first 24 alternate
    # between the h (even) and w (odd) position and the last 8 follow t.

    @staticmethod
    def cos_sin(position_ids: mx.array) -> tuple[mx.array, mx.array]:
        # position_ids: (3, L) int -> cos/sin of shape (L, ROPE_DIM), float32
        half = ROPE_DIM // 2
        inv_freq = 1.0 / (ROPE_THETA ** (mx.arange(0, ROPE_DIM, 2, dtype=mx.float32) / ROPE_DIM))
        f = mx.arange(half)
        axis = mx.where(f >= 24, 0, mx.where(f % 2 == 0, 1, 2))
        pos = position_ids.astype(mx.float32)[axis]  # (half, L)
        freqs = pos.T * inv_freq[None, :]  # (L, half)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb), mx.sin(emb)

    @staticmethod
    def apply(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        # x: (B, H, L, HEAD_DIM)
        cos = cos.astype(x.dtype)
        sin = sin.astype(x.dtype)
        x_rot, x_pass = x[..., :ROPE_DIM], x[..., ROPE_DIM:]
        x1, x2 = x_rot[..., : ROPE_DIM // 2], x_rot[..., ROPE_DIM // 2 :]
        rotated = mx.concatenate([-x2, x1], axis=-1)
        return mx.concatenate([x_rot * cos + rotated * sin, x_pass], axis=-1)


class LingAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_key_value = nn.Linear(HIDDEN_SIZE, (NUM_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM, bias=False)
        self.q_norm = nn.RMSNorm(HEAD_DIM, eps=RMS_EPS)
        self.k_norm = nn.RMSNorm(HEAD_DIM, eps=RMS_EPS)
        self.dense = nn.Linear(NUM_HEADS * HEAD_DIM, HIDDEN_SIZE, bias=False)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        B, L, _ = x.shape
        qkv = self.query_key_value(x).reshape(B, L, NUM_HEADS + 2 * NUM_KV_HEADS, HEAD_DIM)
        q = qkv[:, :, :NUM_HEADS]
        k = qkv[:, :, NUM_HEADS : NUM_HEADS + NUM_KV_HEADS]
        v = qkv[:, :, NUM_HEADS + NUM_KV_HEADS :]
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(k).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        q = LingRope.apply(q, cos, sin)
        k = LingRope.apply(k, cos, sin)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=HEAD_DIM**-0.5, mask="causal")
        return self.dense(out.transpose(0, 2, 1, 3).reshape(B, L, NUM_HEADS * HEAD_DIM))


class LingMLP(nn.Module):
    def __init__(self, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN_SIZE, intermediate_size, bias=False)
        self.up_proj = nn.Linear(HIDDEN_SIZE, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, HIDDEN_SIZE, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class LingGate(nn.Module):
    # Sigmoid router with expert bias and group-limited top-k (8 groups, best 4 by top-2 sum).
    # The bias only steers selection; the mixing weights come from the unbiased scores.
    # Upstream runs this under torch.autocast(bfloat16), which turns its float32 F.linear into a
    # bf16 one, so logits, scores and `scores + expert_bias` are bf16 (the sigmoid itself is
    # evaluated in float32 and rounded once), while autocast's float32 `sum` makes the group
    # scores and the mixing weights float32. With biases near 17.25 (bf16 spacing 0.125 there)
    # most experts tie, and torch.topk resolves ties towards the lower index. All of this is
    # reproduced exactly; routing in plain float32 picks noticeably different experts.

    def __init__(self):
        super().__init__()
        self.weight = mx.zeros((NUM_EXPERTS, HIDDEN_SIZE))
        self.expert_bias = mx.zeros((NUM_EXPERTS,))

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        dtype = mx.bfloat16
        logits = x.astype(dtype) @ self.weight.astype(dtype).T
        scores = mx.sigmoid(logits.astype(mx.float32)).astype(dtype)
        routing = scores + self.expert_bias.astype(dtype)
        n = routing.shape[0]
        per_group = NUM_EXPERTS // N_GROUP
        grouped = routing.reshape(n, N_GROUP, per_group)
        group_scores = mx.topk(grouped, 2, axis=-1).astype(mx.float32).sum(axis=-1)
        group_idx = LingGate._top_k_low_index_first(group_scores, TOPK_GROUP, tie_eps=1e-3)
        group_mask = mx.put_along_axis(mx.zeros((n, N_GROUP)), group_idx, mx.array(1.0), axis=-1)
        expert_mask = mx.repeat(group_mask[:, :, None], per_group, axis=-1).reshape(n, NUM_EXPERTS)
        masked = mx.where(expert_mask > 0, routing.astype(mx.float32), -mx.inf)
        topk_idx = LingGate._top_k_low_index_first(masked, TOP_K, tie_eps=1e-5)
        weights = mx.take_along_axis(scores, topk_idx, axis=-1).astype(mx.float32)
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20) * ROUTED_SCALING
        return topk_idx, weights

    @staticmethod
    def _top_k_low_index_first(values: mx.array, k: int, tie_eps: float) -> mx.array:
        # values are bf16 routing scores near 17 (spacing 0.125) or float32 sums of two of them
        # (multiples of 0.125 near 35), so subtracting index * tie_eps orders ties by index without
        # reordering distinct values; tie_eps is still above float32 spacing at those magnitudes.
        key = values.astype(mx.float32) - mx.arange(values.shape[-1], dtype=mx.float32) * tie_eps
        return mx.argsort(-key, axis=-1)[:, :k]


class LingSparseMoe(nn.Module):
    # MultiRouter MoE: text tokens use `gate`, image-patch tokens (Ming's learned query tokens)
    # use `image_gate`. Experts are stored stacked for gather_mm; one shared expert is always on.

    def __init__(self):
        super().__init__()
        self.gate = LingGate()
        self.image_gate = LingGate()
        self.switch_mlp = SwitchGLU(HIDDEN_SIZE, MOE_INTERMEDIATE, NUM_EXPERTS)
        self.shared_experts = LingMLP(MOE_INTERMEDIATE)

    def __call__(self, x: mx.array, image_mask: mx.array | None) -> mx.array:
        B, L, D = x.shape
        flat = x.reshape(-1, D)
        idx, weights = self.gate(flat)
        if image_mask is not None:
            image_idx, image_weights = self.image_gate(flat)
            m = image_mask.reshape(-1, 1)
            idx = mx.where(m, image_idx, idx)
            weights = mx.where(m, image_weights, weights)
        expert_out = self.switch_mlp(flat, idx)  # (N, TOP_K, D)
        # Weighted and summed in float32 (the weights' dtype), as upstream's moe_infer does.
        y = (expert_out.astype(mx.float32) * weights[..., None]).sum(axis=-2).astype(x.dtype)
        return y.reshape(B, L, D) + self.shared_experts(x)


class LingDecoderLayer(nn.Module):
    def __init__(self, layer_idx: int):
        super().__init__()
        self.attention = LingAttention()
        self.mlp = LingMLP(DENSE_INTERMEDIATE) if layer_idx < FIRST_K_DENSE else LingSparseMoe()
        self.input_layernorm = nn.RMSNorm(HIDDEN_SIZE, eps=RMS_EPS)
        self.post_attention_layernorm = nn.RMSNorm(HIDDEN_SIZE, eps=RMS_EPS)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array, image_mask: mx.array | None) -> mx.array:
        x = x + self.attention(self.input_layernorm(x), cos, sin)
        h = self.post_attention_layernorm(x)
        h = self.mlp(h) if isinstance(self.mlp, LingMLP) else self.mlp(h, image_mask)
        return x + h


class LingMoeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.word_embeddings = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.layers = [LingDecoderLayer(i) for i in range(NUM_LAYERS)]
        self.norm = nn.RMSNorm(HIDDEN_SIZE, eps=RMS_EPS)

    def __call__(
        self,
        inputs_embeds: mx.array,
        position_ids: mx.array,
        image_mask: mx.array | None,
        output_layers: tuple[int, ...],
    ) -> dict[int, mx.array]:
        # Returns HF-style hidden_states entries by index: [i] is the input to layer i
        # (0 = embeddings) and [NUM_LAYERS] is the final-norm output.
        cos, sin = LingRope.cos_sin(position_ids)
        wanted = set(output_layers)
        out = {}
        h = inputs_embeds
        for i, layer in enumerate(self.layers):
            if i in wanted:
                out[i] = h
            h = layer(h, cos, sin, image_mask)
        if NUM_LAYERS in wanted:
            out[NUM_LAYERS] = self.norm(h)
        return out

    @staticmethod
    def stack_experts(weights: dict) -> dict:
        # Turn the checkpoint's per-expert Linear weights into SwitchGLU's stacked layout.
        # Idempotent: weights already stacked (an mflux-saved model) pass through unchanged.
        for layer in weights.get("layers", []):
            mlp = layer.get("mlp", {})
            experts = mlp.pop("experts", None)
            if experts is None:
                continue
            mlp["switch_mlp"] = {
                proj: {"weight": mx.stack([e[proj]["weight"] for e in experts])}
                for proj in ("gate_proj", "up_proj", "down_proj")
            }
        return weights
