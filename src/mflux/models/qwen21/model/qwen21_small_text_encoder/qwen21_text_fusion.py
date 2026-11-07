import mlx.core as mx
from mlx import nn

from mflux.models.common_models.aiartlab_adapter import (
    AdapterAttentionBlock,
    AdapterAttentionMixer,
    AdapterPerLayerNorm,
)


class Qwen21TextFusion(nn.Module):
    # The zen-image-edit text_fusion block: Qwen3.5-0.8B hidden-state slices (B, L, K*d) -> the 4096-dim
    # input of the Qwen-Image-2.1 txt_in. Output = MLP(x) + token attention(x) + slice mixer(x).
    def __init__(self, config: dict):
        super().__init__()
        if config.get("norm") != "ln-per-layer" or config.get("proj_layers", 0) < 2:
            raise ValueError(f"Unsupported text_fusion_config: {config}")
        in_dim, out_dim, hidden = config["in_dim"], config["out_dim"], config["hidden"]
        n_slices = config["n_slices"]
        self.mlp = [AdapterPerLayerNorm(in_dim, n_slices), nn.Linear(in_dim, hidden), nn.GELU(approx="tanh")]
        for _ in range(config["proj_layers"] - 2):
            self.mlp += [nn.Linear(hidden, hidden), nn.GELU(approx="tanh")]
        self.mlp.append(nn.Linear(hidden, out_dim))
        self.attn = (
            AdapterAttentionMixer(
                in_dim, out_dim, config["attn_dim"], config["attn_heads"], config["attention"], config["max_len"]
            )
            if config.get("attention")
            else None
        )
        self.mixer = (
            _SliceMixer(
                n_slices, in_dim // n_slices, out_dim, config["mixer_heads"], config["mixer"], config["mixer_ffn"]
            )
            if config.get("mixer")
            else None
        )

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        # mask: (B, L), true for real tokens; the token attention ignores the padding.
        x = x.astype(mx.float32)
        out = x
        for layer in self.mlp:
            out = layer(out)
        if self.attn is not None:
            out = out + self.attn(x, mask)
        if self.mixer is not None:
            out = out + self.mixer(x)
        return out


class _SliceMixer(nn.Module):
    def __init__(self, n_slices: int, dim: int, out_dim: int, heads: int, blocks: int, ffn_mult: int):
        super().__init__()
        self.n_slices = n_slices
        self.dim = dim
        self.blocks = [AdapterAttentionBlock(dim, heads, ffn_mult) for _ in range(blocks)]
        self.proj = nn.Linear(n_slices, 1, bias=False)
        self.out = nn.Linear(dim, out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        batch, length, _ = x.shape
        h = mx.fast.layer_norm(x.reshape(batch, length, self.n_slices, self.dim), None, None, 1e-5)
        h = h.reshape(batch * length, self.n_slices, self.dim)
        for block in self.blocks:
            h = block(h)
        h = self.proj(h.swapaxes(1, 2)).squeeze(-1)
        return self.out(h).reshape(batch, length, -1)
