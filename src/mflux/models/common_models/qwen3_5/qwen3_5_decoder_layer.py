import mlx.core as mx
from mlx import nn

from mflux.models.common_models.qwen3_5.qwen3_5_attention import Qwen35Attention
from mflux.models.common_models.qwen3_5.qwen3_5_gated_delta_net import Qwen35GatedDeltaNet
from mflux.models.common_models.qwen3_5.qwen3_5_rms_norm import Qwen35RMSNorm


class Qwen35MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen35DecoderLayer(nn.Module):
    def __init__(self, layer_type: str, config: dict, fused_norm: bool = False, beta_in_float32: bool = True):
        super().__init__()
        self.layer_type = layer_type
        hidden_size = config["hidden_size"]
        eps = config["rms_norm_eps"]
        if layer_type == "linear_attention":
            self.linear_attn = Qwen35GatedDeltaNet(
                hidden_size=hidden_size,
                num_key_heads=config["linear_num_key_heads"],
                num_value_heads=config["linear_num_value_heads"],
                key_head_dim=config["linear_key_head_dim"],
                value_head_dim=config["linear_value_head_dim"],
                conv_kernel_size=config["linear_conv_kernel_dim"],
                rms_norm_eps=eps,
                fused_norm=fused_norm,
                beta_in_float32=beta_in_float32,
            )
        elif layer_type == "full_attention":
            self.self_attn = Qwen35Attention(
                hidden_size=hidden_size,
                num_attention_heads=config["num_attention_heads"],
                num_key_value_heads=config["num_key_value_heads"],
                head_dim=config["head_dim"],
                rotary_dim=int(config["head_dim"] * config["partial_rotary_factor"]),
                rope_theta=config["rope_theta"],
                rms_norm_eps=eps,
                fused_norm=fused_norm,
            )
        else:
            raise ValueError(f"Unknown Qwen3.5 layer type: {layer_type!r}")
        self.mlp = Qwen35MLP(hidden_size, config["intermediate_size"])
        self.input_layernorm = Qwen35RMSNorm(hidden_size, eps=eps, fused=fused_norm)
        self.post_attention_layernorm = Qwen35RMSNorm(hidden_size, eps=eps, fused=fused_norm)

    def __call__(self, hidden_states: mx.array, rotary: tuple[mx.array, mx.array] | None = None) -> mx.array:
        normed = self.input_layernorm(hidden_states)
        if self.layer_type == "linear_attention":
            mixed = self.linear_attn(normed)
        else:
            mixed = self.self_attn(normed, rotary)
        hidden_states = hidden_states + mixed
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
