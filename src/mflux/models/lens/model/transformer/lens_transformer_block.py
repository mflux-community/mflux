import mlx.core as mx
from mlx import nn

from mflux.models.lens.model.transformer.lens_attention import LensJointAttention


class LensGateMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class LensTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int):
        super().__init__()
        mlp_hidden = int(dim / 3 * 8)

        self.attn = LensJointAttention(dim=dim, num_heads=num_heads, head_dim=head_dim)

        # List of (SiLU, Linear) so the weight path is img_mod.1, matching the checkpoint.
        self.img_mod = [nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)]
        self.img_norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.img_norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.img_mlp = LensGateMLP(dim, mlp_hidden)

        self.txt_mod = [nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)]
        self.txt_norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.txt_norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.txt_mlp = LensGateMLP(dim, mlp_hidden)

    @staticmethod
    def _modulate(x: mx.array, mod_params: mx.array) -> tuple[mx.array, mx.array]:
        shift, scale, gate = mx.split(mod_params, 3, axis=-1)
        return x * (1 + scale[:, None]) + shift[:, None], gate[:, None]

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        temb: mx.array,
        image_rotary_emb: tuple[mx.array, mx.array],
        text_rotary_emb: tuple[mx.array, mx.array],
    ) -> tuple[mx.array, mx.array]:
        img_mod_params = self.img_mod[1](self.img_mod[0](temb))
        txt_mod_params = self.txt_mod[1](self.txt_mod[0](temb))
        img_mod1, img_mod2 = mx.split(img_mod_params, 2, axis=-1)
        txt_mod1, txt_mod2 = mx.split(txt_mod_params, 2, axis=-1)

        img_modulated, img_gate1 = self._modulate(self.img_norm1(hidden_states), img_mod1)
        txt_modulated, txt_gate1 = self._modulate(self.txt_norm1(encoder_hidden_states), txt_mod1)

        img_attn, txt_attn = self.attn(
            hidden_states=img_modulated,
            encoder_hidden_states=txt_modulated,
            image_rotary_emb=image_rotary_emb,
            text_rotary_emb=text_rotary_emb,
        )

        hidden_states = hidden_states + img_gate1 * img_attn
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn

        img_modulated2, img_gate2 = self._modulate(self.img_norm2(hidden_states), img_mod2)
        hidden_states = hidden_states + img_gate2 * self.img_mlp(img_modulated2)

        txt_modulated2, txt_gate2 = self._modulate(self.txt_norm2(encoder_hidden_states), txt_mod2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * self.txt_mlp(txt_modulated2)

        return encoder_hidden_states, hidden_states
