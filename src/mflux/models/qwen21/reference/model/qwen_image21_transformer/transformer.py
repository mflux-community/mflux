import mlx.core as mx
from mlx import nn

from mflux.models.qwen21.reference.model.qwen_image21_transformer.blocks import (
    FinalNorm,
    TextProjection,
    TimeEmbedding,
    TransformerBlock,
)
from mflux.models.qwen21.reference.model.qwen_image21_transformer.layout import QwenImage21Layout


class QwenImage21Transformer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        if config.get("patch_size", 1) != 1 or not config.get("causal_condition", True):
            raise ValueError("Qwen-Image-2.1 requires patch_size=1 and causal_condition=True.")
        heads, head_dim = config["num_attention_heads"], config["attention_head_dim"]
        dim = heads * head_dim
        eps = config.get("eps", 1e-6)
        self.axes = tuple(config["axes_dims_rope"])
        self.img_in = nn.Linear(config["in_channels"], dim, bias=False)
        self.txt_in = TextProjection(config["context_in_dim"], dim, eps)
        self.time_text_embed = TimeEmbedding(dim)
        self.modulation = [nn.SiLU(), nn.Linear(dim, 4 * dim, bias=False)]
        self.transformer_blocks = [
            TransformerBlock(dim, heads, head_dim, config["mlp_ratio"], eps) for _ in range(config["num_layers"])
        ]
        self.norm_out = FinalNorm(dim, eps)
        self.proj_out = nn.Linear(dim, config["out_channels"], bias=False)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        timestep: mx.array,
        layout: QwenImage21Layout,
        cache: list | None = None,
        encoder_hidden_states_mask: mx.array | None = None,
    ) -> mx.array:
        cached = cache is not None and len(cache) > 0
        target_mask = layout.target_mask
        rope = layout.rope
        if cached:
            if len(cache) != len(self.transformer_blocks):
                raise ValueError("Incomplete Qwen-Image-2.1 prefix cache.")
            x = self.img_in(hidden_states[:, -layout.target_tokens :])
            target_mask = target_mask[layout.prefix_length :]
            rope = tuple(v[layout.prefix_length :] for v in rope)
        else:
            images = self.img_in(hidden_states)
            text = self.txt_in(encoder_hidden_states)
            x = mx.concatenate(
                [text, mx.zeros((text.shape[0], layout.target_tokens // 4, text.shape[-1]), dtype=text.dtype)], axis=1
            )
            x = x[:, layout.repeat_indices]
            x[:, layout.image_indices] = images
        key_valid = None
        if encoder_hidden_states_mask is not None:
            mask = mx.concatenate(
                [
                    encoder_hidden_states_mask.astype(mx.bool_),
                    mx.ones((x.shape[0], layout.target_tokens // 4), dtype=mx.bool_),
                ],
                axis=1,
            )
            key_valid = mask[:, layout.repeat_indices]
            key_valid[:, layout.image_indices] = True
        t = mx.concatenate([timestep.astype(x.dtype).reshape(-1), mx.zeros((1,), dtype=x.dtype)])
        time = self.time_text_embed(t, x.dtype)
        modulation = self.modulation[1](self.modulation[0](time))
        for index, block in enumerate(self.transformer_blocks):
            x, stored = block(
                x,
                modulation,
                target_mask,
                layout,
                rope,
                cached=cache[index] if cached else None,
                extract=cache is not None and not cached,
                key_valid=key_valid,
            )
            if stored is not None:
                cache.append(stored)
            mx.eval(x)
        return self.proj_out(self.norm_out(x, time, target_mask))[:, -layout.target_tokens :]
