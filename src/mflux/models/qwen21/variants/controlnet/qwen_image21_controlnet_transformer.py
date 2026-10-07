# Control branch adapted from VideoX-Fun's QwenImage21ControlTransformer2DModel (Alibaba, Apache-2.0).
import mlx.core as mx
from mlx import nn

from mflux.models.qwen21.model.qwen21_transformer.qwen21_layout import QwenImage21Layout
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer_block import Qwen21TransformerBlock
from mflux.models.qwen21.model.qwen21_transformer.qwen_image21_transformer import QwenImage21Transformer


class QwenImage21ControlBlock(Qwen21TransformerBlock):
    # One block of the control chain. Each hands a skip (after_proj) to a base block; the first
    # also merges the control input into the base stream (before_proj).
    def __init__(
        self, dim: int, num_attention_heads: int, attention_head_dim: int, mlp_ratio: int, eps: float, first: bool
    ):
        super().__init__(dim, num_attention_heads, attention_head_dim, mlp_ratio, eps)
        if first:
            self.before_proj = nn.Linear(dim, dim)
        self.after_proj = nn.Linear(dim, dim)


class QwenImage21ControlNet(nn.Module):
    # The Fun ControlNet-Union branch for Qwen-Image-2.1: a chain of control blocks that runs beside
    # the base transformer and adds one skip to every second base block. The control input is 129
    # channels per latent token: the control image's latents (64), the inpaint mask (1) and the masked
    # image's latents (64). The checkpoint holds this module only.
    CONTROL_IN_DIM = 129

    def __init__(self, config: dict):
        super().__init__()
        dim = config["num_attention_heads"] * config["attention_head_dim"]
        layers = list(config.get("control_layers") or range(0, config["num_layers"], 2))
        if layers[0] != 0:
            raise ValueError("The first control layer must be 0: that block merges the control input.")
        self._slots = {layer: slot for slot, layer in enumerate(layers)}
        self.control_blocks = [
            QwenImage21ControlBlock(
                dim=dim,
                num_attention_heads=config["num_attention_heads"],
                attention_head_dim=config["attention_head_dim"],
                mlp_ratio=config["mlp_ratio"],
                eps=config.get("eps", 1e-6),
                first=slot == 0,
            )
            for slot in range(len(layers))
        ]
        self.control_img_in = nn.Linear(config.get("control_in_dim", self.CONTROL_IN_DIM), dim)

    def __call__(
        self,
        transformer: QwenImage21Transformer,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        timestep: mx.array,
        layout: QwenImage21Layout,
        control_context: mx.array,
        control_scale: float = 1.0,
        encoder_hidden_states_mask: mx.array | None = None,
    ) -> mx.array:
        # The uncached pass of Qwen21Transformer.forward_reference (the reference turns the prefix KV
        # cache off under control), with the control chain run first on the same joint stream.
        if control_context.shape[1] != layout.image_indices.shape[0]:
            raise ValueError("The control input needs one row per latent token of the joint stream.")
        images = transformer.img_in(hidden_states)
        text = transformer.txt_in(encoder_hidden_states)
        x = mx.concatenate(
            [text, mx.zeros((text.shape[0], layout.target_tokens // 4, text.shape[-1]), dtype=text.dtype)], axis=1
        )
        x = x[:, layout.repeat_indices]
        x[:, layout.image_indices] = images
        control = mx.zeros_like(x)
        control[:, layout.image_indices] = self.control_img_in(control_context).astype(x.dtype)
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
        time = transformer.time_text_embed(t, x.dtype)
        modulation = transformer.modulation(time)
        target_mask, rope = layout.target_mask, layout.rope

        skips = []
        c = self.control_blocks[0].before_proj(control) + x
        for block in self.control_blocks:
            c, _ = block.forward_reference(c, modulation, target_mask, layout, rope, key_valid=key_valid)
            skips.append(block.after_proj(c))
            mx.eval(c, skips[-1])

        for index, block in enumerate(transformer.transformer_blocks):
            x, _ = block.forward_reference(x, modulation, target_mask, layout, rope, key_valid=key_valid)
            if index in self._slots:
                x = x + skips[self._slots[index]] * control_scale
            mx.eval(x)
        scale = Qwen21TransformerBlock.select_rows(transformer.norm_out.linear(nn.silu(time)), target_mask)
        return transformer.proj_out(transformer.norm_out(x, scale))[:, -layout.target_tokens :]
