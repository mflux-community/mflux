import mlx.core as mx

from mflux.models.qwen21.model.qwen21_transformer.qwen21_layout import QwenImage21Layout
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer


class QwenImage21Transformer(Qwen21Transformer):
    def __init__(self, config: dict):
        if config.get("patch_size", 1) != 1 or not config.get("causal_condition", True):
            raise ValueError("Qwen-Image-2.1 requires patch_size=1 and causal_condition=True.")
        super().__init__(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            num_layers=config["num_layers"],
            attention_head_dim=config["attention_head_dim"],
            num_attention_heads=config["num_attention_heads"],
            context_in_dim=config["context_in_dim"],
            mlp_ratio=config["mlp_ratio"],
            axes_dims_rope=tuple(config["axes_dims_rope"]),
            eps=config.get("eps", 1e-6),
        )

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        timestep: mx.array,
        layout: QwenImage21Layout,
        cache: list | None = None,
        encoder_hidden_states_mask: mx.array | None = None,
    ) -> mx.array:
        return self.forward_reference(
            hidden_states, encoder_hidden_states, timestep, layout, cache, encoder_hidden_states_mask
        )
