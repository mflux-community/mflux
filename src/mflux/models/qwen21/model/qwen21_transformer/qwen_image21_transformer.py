import mlx.core as mx

from mflux.models.qwen21.model.qwen21_transformer.qwen21_layout import QwenImage21Layout
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer


class StepCache:
    """First-block step skipping (FBCache-style) for the prefix-cached target pass.

    While the accumulated relative-L1 change in the first block's output stays under the
    threshold, the remaining blocks are skipped and their residual is extrapolated from the
    last two full evaluations and added to the current first-block output. norm_out and
    proj_out always re-run with the current timestep, so skipped steps still track the
    schedule. Use one instance per guidance branch.
    """

    def __init__(self, threshold: float = 0.12) -> None:
        self.threshold = threshold
        self.hidden: mx.array | None = None
        self._signal: mx.array | None = None
        self._accumulated = 0.0
        self._residual: mx.array | None = None
        self._residual_slope: mx.array | None = None
        self._steps_since_store = 0

    def should_skip(self, signal: mx.array) -> bool:
        if self._signal is None or self._residual is None:
            return False
        self._steps_since_store += 1
        self._accumulated += float(mx.mean(mx.abs(signal - self._signal)) / (mx.mean(mx.abs(signal)) + 1e-6))
        if self._accumulated < self.threshold:
            residual = self._residual
            if self._residual_slope is not None:
                residual = residual + self._steps_since_store * self._residual_slope
            self.hidden = (signal.astype(mx.float32) + residual).astype(signal.dtype)
            return True
        self._accumulated = 0.0
        return False

    def store(self, hidden: mx.array, signal: mx.array) -> None:
        residual = hidden.astype(mx.float32) - signal.astype(mx.float32)
        if self._residual is not None and self._steps_since_store:
            self._residual_slope = (residual - self._residual) / self._steps_since_store
        else:
            self._residual_slope = None
        self._residual = residual
        self._steps_since_store = 0
        self.hidden = hidden
        self._signal = signal


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
        step_cache: StepCache | None = None,
    ) -> mx.array:
        return self.forward_reference(
            hidden_states, encoder_hidden_states, timestep, layout, cache, encoder_hidden_states_mask, step_cache
        )
