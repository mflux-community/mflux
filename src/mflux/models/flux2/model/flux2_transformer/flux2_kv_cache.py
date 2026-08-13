from __future__ import annotations

from typing import Literal

import mlx.core as mx

from mflux.models.flux.model.flux_transformer.common.attention_utils import AttentionUtils

CacheMode = Literal["extract", "cached"]
StreamType = Literal["double", "single"]


class Flux2KVCache:
    def __init__(self, num_double_layers: int, num_single_layers: int) -> None:
        self._double: list[tuple[mx.array, mx.array] | None] = [None] * num_double_layers
        self._single: list[tuple[mx.array, mx.array] | None] = [None] * num_single_layers
        self.mode: CacheMode | None = None
        self.num_ref_tokens: int = 0

    def configure(
        self,
        *,
        mode: CacheMode,
        num_ref_tokens: int,
    ) -> None:
        self.mode = mode
        self.num_ref_tokens = int(num_ref_tokens)

    @property
    def has_reference_tokens(self) -> bool:
        return self.num_ref_tokens > 0

    @property
    def is_extracting(self) -> bool:
        return self.mode == "extract" and self.has_reference_tokens

    @property
    def is_cached(self) -> bool:
        return self.mode == "cached"

    def store_reference(self, stream: StreamType, layer_idx: int | None, key: mx.array, value: mx.array) -> None:
        if not self.is_extracting:
            return
        # mflux uses [txt, target, ref], so reference tokens are the trailing slice.
        self._slots(stream)[self._layer_idx(layer_idx)] = (
            key[:, :, -self.num_ref_tokens :, :],
            value[:, :, -self.num_ref_tokens :, :],
        )

    def append_reference(
        self,
        stream: StreamType,
        layer_idx: int | None,
        key: mx.array,
        value: mx.array,
    ) -> tuple[mx.array, mx.array]:
        if not self.is_cached:
            return key, value
        ref_key, ref_value = self._load(stream, layer_idx)
        return mx.concatenate([key, ref_key], axis=2), mx.concatenate([value, ref_value], axis=2)

    def compute_extract_attention(
        self,
        *,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        batch_size: int,
        num_heads: int,
        head_dim: int,
    ) -> mx.array:
        if not self.is_extracting:
            return AttentionUtils.compute_attention(
                query=query,
                key=key,
                value=value,
                batch_size=batch_size,
                num_heads=num_heads,
                head_dim=head_dim,
            )

        non_ref_count = query.shape[2] - self.num_ref_tokens
        non_ref_attn = AttentionUtils.compute_attention(
            query=query[:, :, :non_ref_count, :],
            key=key,
            value=value,
            batch_size=batch_size,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        ref_attn = AttentionUtils.compute_attention(
            query=query[:, :, non_ref_count:, :],
            key=key[:, :, non_ref_count:, :],
            value=value[:, :, non_ref_count:, :],
            batch_size=batch_size,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        return mx.concatenate([non_ref_attn, ref_attn], axis=1)

    def _slots(self, stream: StreamType) -> list[tuple[mx.array, mx.array] | None]:
        if stream == "double":
            return self._double
        if stream == "single":
            return self._single
        raise ValueError(f"Unknown stream {stream!r}")

    @staticmethod
    def _layer_idx(layer_idx: int | None) -> int:
        if layer_idx is None:
            raise ValueError("KV cache layer index is required")
        return layer_idx

    def _load(self, stream: StreamType, layer_idx: int | None) -> tuple[mx.array, mx.array]:
        slot = self._slots(stream)[self._layer_idx(layer_idx)]
        if slot is None:
            raise RuntimeError(f"KV cache slot for {stream} layer {layer_idx} is empty")
        return slot
