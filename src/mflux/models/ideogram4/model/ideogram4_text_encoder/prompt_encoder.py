from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np

from mflux.models.common.tokenizer import Tokenizer
from mflux.models.ideogram4.model.ideogram4_text_encoder.caption import Ideogram4Caption
from mflux.models.ideogram4.model.ideogram4_text_encoder.text_encoder import Qwen3TextEncoder


class Ideogram4PromptEncoder:
    @staticmethod
    def resolve_prompt(
        prompt: str | dict[str, Any],
        *,
        strict_caption_validation: bool,
        warn_on_caption_issues: bool,
    ) -> str:
        prepared_prompt = Ideogram4Caption.prepare(prompt)
        if strict_caption_validation:
            Ideogram4Caption.raise_for_warnings(prepared_prompt.warnings)
        elif warn_on_caption_issues:
            Ideogram4Caption.emit_warnings(prepared_prompt.warnings, stacklevel=3)
        if not prepared_prompt.prompt:
            raise ValueError("prompt must not be empty")
        return prepared_prompt.prompt

    @staticmethod
    def build_inputs(
        tokenizer: Tokenizer,
        prompts: list[str],
        *,
        height: int,
        width: int,
    ) -> dict[str, Any]:
        tokenized = [(tokenizer.tokenize_one(prompt), 0) for prompt in prompts]
        tokenized = [(tokens, int(tokens.shape[0])) for tokens, _ in tokenized]
        batch_size = len(prompts)
        grid_h = height // 16
        grid_w = width // 16
        num_image_tokens = grid_h * grid_w
        max_text_tokens = max(num_text for _, num_text in tokenized)
        total_seq_len = max_text_tokens + num_image_tokens

        h_idx = np.repeat(np.arange(grid_h, dtype=np.int64), grid_w)
        w_idx = np.tile(np.arange(grid_w, dtype=np.int64), grid_h)
        t_idx = np.zeros_like(h_idx)
        image_pos = np.stack([t_idx, h_idx, w_idx], axis=1) + 65536

        token_ids = np.zeros((batch_size, total_seq_len), dtype=np.int64)
        text_position_ids = np.zeros((batch_size, total_seq_len, 3), dtype=np.int64)
        position_ids = np.zeros((batch_size, total_seq_len, 3), dtype=np.int64)
        segment_ids = np.full((batch_size, total_seq_len), -1, dtype=np.int64)
        indicator = np.zeros((batch_size, total_seq_len), dtype=np.int64)

        for batch_idx, (tokens, num_text) in enumerate(tokenized):
            pad_len = max_text_tokens - num_text
            total_unpadded = num_text + num_image_tokens
            offset = pad_len
            token_ids[batch_idx, offset : offset + num_text] = tokens

            text_pos = np.arange(num_text, dtype=np.int64)
            text_pos_3d = np.stack([text_pos, text_pos, text_pos], axis=1)
            text_position_ids[batch_idx, offset : offset + num_text] = text_pos_3d
            position_ids[batch_idx, offset : offset + num_text] = text_pos_3d
            position_ids[batch_idx, offset + num_text :] = image_pos

            indicator[batch_idx, offset : offset + num_text] = 3
            indicator[batch_idx, offset + num_text :] = 2
            segment_ids[batch_idx, offset : offset + total_unpadded] = 1

        return {
            "token_ids": mx.array(token_ids, dtype=mx.int32),
            "text_position_ids": mx.array(text_position_ids, dtype=mx.int32),
            "position_ids": mx.array(position_ids, dtype=mx.int32),
            "segment_ids": mx.array(segment_ids, dtype=mx.int32),
            "indicator": mx.array(indicator, dtype=mx.int32),
            "num_image_tokens": num_image_tokens,
            "grid_h": grid_h,
            "grid_w": grid_w,
            "max_text_tokens": max_text_tokens,
        }

    @staticmethod
    def encode_prompt(
        *,
        prompt: str,
        width: int,
        height: int,
        inputs: dict[str, Any],
        text_encoder: Qwen3TextEncoder | None,
        prompt_cache: dict[tuple[str, int, int], mx.array],
    ) -> mx.array:
        cache_key = (prompt, width, height)
        if cache_key in prompt_cache:
            return prompt_cache[cache_key]
        if text_encoder is None:
            raise RuntimeError("Text encoder has been evicted and prompt features are not cached.")
        attention_mask = (inputs["indicator"] == 3).astype(mx.int32)
        pos_2d = inputs["text_position_ids"][:, :, 0]
        embeds = text_encoder.get_prompt_embeds(
            inputs["token_ids"],
            attention_mask,
            pos_2d,
        )
        embeds = embeds * attention_mask[..., None].astype(embeds.dtype)
        embeds = embeds.astype(mx.float32)
        mx.eval(embeds)
        prompt_cache[cache_key] = embeds
        return embeds

    @staticmethod
    def negative_inputs(inputs: dict[str, Any], llm_features: mx.array) -> dict[str, mx.array]:
        max_text_tokens = int(inputs["max_text_tokens"])
        num_image_tokens = int(inputs["num_image_tokens"])
        return {
            "position_ids": inputs["position_ids"][:, max_text_tokens:, :],
            "segment_ids": inputs["segment_ids"][:, max_text_tokens:],
            "indicator": inputs["indicator"][:, max_text_tokens:],
            "llm_features": mx.zeros((1, num_image_tokens, llm_features.shape[-1]), dtype=llm_features.dtype),
        }
