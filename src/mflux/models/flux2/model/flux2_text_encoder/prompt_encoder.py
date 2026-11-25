import mlx.core as mx

from mflux.models.common.tokenizer import Tokenizer
from mflux.models.flux2.model.flux2_text_encoder.pruned_text_encoder import Flux2PrunedTextEncoder
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
from mflux.models.flux2.model.flux2_text_encoder.small_text_encoder import Flux2SmallTextEncoder


class Flux2PromptEncoder:
    @staticmethod
    def encode_prompt(
        prompt: str | list[str],
        tokenizer: Tokenizer,
        text_encoder: Qwen3TextEncoder | Flux2SmallTextEncoder | Flux2PrunedTextEncoder,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        text_encoder_out_layers: tuple[int, ...] = (9, 18, 27),
    ) -> tuple[mx.array, mx.array]:
        if isinstance(text_encoder, Flux2SmallTextEncoder):
            return Flux2PromptEncoder._encode_with_small_encoder(prompt, tokenizer, text_encoder, num_images_per_prompt)
        prompt_embeds = Flux2PromptEncoder._get_qwen3_prompt_embeds(
            prompt=prompt,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            max_sequence_length=max_sequence_length,
            hidden_state_layers=text_encoder_out_layers,
        )
        if num_images_per_prompt > 1:
            prompt_embeds = mx.repeat(prompt_embeds, num_images_per_prompt, axis=0)
        text_ids = Flux2PromptEncoder.prepare_text_ids(prompt_embeds)
        return prompt_embeds, text_ids

    @staticmethod
    def _encode_with_small_encoder(
        prompt: str | list[str],
        tokenizer: Tokenizer,
        text_encoder: Flux2SmallTextEncoder,
        num_images_per_prompt: int,
    ) -> tuple[mx.array, mx.array]:
        tokens = tokenizer.tokenize(prompt=prompt, max_length=text_encoder.max_sequence_length, keep_empty=True)
        prompt_embeds = text_encoder(tokens.input_ids, tokens.attention_mask)
        # Ids cover the full padded sequence; the leading tokens are dropped from embeds and ids alike.
        text_ids = Flux2PromptEncoder.prepare_text_ids(prompt_embeds)
        prompt_embeds = prompt_embeds[:, text_encoder.drop_first :]
        text_ids = text_ids[:, text_encoder.drop_first :]
        if num_images_per_prompt > 1:
            prompt_embeds = mx.repeat(prompt_embeds, num_images_per_prompt, axis=0)
            text_ids = mx.repeat(text_ids, num_images_per_prompt, axis=0)
        return prompt_embeds, text_ids

    @staticmethod
    def _get_qwen3_prompt_embeds(
        prompt: str | list[str],
        tokenizer: Tokenizer,
        text_encoder: Qwen3TextEncoder | Flux2PrunedTextEncoder,
        max_sequence_length: int,
        hidden_state_layers: tuple[int, ...],
    ) -> mx.array:
        tokens = tokenizer.tokenize(prompt=prompt, max_length=max_sequence_length)
        return text_encoder.get_prompt_embeds(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
            hidden_state_layers=hidden_state_layers,
        )

    @staticmethod
    def prepare_text_ids(x: mx.array, t_coord: mx.array | None = None) -> mx.array:
        batch_size, seq_len, _ = x.shape
        out_ids = []
        for i in range(batch_size):
            if t_coord is None:
                t = mx.zeros((seq_len,), dtype=mx.int32)
            else:
                t = t_coord[i]
                if t.ndim == 0:
                    t = mx.full((seq_len,), t, dtype=mx.int32)
                elif t.shape[0] != seq_len:
                    t = mx.broadcast_to(t, (seq_len,))
                t = t.astype(mx.int32)
            h = mx.zeros((seq_len,), dtype=mx.int32)
            w = mx.zeros((seq_len,), dtype=mx.int32)
            token_ids = mx.arange(seq_len, dtype=mx.int32)
            coords = mx.stack([t, h, w, token_ids], axis=1)
            out_ids.append(coords)
        return mx.stack(out_ids, axis=0)
