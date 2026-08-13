import mlx.core as mx

from mflux.models.common.tokenizer import Tokenizer
from mflux.models.ernie_image.model.ernie_text_encoder.text_encoder import ErnieMistralTextEncoder


class ErniePromptEncoder:
    @staticmethod
    def encode_prompt(
        prompt: str,
        tokenizer: Tokenizer,
        text_encoder: ErnieMistralTextEncoder,
        max_length: int | None = None,
    ) -> tuple[mx.array, int]:
        output = tokenizer.tokenize(prompt, max_length=max_length)
        hidden = text_encoder(output.input_ids, output.attention_mask)
        num_valid = int(mx.sum(output.attention_mask[0]).item())
        return hidden[0, :num_valid, :], num_valid

    @staticmethod
    def build_text_batch(
        prompts: list[str],
        tokenizer: Tokenizer,
        text_encoder: ErnieMistralTextEncoder,
        max_length: int | None = None,
        pad_to: int | None = None,
        hidden_size: int = 3072,
    ) -> tuple[mx.array, mx.array]:
        embeddings = []
        lengths = []
        for p in prompts:
            emb, length = ErniePromptEncoder.encode_prompt(p, tokenizer, text_encoder, max_length)
            embeddings.append(emb)
            lengths.append(length)

        t_max = pad_to if pad_to is not None else max(lengths)
        batch = []
        for emb in embeddings:
            pad_len = t_max - emb.shape[0]
            if pad_len > 0:
                pad = mx.zeros((pad_len, hidden_size), dtype=emb.dtype)
                emb = mx.concatenate([emb, pad], axis=0)
            batch.append(emb)

        text_bth = mx.stack(batch, axis=0)
        text_lens = mx.array(lengths, dtype=mx.int32)
        return text_bth, text_lens
