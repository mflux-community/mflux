import mlx.core as mx

from mflux.models.common.tokenizer import Tokenizer
from mflux.models.krea2.model.krea2_text_encoder.text_encoder import Krea2TextEncoder


class Krea2PromptEncoder:
    @staticmethod
    def encode_prompt(
        prompt: str,
        tokenizer: Tokenizer,
        text_encoder: Krea2TextEncoder,
    ) -> mx.array:
        tokens = tokenizer.tokenize(prompt)
        return text_encoder.get_prompt_embeds(tokens.input_ids, tokens.attention_mask)

    @staticmethod
    def encode_prompt_pair(
        *,
        prompt: str,
        negative_prompt: str | None,
        guidance: float,
        tokenizer: Tokenizer,
        text_encoder: Krea2TextEncoder,
        prompt_cache: dict[tuple[str, str | None, float], tuple[mx.array, mx.array | None]],
    ) -> tuple[mx.array, mx.array | None]:
        cache_key = (prompt, negative_prompt, guidance)
        if cache_key in prompt_cache:
            return prompt_cache[cache_key]

        embeds = Krea2PromptEncoder.encode_prompt(prompt, tokenizer, text_encoder)
        neg_embeds = None
        if guidance != 1.0:
            neg = negative_prompt if negative_prompt and negative_prompt.strip() else " "
            neg_embeds = Krea2PromptEncoder.encode_prompt(neg, tokenizer, text_encoder)

        mx.eval(embeds)
        if neg_embeds is not None:
            mx.eval(neg_embeds)
        result = (embeds, neg_embeds)
        prompt_cache[cache_key] = result
        return result
