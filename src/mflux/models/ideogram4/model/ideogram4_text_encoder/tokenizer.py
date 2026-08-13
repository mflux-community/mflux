import mlx.core as mx
import numpy as np
from PIL import Image

from mflux.models.common.tokenizer.tokenizer import BaseTokenizer
from mflux.models.common.tokenizer.tokenizer_output import TokenizerOutput


class Ideogram4Tokenizer(BaseTokenizer):
    def tokenize(
        self,
        prompt: str | list[str],
        images: list[Image.Image] | None = None,
        max_length: int | None = None,
        **kwargs,
    ) -> TokenizerOutput:
        del images, kwargs
        max_length = max_length or self.max_length
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        tokenized = [self.tokenize_one(p, max_length=max_length) for p in prompts]
        max_tokens = max(len(tokens) for tokens in tokenized) if tokenized else 0
        input_ids = np.zeros((len(tokenized), max_tokens), dtype=np.int32)
        attention_mask = np.zeros((len(tokenized), max_tokens), dtype=np.int32)
        for idx, tokens in enumerate(tokenized):
            input_ids[idx, : len(tokens)] = tokens
            attention_mask[idx, : len(tokens)] = 1
        return TokenizerOutput(
            input_ids=mx.array(input_ids),
            attention_mask=mx.array(attention_mask),
        )

    def tokenize_one(self, prompt: str, max_length: int | None = None) -> np.ndarray:
        max_length = max_length or self.max_length
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        encoded = self.tokenizer(text, add_special_tokens=False)
        token_ids = np.asarray(encoded["input_ids"], dtype=np.int64)
        if token_ids.shape[0] > max_length:
            raise ValueError(f"prompt has {token_ids.shape[0]} tokens, exceeds max_length={max_length}")
        return token_ids
