import logging

import mlx.core as mx

from mflux.models.common.tokenizer import Tokenizer
from mflux.models.z_image.model.z_image_text_encoder.text_encoder import TextEncoder

log = logging.getLogger(__name__)


class PromptEncoder:
    @staticmethod
    def encode_prompt(
        prompt: str,
        tokenizer: Tokenizer,
        text_encoder: TextEncoder,
    ) -> mx.array:
        output = tokenizer.tokenize(prompt)
        cap_feats = text_encoder(output.input_ids, output.attention_mask)
        num_valid = int(mx.sum(output.attention_mask[0]).item())
        PromptEncoder._warn_if_truncated(num_valid, getattr(tokenizer, "max_length", None))
        return cap_feats[0, :num_valid, :]

    @staticmethod
    def _warn_if_truncated(num_valid: int, max_length: int | None) -> None:
        # The tokenizer truncates silently, so a full mask is the only sign that the tokenizer dropped text.
        if max_length is not None and num_valid >= max_length:
            log.warning(
                "The prompt fills the %s-token limit of Z-Image. The model may ignore the text after the limit.",
                max_length,
            )
