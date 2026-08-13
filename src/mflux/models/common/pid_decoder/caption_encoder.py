import mlx.core as mx
from mlx import nn

from mflux.models.common.pid_decoder.caption_encoder_constants import CHI_PROMPT
from mflux.models.common.pid_decoder.gemma2.gemma2_model import Gemma2Model
from mflux.models.common.tokenizer.tokenizer import LanguageTokenizer

MODEL_MAX_LENGTH = 300


class PidCaptionEncoder(nn.Module):
    """CHI-prompt + Gemma-2 caption encoder, faithful to PiD's `_encode_text_raw`.

    Source: pid/_src/models/pixeldit_model.py::_encode_text_raw (verified 2026-07-24).
    """

    def __init__(self, gemma2: Gemma2Model, tokenizer: LanguageTokenizer):
        super().__init__()
        self.gemma2 = gemma2
        self.tokenizer = tokenizer
        # num_chi_tokens counts the CHI prompt alone, WITH special tokens (bos),
        # exactly as the reference's `self.tokenizer(chi_prompt_str, ...)` does
        # before computing `max_length_all`.
        # `tokenizer.tokenizer` is the underlying HF PreTrainedTokenizer (see
        # LanguageTokenizer/BaseTokenizer in mflux.models.common.tokenizer.tokenizer);
        # the brief's `raw_tokenizer` name does not exist on that class.
        self.num_chi_tokens = len(tokenizer.tokenizer(CHI_PROMPT, add_special_tokens=True)["input_ids"])

    def __call__(self, caption: str) -> mx.array:
        max_length = self.num_chi_tokens + MODEL_MAX_LENGTH - 2
        full_prompt = CHI_PROMPT + caption
        encoded = self.tokenizer.tokenizer(
            full_prompt,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        input_ids = mx.array(encoded["input_ids"])
        attention_mask = mx.array(encoded["attention_mask"])

        hidden = self.gemma2(input_ids, attention_mask)  # [1, max_length, 2304]

        select_index = [0] + list(range(-(MODEL_MAX_LENGTH - 1), 0))
        select_index = mx.array([i % max_length for i in select_index])
        return mx.take(hidden, select_index, axis=1)  # [1, 300, 2304]
