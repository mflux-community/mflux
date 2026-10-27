import mlx.core as mx

from mflux.models.anima.model.anima_transformer.anima_transformer import AnimaTransformer
from mflux.models.common.tokenizer.tokenizer import BaseTokenizer
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder


class AnimaPromptEncoder:
    # The reference reads an empty prompt as this one Qwen3 token (<|endoftext|>).
    QWEN_PAD_TOKEN = 151643

    @staticmethod
    def encode(
        prompt: str,
        qwen_tokenizer: BaseTokenizer,
        t5_tokenizer: BaseTokenizer,
        text_encoder: Qwen3TextEncoder,
        transformer: AnimaTransformer,
    ) -> mx.array:
        # The prompt goes in twice: Qwen3 token ids for the encoder (no special tokens), and T5 token ids (with the
        # closing </s>) for the adapter that turns the Qwen3 states into the transformer's context.
        qwen_ids = qwen_tokenizer.tokenize(prompt, keep_empty=True).input_ids
        if qwen_ids.size == 0:
            qwen_ids = mx.array([[AnimaPromptEncoder.QWEN_PAD_TOKEN]])
        t5_ids = t5_tokenizer.tokenize(prompt, keep_empty=True).input_ids
        # The text path runs once per prompt, in float32 like the reference's text encoder. In bfloat16 the first
        # denoising steps drift twice as far from the float32 result, most on short prompts.
        hidden_states, _ = text_encoder(qwen_ids, dtype=mx.float32)
        return transformer.encode_text(hidden_states, t5_ids)
