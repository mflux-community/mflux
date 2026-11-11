import mlx.core as mx
import pytest

from mflux.models.sensenova.model.sensenova_text_encoder.prompt_encoder import (
    SYSTEM_MESSAGE_FOR_GEN,
    SenseNovaPromptEncoder,
)


class _HFTokenizer:
    def __init__(self):
        self.queries = []

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        self.queries.append(text)
        return [ord(c) for c in text[:5]]


class _Tokenizer:
    def __init__(self):
        self.tokenizer = _HFTokenizer()


@pytest.mark.fast
class TestSenseNovaPromptEncoder:
    def test_the_cond_query_is_system_prompt_and_the_empty_think_block_before_the_image_tag(self):
        query = SenseNovaPromptEncoder.cond_query("a red fox")

        assert query == (
            f"<|im_start|>system\n{SYSTEM_MESSAGE_FOR_GEN}<|im_end|>\n"
            "<|im_start|>user\na red fox<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n<img>"
        )

    def test_the_uncond_query_is_exactly_an_empty_user_turn_with_no_system_block(self):
        assert SenseNovaPromptEncoder.uncond_query() == "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n<img>"

    def test_the_system_message_is_the_one_of_the_reference(self):
        assert SYSTEM_MESSAGE_FOR_GEN.startswith("You are an image generation and editing assistant")
        assert SYSTEM_MESSAGE_FOR_GEN.endswith("use the user's input language.")

    def test_an_empty_system_message_drops_the_whole_block(self):
        assert SenseNovaPromptEncoder.build_query("hi", "", "<img>") == "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<img>"  # fmt: skip

    def test_tokenizing_runs_the_finished_query_once_without_special_tokens(self):
        tokenizer = _Tokenizer()

        ids = SenseNovaPromptEncoder.tokenize("query", tokenizer)

        assert tokenizer.tokenizer.queries == ["query"]
        assert ids.shape == (1, 5)
        assert ids.dtype == mx.int32

    def test_the_prompt_text_goes_into_the_cond_query_unchanged(self):
        tokenizer = _Tokenizer()
        encoder = type("Encoder", (), {"__call__": lambda self, ids: [("k", "v")]})()

        kv, length = SenseNovaPromptEncoder.encode_prompt("a fox", tokenizer, encoder)

        assert tokenizer.tokenizer.queries == [SenseNovaPromptEncoder.cond_query("a fox")]
        assert (kv, length) == ([("k", "v")], 5)
