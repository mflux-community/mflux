import logging

import mlx.core as mx
import pytest

from mflux.models.common.tokenizer.tokenizer_output import TokenizerOutput
from mflux.models.z_image.model.z_image_text_encoder.prompt_encoder import PromptEncoder
from mflux.models.z_image.weights.z_image_weight_definition import ZImageWeightDefinition


class _FakeTokenizer:
    def __init__(self, num_valid: int, max_length: int):
        self.max_length = max_length
        self._num_valid = num_valid

    def tokenize(self, prompt, images=None, max_length=None, **kwargs) -> TokenizerOutput:
        mask = [1] * self._num_valid + [0] * (self.max_length - self._num_valid)
        return TokenizerOutput(
            input_ids=mx.zeros((1, self.max_length), dtype=mx.int32), attention_mask=mx.array([mask])
        )


def _fake_text_encoder(input_ids: mx.array, attention_mask: mx.array) -> mx.array:
    return mx.zeros((1, input_ids.shape[1], 4))


@pytest.mark.fast
def test_z_image_tokenizer_max_length_matches_diffusers_default() -> None:
    tokenizer_defs = ZImageWeightDefinition.get_tokenizers()
    assert len(tokenizer_defs) == 1
    assert tokenizer_defs[0].max_length == 512


@pytest.mark.fast
def test_z_image_prompt_encoder_warns_when_prompt_fills_the_limit(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        embeds = PromptEncoder.encode_prompt("long", _FakeTokenizer(num_valid=8, max_length=8), _fake_text_encoder)
    assert embeds.shape == (8, 4)
    assert "8-token limit" in caplog.text


@pytest.mark.fast
def test_z_image_prompt_encoder_is_quiet_for_short_prompts(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        embeds = PromptEncoder.encode_prompt("short", _FakeTokenizer(num_valid=3, max_length=8), _fake_text_encoder)
    assert embeds.shape == (3, 4)
    assert caplog.text == ""
