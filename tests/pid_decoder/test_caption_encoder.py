from mflux.models.common.pid_decoder.caption_encoder_constants import CHI_PROMPT


def test_chi_prompt_is_nonempty_and_ends_with_user_prompt_label():
    assert CHI_PROMPT.endswith("User Prompt: ")
    assert "Enhanced prompt" in CHI_PROMPT
