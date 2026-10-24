import json

import pytest

from mflux.models.fibo2.model.fibo2_text_encoder.fibo2_caption_check import Fibo2CaptionCheck
from mflux.models.fibo2.variants import Fibo2, Fibo2Edit

pytestmark = pytest.mark.fast

# Every field the training captions always have
CAPTION = {
    "short_description": "A red bicycle against a white wall.",
    "background_setting": "a white brick wall",
    "lighting": {"conditions": "morning sun", "direction": "from the right"},
    "aesthetics": {"composition": "centered", "color_scheme": "red on white", "mood_atmosphere": "calm"},
    "context": "a street photograph",
}


def test_a_complete_caption_and_plain_text_pass_without_warnings():
    assert Fibo2CaptionCheck.check(json.dumps(CAPTION), edit=False) == []
    assert Fibo2CaptionCheck.check("a red bicycle", edit=False) == []
    assert Fibo2CaptionCheck.check("[cinematic] a red bicycle", edit=False) == []


def test_a_prompt_that_starts_like_json_but_does_not_parse_is_an_error():
    with pytest.raises(ValueError, match=r"not valid JSON: .* at line 1, column 26"):
        Fibo2CaptionCheck.check('{"short_description": "x",}', edit=False)


def test_every_field_of_the_wrong_type_is_named():
    caption = {**CAPTION, "short_description": 5, "objects": {"description": "d"}, "lighting": "soft"}
    with pytest.raises(ValueError) as error:
        Fibo2CaptionCheck.check(json.dumps(caption), edit=False)
    assert str(error.value).splitlines()[1:] == [
        "  short_description has to be a string, not the number 5.",
        "  lighting has to be a JSON object, not the string 'soft'.",
        "  objects has to be a list of objects, not a JSON object.",
    ]


def test_a_list_entry_of_the_wrong_type_has_its_index():
    caption = {**CAPTION, "text_render": [{"text": "OPEN", "location": "top", "size": "m", "color": "w", "font": 3}]}
    with pytest.raises(ValueError, match=r"text_render\[0\]\.font has to be a string, not the number 3"):
        Fibo2CaptionCheck.check(json.dumps(caption), edit=False)


@pytest.mark.parametrize(("value", "valid"), [(3, True), ("3", True), (3.0, True), (True, True), (3.5, False), ("three", False)])  # fmt: off
def test_a_count_takes_what_thales_vgl_takes_as_an_integer(value, valid):
    # pydantic's lax mode in ThalesVGL (bria-vgl 0.10.0) accepts the first four and rejects the last two
    caption = {**CAPTION, "objects": [{"description": "d", "location": "l", "relationship": "r", "number_of_objects": value}]}  # fmt: off
    if valid:
        assert Fibo2CaptionCheck.check(json.dumps(caption), edit=False) == []
    else:
        with pytest.raises(ValueError, match="number_of_objects has to be an integer"):
            Fibo2CaptionCheck.check(json.dumps(caption), edit=False)


def test_unknown_and_missing_fields_are_warnings():
    caption = {key: value for key, value in CAPTION.items() if key != "context"} | {"short_desciption": "typo"}
    assert Fibo2CaptionCheck.check(json.dumps(caption), edit=False) == [
        "Missing: context. FIBO 2's training captions have these fields.",
        "Unknown field 'short_desciption': the model was not trained on it. Did you mean 'short_description'?",
    ]


def test_fibo1_scores_and_empty_values_are_not_reported():
    caption = {**CAPTION, "aesthetics": {**CAPTION["aesthetics"], "aesthetic_score": "very high"}, "objects": []}
    assert Fibo2CaptionCheck.check(json.dumps(caption), edit=False) == []


def test_an_edit_caption_needs_its_instruction_as_a_string():
    with pytest.raises(ValueError, match="edit_instruction"):
        Fibo2CaptionCheck.check(json.dumps(CAPTION), edit=True)
    with pytest.raises(ValueError, match='"edit_instruction" has to be a string, not the number 3'):
        Fibo2CaptionCheck.check(json.dumps({**CAPTION, "edit_instruction": 3}), edit=True)
    assert Fibo2CaptionCheck.check(json.dumps({**CAPTION, "edit_instruction": "Make it blue."}), edit=True) == []
    assert Fibo2CaptionCheck.check("Make it blue.", edit=True) == []


def test_a_structured_caption_is_checked_like_a_caption():
    prompt = json.dumps({"structured_caption": {**CAPTION, "lighting": "soft"}, "edit_instruction": "Make it blue."})
    with pytest.raises(ValueError, match="lighting has to be a JSON object"):
        Fibo2CaptionCheck.check(prompt, edit=True)


def test_a_generation_says_that_it_ignores_an_instruction():
    warnings = Fibo2CaptionCheck.check(json.dumps({**CAPTION, "edit_instruction": "Make it blue."}), edit=False)
    assert warnings == ["edit_instruction is read only when editing images; this generation ignores it."]


@pytest.mark.parametrize(("model_class", "extra"), [(Fibo2, {}), (Fibo2Edit, {"image_paths": ["a.png"]})])
def test_generate_image_logs_the_warnings_and_raises_the_errors(monkeypatch, caplog, model_class, extra):
    # The Python API checks the caption as the commands do, before any image is loaded
    monkeypatch.setattr(model_class, "_generate", lambda self, **kwargs: kwargs)
    monkeypatch.setattr("mflux.models.fibo2.variants.edit.fibo2_edit.Fibo2EditImages.prepare", lambda *args: ([], [], (64, 64)))  # fmt: off
    model = model_class.__new__(model_class)
    edit = {"edit_instruction": "Make it blue."} if extra else {}
    model.generate_image(seed=1, prompt=json.dumps({**CAPTION, **edit, "lightning": "soft"}), **extra)
    assert "Unknown field 'lightning'" in caplog.text
    with pytest.raises(ValueError, match="lighting has to be a JSON object"):
        model.generate_image(seed=1, prompt=json.dumps({**CAPTION, **edit, "lighting": "soft"}), **extra)
