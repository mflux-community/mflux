import json

import pytest

from mflux.models.fibo2.model.fibo2_text_encoder.fibo2_text_images import Fibo2TextImages
from mflux.models.fibo2.model.fibo2_text_encoder.prompt_encoder import Fibo2PromptEncoder

pytestmark = pytest.mark.fast

MARKER = Fibo2PromptEncoder.VISION_MARKER

# What diffusers' BriaFibo2Pipeline encodes for these text-to-image prompts (_parse_prompt, then __call__'s formatting)
FORMATTED = [
    ("A plain description.", "A plain description."),
    ('{"short_description": "x", "edit_instruction": "make it red"}', '{"short_description":"x"}'),
    (
        '{"structured_caption": {"short_description": "y", "text_render": [{"text": "Hi"}], "lighting": {}}, "edit_instruction": "z"}',  # noqa: E501
        '{"short_description":"y","text_render":[{"text":"Hi"}]}',
    ),
    # A short description in "structured_caption" keeps the prompt as it was given, in diffusers too
    (
        '{"structured_caption": "A short one.", "edit_instruction": "z"}',
        '{"structured_caption": "A short one.", "edit_instruction": "z"}',
    ),
    ('{"structured_caption": ""}', "{}"),
    ("[1, 2]", "[1, 2]"),
]


def test_caption_drops_empty_values_and_scores_and_serializes_compactly():
    prompt = json.dumps(
        {
            "short_description": "A fox.",
            "objects": [{"description": "A red fox.", "location": ""}, {}],
            "lighting": {},
            "aesthetics": {"composition": "centered", "aesthetic_score": 9, "preference_score": 8},
            "aesthetic_score": 7,
            "context": None,
        }
    )
    expected = {"short_description": "A fox.", "objects": [{"description": "A red fox."}], "aesthetics": {"composition": "centered"}}  # fmt: off
    assert Fibo2PromptEncoder.format_prompt(prompt) == json.dumps(expected, separators=(",", ":"))


def test_short_description_with_text_render_keeps_an_empty_list():
    prompt = json.dumps({"text_render": [], "short_description": "A plain wall."})
    assert Fibo2PromptEncoder.format_prompt(prompt) == '{"short_description":"A plain wall.","text_render":[]}'


@pytest.mark.parametrize(("prompt", "expected"), FORMATTED)
def test_text_to_image_prompts_are_formatted_as_diffusers_formats_them(prompt, expected):
    assert Fibo2PromptEncoder.format_prompt(prompt) == expected


def test_edit_instruction_alone():
    prompt = Fibo2PromptEncoder.edit_prompt("Make the bicycle blue", 1)
    assert json.loads(prompt) == {"image": MARKER, "edit_instruction": "Make the bicycle blue"}


def test_edit_with_several_images_and_a_structured_caption():
    prompt = json.dumps({"structured_caption": "A blue bicycle.", "edit_instruction": "Make it blue"})
    edit = json.loads(Fibo2PromptEncoder.edit_prompt(prompt, 2))
    assert list(edit) == ["image_1", "image_2", "structured_caption", "edit_instruction"]
    assert edit["structured_caption"] == "A blue bicycle."


def test_edit_marks_text_entries_only_with_text_images():
    prompt = json.dumps({"short_description": "A sign.", "text_render": [{"text": "OPEN"}, {"text": ""}], "edit_instruction": "Add a sign"})  # fmt: off
    unmarked = json.loads(Fibo2PromptEncoder.edit_prompt(prompt, 1))
    marked = json.loads(Fibo2PromptEncoder.edit_prompt(prompt, 1, text_images=True))
    assert "image_tokens" not in unmarked["structured_caption"]["text_render"][0]
    assert marked["structured_caption"]["text_render"] == [{"text": "OPEN", "image_tokens": MARKER}]


def test_generation_marks_each_text_entry_with_its_image_or_empty():
    prompt = Fibo2PromptEncoder.format_prompt(json.dumps({"short_description": "A shop.", "text_render": [{"text": "OPEN"}, {"text": "SALE"}]}))  # fmt: off
    base = json.loads(Fibo2PromptEncoder.mark_text_images(prompt, ""))
    marked = json.loads(Fibo2PromptEncoder.mark_text_images(prompt, MARKER))
    assert [entry["image_tokens"] for entry in base["text_render"]] == ["", ""]
    assert [entry["image_tokens"] for entry in marked["text_render"]] == [MARKER, MARKER]


@pytest.mark.parametrize(
    "prompt",
    [
        json.dumps({"short_description": "x"}),
        json.dumps({"structured_caption": {"short_description": "x"}, "objects": [], "edit_instruction": "y"}),
        json.dumps({"structured_caption": 3, "edit_instruction": "y"}),
    ],
)
def test_malformed_json_edit_prompts_are_refused(prompt):
    with pytest.raises(ValueError):
        Fibo2PromptEncoder.check_edit_prompt(prompt)


@pytest.mark.parametrize(
    "prompt",
    ["Make it blue", json.dumps({"structured_caption": "A blue bike.", "edit_instruction": "Make it blue"})],
)
def test_valid_edit_prompts_pass(prompt):
    Fibo2PromptEncoder.check_edit_prompt(prompt)


def test_text_images_come_one_per_entry_with_text_in_order_black_on_white():
    caption = {"text_render": [{"text": "A"}, {"text": ""}, {"font": "serif"}, {"text": "A much longer text"}]}
    images = Fibo2TextImages.draw_for_caption(caption, fonts_dir=None)
    assert len(images) == 2
    assert images[0].width < images[1].width
    for image in images:
        colors = {color for _, color in image.getcolors(maxcolors=1 << 16)}
        assert (255, 255, 255) in colors and (0, 0, 0) in colors
        # The padding is white on every side
        assert image.getpixel((0, 0)) == image.getpixel((image.width - 1, image.height - 1)) == (255, 255, 255)


def test_a_text_longer_than_256_characters_is_split_into_lines_of_256():
    line = Fibo2TextImages.draw("A" * 256, [])
    two_lines = Fibo2TextImages.draw("A" * 257, [])
    three_lines = Fibo2TextImages.draw("A" * 513, [])
    assert line.width == two_lines.width == three_lines.width
    assert line.height < two_lines.height < three_lines.height
    # Each further line adds the same height: a line and the spacing between lines
    assert three_lines.height - two_lines.height == two_lines.height - line.height
