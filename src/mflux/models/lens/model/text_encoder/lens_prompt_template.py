"""Lens harmony prompt template.

Byte-compatible with the reference pipeline: the date is FROZEN at 2026-05-23 and the
assistant thinking line is fixed. Changing either changes the text embeddings and
breaks parity with every reference output. The 97-token offset trims the rendered
system + developer blocks plus the user-message header, so the DiT sees features from
the user's prompt content onward (verified token-exact against the reference).
"""

LENS_SELECTED_LAYERS = (5, 11, 17, 23)
LENS_TXT_OFFSET = 97
LENS_MAX_TOKENS = 512
LENS_PAD_TOKEN_ID = 199999  # <|endoftext|>

_SYSTEM = (
    "Describe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background."
)
_THINKING = "Need to generate one image according to the description."
_DATE = "2026-05-23"


def render_lens_chat(prompt: str) -> str:
    return (
        f"<|start|>system<|message|>"
        f"You are ChatGPT, a large language model trained by OpenAI.\n"
        f"Knowledge cutoff: 2024-06\n"
        f"Current date: {_DATE}\n\n"
        f"Reasoning: medium\n\n"
        f"# Valid channels: analysis, commentary, final. "
        f"Channel must be included for every message.<|end|>"
        f"<|start|>developer<|message|># Instructions\n\n"
        f"{_SYSTEM}\n\n<|end|>"
        f"<|start|>user<|message|>{prompt}<|end|>"
        f"<|start|>assistant<|channel|>analysis<|message|>"
        f"{_THINKING}<|end|>"
        f"<|start|>assistant<|channel|>final<|message|>"
    )
