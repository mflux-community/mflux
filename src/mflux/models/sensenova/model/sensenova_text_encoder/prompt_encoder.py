import mlx.core as mx

from mflux.models.common.tokenizer import Tokenizer
from mflux.models.sensenova.model.sensenova_text_encoder.sensenova_text_encoder import SenseNovaTextEncoder

SYSTEM_MESSAGE_FOR_GEN = (
    "You are an image generation and editing assistant that accurately understands and executes "
    "user intent.\n\nYou support two modes:\n\n1. Think Mode:\nIf the task requires reasoning, you "
    "MUST start with a <think></think> block. Put all reasoning inside the block using plain text. "
    "DO NOT include any image tags. Keep it reasonable and directly useful for producing the final "
    "image.\n\n2. Non-Think Mode:\nIf no reasoning is needed, directly produce the final image.\n\n"
    "Task Types:\n\nA. Text-to-Image Generation:\n"
    "- Generate a high-quality image based on the user's description.\n"
    "- Ensure visual clarity, semantic consistency, and completeness.\n"
    "- DO NOT introduce elements that contradict or override the user's intent.\n\n"
    "B. Image Editing:\n"
    "- Use the provided image(s) as input or reference for modification or transformation.\n"
    "- The result can be an edited image or a new image based on the reference(s).\n"
    "- Preserve all unspecified attributes unless explicitly changed.\n\n"
    "General Rules:\n"
    "- For any visible text in the image, follow the language specified for the rendered text in "
    "the user's description, not the language of the prompt. If no language is specified, use the "
    "user's input language."
)


class SenseNovaPromptEncoder:
    COND_SUFFIX = "<think>\n\n</think>\n\n<img>"
    UNCOND_SUFFIX = "<img>"

    @staticmethod
    def build_query(prompt: str, system: str, suffix: str) -> str:
        system_block = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
        return f"{system_block}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{suffix}"

    @staticmethod
    def cond_query(prompt: str) -> str:
        return SenseNovaPromptEncoder.build_query(prompt, SYSTEM_MESSAGE_FOR_GEN, SenseNovaPromptEncoder.COND_SUFFIX)

    @staticmethod
    def uncond_query() -> str:
        return SenseNovaPromptEncoder.build_query("", "", SenseNovaPromptEncoder.UNCOND_SUFFIX)

    @staticmethod
    def tokenize(query: str, tokenizer: Tokenizer) -> mx.array:
        # The query is already the full chat string: no template, padding, BOS or EOS on top
        return mx.array([tokenizer.tokenizer.encode(query, add_special_tokens=False)], dtype=mx.int32)

    @staticmethod
    def encode_prompt(
        prompt: str, tokenizer: Tokenizer, text_encoder: SenseNovaTextEncoder
    ) -> tuple[list[tuple[mx.array, mx.array]], int]:
        input_ids = SenseNovaPromptEncoder.tokenize(SenseNovaPromptEncoder.cond_query(prompt), tokenizer)
        return text_encoder(input_ids), input_ids.shape[1]

    @staticmethod
    def encode_negative_prompt(
        tokenizer: Tokenizer, text_encoder: SenseNovaTextEncoder
    ) -> tuple[list[tuple[mx.array, mx.array]], int]:
        input_ids = SenseNovaPromptEncoder.tokenize(SenseNovaPromptEncoder.uncond_query(), tokenizer)
        return text_encoder(input_ids), input_ids.shape[1]
