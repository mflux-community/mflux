from pathlib import Path

from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil


class QwenImage21Processor:
    def __init__(self, path: Path, tokenizer):
        self.tokenizer = tokenizer
        self.image_processor = Qwen2VLImageProcessorPil.from_pretrained(str(path), local_files_only=True)

    def __call__(self, text: list[str], images=None, **kwargs) -> dict:
        image_inputs = self.image_processor(images=images, return_tensors="np") if images else {}
        prompts = list(text)
        cursor = 0
        for index, prompt in enumerate(prompts):
            while "<|image_pad|>" in prompt:
                if cursor >= len(image_inputs.get("image_grid_thw", [])):
                    raise ValueError("Image placeholders must match the supplied reference images.")
                grid = image_inputs["image_grid_thw"][cursor]
                count = int(grid.prod()) // self.image_processor.merge_size**2
                prompt = prompt.replace("<|image_pad|>", "<|placeholder|>" * count, 1)
                cursor += 1
            prompts[index] = prompt.replace("<|placeholder|>", "<|image_pad|>")
        return {**self.tokenizer(prompts, **kwargs), **image_inputs}

    def apply_chat_template(self, messages, **kwargs):
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def save_pretrained(self, path: str) -> None:
        self.tokenizer.save_pretrained(path)
        self.image_processor.save_pretrained(path)
