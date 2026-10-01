import mlx.core as mx
from PIL import Image


class QwenImage21PromptEncoder:
    SYSTEM = "Comprehend and analyze the provided prompt."

    @staticmethod
    def encode(prompt: str, images: list[Image.Image], processor, text_encoder) -> tuple[mx.array, mx.array]:
        prefix = " ".join(f"<image{i + 1}><|vision_start|><|image_pad|><|vision_end|>" for i in range(len(images)))
        text = (
            f"<|im_start|>system\n{QwenImage21PromptEncoder.SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{prefix}{prompt or ' '}<|im_end|>\n<|im_start|>assistant\n"
        )
        rgb_images = []
        for img in images:
            white = Image.new("RGB", img.size, "white")
            white.paste(img, mask=img.getchannel("A"))
            rgb_images.append(white)
        kwargs = {"images": rgb_images} if images else {}
        inputs = processor(text=[text], padding=True, padding_side="left", return_tensors="np", **kwargs)
        ids = mx.array(inputs["input_ids"])
        hidden = text_encoder(
            ids,
            pixel_values=mx.array(inputs["pixel_values"]) if images else None,
            image_grid_thw=mx.array(inputs["image_grid_thw"]) if images else None,
        )
        system = [{"role": "system", "content": [{"type": "text", "text": QwenImage21PromptEncoder.SYSTEM}]}]
        tokens = processor.apply_chat_template(system, tokenize=True, return_dict=False)
        drop = len(tokens[0]) if isinstance(tokens[0], list) else len(tokens)
        mask = ids[0, drop:] == text_encoder.image_token_id
        return hidden[:, drop:], mask
