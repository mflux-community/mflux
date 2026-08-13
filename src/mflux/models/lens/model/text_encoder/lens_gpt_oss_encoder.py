"""Lens text encoder: GPT-OSS 20B with multi-layer hidden-state capture.

Wraps the vendored GPT-OSS architecture and mirrors the reference extraction
exactly: capture the residual stream AFTER blocks (5, 11, 17, 23), no final
norm, stack to [B, S, 4, 2880] and drop the first 97 tokens of the rendered
harmony template. The per-layer normalization and the 11520 -> inner_dim
projection belong to the transformer (txt_norm.0-3 + txt_in), not here.

The default checkpoint is the mlx-format community conversion; its config
carries the quantization recipe (mxfp4 experts + q8 elsewhere), so loading
is self-describing and needs no -q flag.
"""

import json
import re
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from mflux.models.common_models.gpt_oss.base_utils import create_attention_mask
from mflux.models.common_models.gpt_oss.gpt_oss import (
    Model as GptOssModel,
    ModelArgs as GptOssModelArgs,
)
from mflux.models.lens.model.text_encoder.lens_prompt_template import (
    LENS_MAX_TOKENS,
    LENS_SELECTED_LAYERS,
    LENS_TXT_OFFSET,
    render_lens_chat,
)

DEFAULT_ENCODER_REPO = "mlx-community/gpt-oss-20b-MXFP4-Q8"


# Any <|...|> sequence: the harmony control vocabulary, present and future.
_HARMONY_MARKER = re.compile(r"<\|[^|>]*\|>")


def capture_hidden_states(model: GptOssModel, input_ids: mx.array, selected=LENS_SELECTED_LAYERS) -> list[mx.array]:
    """Residual stream AFTER each selected block, no final norm (reference semantics)."""
    inner = model.model
    x = inner.embed_tokens(input_ids)
    full_mask = create_attention_mask(x, None)
    swa_mask = create_attention_mask(x, None, window_size=inner.window_size)
    wanted = {idx: pos for pos, idx in enumerate(selected)}
    captured: list = [None] * len(selected)
    for i, (layer, layer_type) in enumerate(zip(inner.layers, inner.layer_types, strict=True)):
        mask = full_mask if layer_type == "full_attention" else swa_mask
        x = layer(x, mask, None)
        if i in wanted:
            captured[wanted[i]] = x
        if i >= max(selected):
            break
    return captured


class LensGptOssEncoder:
    def __init__(self, model_path: str):
        root = Path(model_path)
        config = json.loads((root / "config.json").read_text())
        self.model = GptOssModel(GptOssModelArgs.from_dict(config))

        weights = {}
        for shard in sorted(root.glob("model*.safetensors")):
            weights.update(mx.load(str(shard)))
        if hasattr(self.model, "sanitize"):
            weights = self.model.sanitize(weights)

        quantization = config.get("quantization")
        if quantization is not None:

            def class_predicate(path, module):
                if path in quantization:
                    return quantization[path]
                if not hasattr(module, "to_quantized"):
                    return False
                return f"{path}.scales" in weights

            nn.quantize(
                self.model,
                group_size=quantization["group_size"],
                bits=quantization["bits"],
                mode=quantization.get("mode", "affine"),
                class_predicate=class_predicate,
            )
        # Feature extraction never reaches lm_head; dropping it and its weights
        # saves ~1.2 GB (2880 x 201088 at bf16) per load.
        self.model.lm_head = nn.Identity()
        weights = {k: v for k, v in weights.items() if not k.startswith("lm_head.")}
        self.model.load_weights(list(weights.items()))

        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))

    def encode(self, prompt: str) -> mx.array:
        """Return stacked features [1, S, len(selected), hidden] for the prompt."""
        ids = self._template_ids(prompt)
        input_ids = mx.array([ids])

        captured = capture_hidden_states(self.model, input_ids)
        if any(c is None for c in captured):
            raise ValueError(
                f"selected layers {LENS_SELECTED_LAYERS} exceed the encoder depth; capture came back incomplete"
            )
        stacked = mx.stack(captured, axis=2)  # [B, S, L, H]
        return stacked[:, LENS_TXT_OFFSET:]

    def _template_ids(self, prompt: str) -> list[int]:
        # Harmony control markers inside the prompt would forge template blocks and
        # desynchronize the fixed 97-token offset. The vocabulary carries more of them
        # than the template uses (<|constrain|>, <|call|>, <|endoftext|>, ...), and it
        # grows across checkpoints, so strip the whole shape rather than a kept list.
        prompt = _HARMONY_MARKER.sub(" ", prompt)

        encode = lambda text: self.tokenizer.encode(text, add_special_tokens=False).ids  # noqa: E731
        ids = encode(render_lens_chat(prompt))
        if len(ids) <= LENS_MAX_TOKENS:
            return ids

        # Over-long prompt: truncating the RENDERED sequence would cut the fixed
        # assistant tail and misalign the template. Truncate the prompt segment
        # instead, keeping the frozen prefix and suffix intact.
        empty = render_lens_chat("")
        cut = empty.index("<|end|><|start|>assistant")
        prefix_ids = encode(empty[:cut])  # system + developer + user header (the 97-token offset)
        suffix_ids = encode(empty[cut:])  # user close + both assistant blocks
        budget = max(0, LENS_MAX_TOKENS - len(prefix_ids) - len(suffix_ids))
        prompt_ids = encode(prompt)[:budget]
        return prefix_ids + prompt_ids + suffix_ids
