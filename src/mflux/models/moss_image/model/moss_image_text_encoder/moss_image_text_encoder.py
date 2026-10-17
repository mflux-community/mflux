from pathlib import Path

import mlx.core as mx
import torch
from transformers import AutoModel, AutoTokenizer


class MOSSImageTextEncoder:
    def __init__(self, model_path: str | Path):
        self.path = Path(model_path)
        text_path = self.path / "text_encoder" if (self.path / "text_encoder").exists() else self.path
        tok_path = self.path / "tokenizer" if (self.path / "tokenizer").exists() else text_path

        print(f"📦 Loading MOSS Text Encoder from: {text_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(tok_path))
        self.model = AutoModel.from_pretrained(
            str(text_path),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval()
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)
        print("✅ MOSS Text Encoder loaded successfully!")

    @torch.no_grad()
    def encode_prompt(self, prompt: str, max_sequence_length: int = 4096) -> mx.array:
        formatted = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=max_sequence_length,
        ).to(self.device)

        outputs = self.model(**inputs, use_cache=False, return_dict=True)
        hidden_state = outputs.last_hidden_state[0] # (seq_len, 4096)
        hidden_np = hidden_state.float().cpu().numpy()

        # Return MLX array of shape (1, seq_len, 4096)
        return mx.expand_dims(mx.array(hidden_np), axis=0)
