# Writes tests/resources/fibo2_transformer_parity.npz for tests/test_fibo2_parity.py: a tiny diffusers
# BriaFibo2Transformer2DModel with random weights, its inputs and its float32 CPU outputs. Run it in an environment with
# diffusers main at 48cdbae:
#   python scripts/fibo2/export_parity_reference.py tests/resources/fibo2_transformer_parity.npz
import argparse
import json
from pathlib import Path

import numpy as np
import torch

# Diffusers is only needed when regenerating this fixture, not by mflux or its test suite.
from diffusers import BriaFibo2Transformer2DModel  # ty: ignore[unresolved-import]

CONFIG = {
    "in_channels": 4,
    "patch_size": 2,
    "dim": 64,
    "n_layers": 4,
    "n_refiner_layers": 2,
    "n_heads": 2,
    "norm_eps": 1e-5,
    "cap_feat_dim": 48,
    "injection_layer_ids": (1, 3),
    "perceiver_num_layers": 2,
    "min_num_gist_tokens": 4,
    "max_num_gist_tokens": 8,
    "gist_step": 2,
    "gist_min_text_len": 4,
    "gist_max_text_len": 12,
    "rope_theta": 256.0,
    "t_scale": 1000.0,
    "axes_dims": (8, 12, 12),
    "axes_lens": (16, 8, 8),
}
# case: (text length, context images, timestep)
CASES = {
    "txt2img": (6, 0, 0.75),
    "short_text": (3, 0, 0.3),
    "long_text": (12, 0, 0.95),
    "edit": (6, 1, 0.6),
    "multi_edit": (12, 2, 0.4),
}


class Fibo2ParityExporter:
    @staticmethod
    def export(destination: Path) -> None:
        torch.manual_seed(0)
        model = BriaFibo2Transformer2DModel(**CONFIG).to(torch.float32).eval()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                # Random everywhere, the zero-initialised cross-attention gates and the unit norms included, so every
                # weight changes the output. Rounded to float16, which the fixture stores to stay small.
                base = 1.0 if parameter.ndim == 1 and ("norm" in name and "adaLN" not in name) else 0.0
                parameter.copy_((base + 0.2 * torch.randn_like(parameter)).half().float())
        data = {"config": np.array(json.dumps(CONFIG))}
        data.update({f"weight/{name}": value.detach().half().numpy() for name, value in model.state_dict().items()})
        latents = torch.randn(1, 4, 8, 8)
        contexts = [torch.randn(1, 4, 8, 8), torch.randn(1, 4, 6, 10)]
        texts = {length: torch.randn(1, 3, length, 48) for length in (3, 6, 12)}
        data["input/latents"] = latents.numpy()
        data["input/context_0"] = contexts[0].numpy()
        data["input/context_1"] = contexts[1].numpy()
        for length, text in texts.items():
            data[f"input/text_{length}"] = text.numpy()
        with torch.no_grad():
            for case, (length, num_contexts, sigma) in CASES.items():
                output = model(
                    hidden_states=latents,
                    timestep=torch.tensor([sigma]),
                    encoder_hidden_states=texts[length],
                    context_latents=contexts[:num_contexts] or None,
                    return_dict=False,
                )[0]
                data[f"expected/{case}"] = output.numpy()
                data[f"case/{case}"] = np.array([length, num_contexts, sigma], dtype=np.float32)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, **data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    Fibo2ParityExporter.export(parser.parse_args().destination)


if __name__ == "__main__":
    main()
