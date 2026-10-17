import argparse
import hashlib
import importlib.metadata
import inspect
import json
import platform
from pathlib import Path

import numpy as np
import torch

# Diffusers is only needed when regenerating this fixture, not by mflux or its test suite.
Flux3Transformer2DModel = importlib.import_module("diffusers").Flux3Transformer2DModel

REFERENCE_REVISION = "7ecb1e1b4fe500be7fb7381a7bb79a24d7fa3941"
PROFILES = ("fp32", "bf16_timestep", "bf16")


class BFloat16Timesteps(torch.nn.Module):
    def __init__(self, projection: torch.nn.Module):
        super().__init__()
        self.projection = projection

    def forward(self, scaled_timestep: torch.Tensor) -> torch.Tensor:
        # Diffusers passes t * 1000 in the stream dtype. Reproduce BFL's BF16 cast BEFORE that multiply.
        scaled_timestep = (scaled_timestep / 1000).to(torch.bfloat16) * 1000
        return self.projection(scaled_timestep.float()).to(torch.bfloat16).float()


class Flux3ParityExporter:
    CONFIG = {
        "in_channels": 8,
        "cond_in_channels": 8,
        "num_layers": 1,
        "num_single_layers": 2,
        "attention_head_dim": 8,
        "num_attention_heads": 4,
        "joint_attention_dim": 16,
        "timestep_channels": 256,
        "mlp_ratio": 2.0,
        "axes_dims_rope": [2, 2, 2, 2],
    }

    @staticmethod
    def export(destination: Path) -> None:
        if not torch.backends.mps.is_available():
            raise RuntimeError("This reference is exported on MPS. See README.md for the pinned environment.")
        distribution = importlib.metadata.distribution("diffusers")
        direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
        if REFERENCE_REVISION not in json.dumps(direct_url):
            raise RuntimeError(f"Install the pinned Diffusers revision {REFERENCE_REVISION} before exporting.")
        torch.manual_seed(20261016)
        inputs = {
            "hidden_states": torch.randn(1, 4, 8),
            "encoder_hidden_states": torch.randn(1, 3, 16),
            "timestep": torch.tensor([0.7137]),
            "img_ids": torch.tensor([[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 1, 0]]).float(),
            "txt_ids": torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]]).float(),
        }
        condition = {
            "image_cond_hidden_states": torch.randn(1, 4, 8),
            "image_cond_ids": torch.tensor([[0, 0, 0, -1], [0, 0, 1, -1], [0, 0, 0, -2], [0, 1, 0, -2]]).float(),
        }
        arrays = {f"input/{key}": value.numpy() for key, value in inputs.items()}
        arrays.update({f"condition/{key}": value.numpy() for key, value in condition.items()})
        arrays["input/target_timestep"] = np.array([0.3713], dtype=np.float32)
        for kind, flow_map in (("distilled", True), ("base", False)):
            torch.manual_seed(20261017)
            config = {**Flux3ParityExporter.CONFIG, "flow_map": flow_map}
            source = Flux3Transformer2DModel(**config).eval()
            state = source.state_dict()
            arrays[f"config/{kind}"] = np.array(json.dumps(config, sort_keys=True))
            arrays.update({f"weight/{kind}/{key}": value.numpy() for key, value in state.items()})
            for profile in PROFILES:
                model = Flux3Transformer2DModel(**config).eval()
                model.load_state_dict(state, strict=True)
                model.to("mps")
                if profile == "bf16_timestep":
                    model.time_proj = BFloat16Timesteps(model.time_proj)
                if profile == "bf16":
                    model.to(torch.bfloat16)
                    for name, module in model.named_modules():
                        if name.endswith(("norm_q", "norm_k")):
                            module.float()
                cases = ["txt2img", "reference", "multi_reference"]
                if flow_map:
                    cases.append("target_zero")
                for case in cases:
                    args = Flux3ParityExporter._inputs(inputs, condition, profile, case, flow_map)
                    with torch.inference_mode():
                        output = model(**args).sample.float().cpu().numpy()
                    arrays[f"expected/{kind}/{profile}/{case}"] = output
                del model
                torch.mps.empty_cache()
        source_path = Path(inspect.getfile(Flux3Transformer2DModel))
        metadata = {
            "diffusers_revision": REFERENCE_REVISION,
            "diffusers_version": distribution.version,
            "transformer_source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "macos_version": platform.mac_ver()[0],
            "device": "mps",
            "inputs_seed": 20261016,
            "weights_seed": 20261017,
            "profiles": list(PROFILES),
            "bf16_timestep_adapter": "Round t before multiplying by 1000; round sinusoidal output to BF16; other computation remains FP32.",
            "weights": "Synthetic random weights, no downloaded model checkpoint",
        }
        arrays["metadata"] = np.array(json.dumps(metadata, sort_keys=True))
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, **arrays)
        print(json.dumps({"destination": str(destination), "bytes": destination.stat().st_size, **metadata}, indent=2))

    @staticmethod
    def _inputs(
        inputs: dict[str, torch.Tensor],
        condition: dict[str, torch.Tensor],
        profile: str,
        case: str,
        flow_map: bool,
    ) -> dict[str, torch.Tensor]:
        args = {key: value.to("mps") for key, value in inputs.items()}
        if flow_map:
            args["target_timestep"] = torch.tensor([0.0 if case == "target_zero" else 0.3713], device="mps")
        if case in ("reference", "multi_reference"):
            count = 2 if case == "reference" else 4
            args["image_cond_hidden_states"] = condition["image_cond_hidden_states"][:, :count].to("mps")
            args["image_cond_ids"] = condition["image_cond_ids"][:count].to("mps")
        if profile == "bf16":
            for key in ("hidden_states", "encoder_hidden_states", "image_cond_hidden_states"):
                if key in args:
                    args[key] = args[key].to(torch.bfloat16)
        return args


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    Flux3ParityExporter.export(parser.parse_args().destination)


if __name__ == "__main__":
    main()
