"""Find the Z-Image Turbo bf16 transformer checkpoint without assuming a fixed mount."""

import glob
import json
import os
from pathlib import Path

REPO = "models--Tongyi-MAI--Z-Image-Turbo"


def find_transformer_dir() -> str:
    """Order: MFLUX_ZIMAGE_CHECKPOINT (transformer dir), HF_HUB_CACHE, HF_HOME/hub, ~/.cache/huggingface/hub."""
    env = os.environ.get("MFLUX_ZIMAGE_CHECKPOINT")
    if env:
        if not (Path(env) / "diffusion_pytorch_model.safetensors.index.json").exists():
            raise FileNotFoundError(f"MFLUX_ZIMAGE_CHECKPOINT has no safetensors index: {env}")
        return env.rstrip("/") + "/"
    roots = []
    if os.environ.get("HF_HUB_CACHE"):
        roots.append(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        roots.append(os.path.join(os.environ["HF_HOME"], "hub"))
    roots.append(os.path.expanduser("~/.cache/huggingface/hub"))
    for root in roots:
        hits = sorted(
            h
            for h in glob.glob(f"{root}/{REPO}/snapshots/*/transformer/")
            if os.path.exists(h + "diffusion_pytorch_model.safetensors.index.json")
        )
        if hits:
            return hits[-1]
    raise FileNotFoundError(
        "Z-Image Turbo bf16 checkpoint not found. Set MFLUX_ZIMAGE_CHECKPOINT to the transformer folder, "
        f"or set HF_HUB_CACHE. Searched: {roots}"
    )


def load_bf16_weight(key: str):
    from safetensors import safe_open

    snap = find_transformer_dir()
    shard = json.load(open(snap + "diffusion_pytorch_model.safetensors.index.json"))["weight_map"][key]
    with safe_open(snap + shard, "pt") as f:
        return f.get_tensor(key).float().numpy()
