import os
from functools import lru_cache
from pathlib import Path

import platformdirs

BATTERY_PERCENTAGE_STOP_LIMIT = 5
CONTROLNET_STRENGTH = 0.4
DEFAULT_DEV_FILL_GUIDANCE = 30
DEFAULT_DEPTH_GUIDANCE = 10
DIMENSION_STEP_PIXELS = 16
GUIDANCE_SCALE = 3.5
GUIDANCE_SCALE_KONTEXT = 2.5
HEIGHT, WIDTH = 1024, 1024
IMAGE_STRENGTH = 0.4
DEFAULT_INFERENCE_STEPS = 25

# Keyed by the *canonical* AVAILABLE_MODELS key, never by alias: aliases are looked up
# through the registry by model_inference_steps() below. Keying this table by alias is
# what let `--model klein-4b` (and every other non-literal spelling) fall through to the
# 25-step FLUX.1-dev default.
MODEL_INFERENCE_STEPS = {
    "boogu-image-turbo": 4,
    "dev": 25,
    "dev-controlnet-canny": 25,
    "dev-controlnet-upscaler": 25,
    "dev-depth": 25,
    "dev-fill": 25,
    "dev-fill-catvton": 25,
    "dev-kontext": 25,
    "dev-redux": 25,
    "ernie-image": 50,
    "ernie-image-turbo": 8,
    "fibo": 50,
    "fibo-edit": 50,
    "fibo-edit-rmbg": 10,
    "fibo-lite": 8,
    "flux2-klein-4b": 4,
    "flux2-klein-9b": 4,
    "flux2-klein-9b-kv": 4,
    "flux2-klein-base-4b": 50,
    "flux2-klein-base-9b": 50,
    "ideogram-4-fp8": 20,
    "krea-2": 8,
    "krea-dev": 25,
    "lens-turbo": 4,
    "qwen-image": 20,
    "qwen-image-edit": 20,
    "schnell": 4,
    "schnell-controlnet-canny": 4,
    "z-image": 50,
    "z-image-turbo": 9,
    "z-image-turbo-controlnet-union-2.1": 8,
}


QUANTIZE_CHOICES = [3, 5, 4, 6, 8]

if os.environ.get("MFLUX_CACHE_DIR"):
    MFLUX_CACHE_DIR = Path(os.environ["MFLUX_CACHE_DIR"]).resolve()
else:
    MFLUX_CACHE_DIR = Path(platformdirs.user_cache_dir(appname="mflux"))

MFLUX_LORA_CACHE_DIR = MFLUX_CACHE_DIR / "loras"


@lru_cache(maxsize=1)
def model_choices() -> tuple[str, ...]:
    # Every spelling of a built-in model: canonical AVAILABLE_MODELS keys plus their
    # aliases. Hand-maintaining this list is what made `--model lens-turbo` (and ~40 other
    # valid names) resolve to model_path='lens-turbo' and die with "Model not found" —
    # anything missing here is treated as a local checkpoint directory.
    # Imported lazily for the same cycle reason as model_inference_steps() below.
    from mflux.models.common.config.model_config import AVAILABLE_MODELS

    names = set(AVAILABLE_MODELS)
    for config in AVAILABLE_MODELS.values():
        names.update(config.aliases)
    return tuple(sorted(names))


@lru_cache(maxsize=1)
def canonical_model_choices() -> tuple[str, ...]:
    # The canonical key of each registry entry, in registry (priority) order — the short
    # list worth printing in --help, where every alias would be noise.
    from mflux.models.common.config.model_config import AVAILABLE_MODELS

    return tuple(AVAILABLE_MODELS)


def model_inference_steps(model_name: str | None) -> int:
    # Accepts a canonical key, any alias, or a HuggingFace repo id. Unknown names —
    # third-party checkpoints and local paths — fall back to DEFAULT_INFERENCE_STEPS, as
    # does any registry entry with no declared count (the SeedVR2 upscalers never step).
    if model_name is None:
        return DEFAULT_INFERENCE_STEPS

    if model_name in MODEL_INFERENCE_STEPS:
        return MODEL_INFERENCE_STEPS[model_name]

    # Imported lazily: model_config pulls in mlx, and weight_loader / lora_resolution
    # already import this module, so a module-level import would close a cycle.
    from mflux.models.common.config.model_config import AVAILABLE_MODELS

    # Aliases are unique across the registry, so match them before repo ids.
    for key, config in AVAILABLE_MODELS.items():
        if model_name in config.aliases:
            return MODEL_INFERENCE_STEPS.get(key, DEFAULT_INFERENCE_STEPS)

    # Several entries share one repo id (z-image-turbo and its ControlNet, the FLUX.1-dev
    # ControlNets). Break the tie by priority, the same order ConfigResolution's
    # exact-match rule uses, so the step count matches the config that actually gets built.
    for key, config in sorted(AVAILABLE_MODELS.items(), key=lambda kv: kv[1].priority):
        if model_name == config.model_name:
            return MODEL_INFERENCE_STEPS.get(key, DEFAULT_INFERENCE_STEPS)

    return DEFAULT_INFERENCE_STEPS
