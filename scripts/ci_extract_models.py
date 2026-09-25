"""ci_extract_models.py: dump mflux's supported-model registry as JSON.

Mechanically extracted from mflux.models.common.config.model_config.AVAILABLE_MODELS:
model key, aliases, upstream repo, controlnet repo, custom-transformer repo.

Everything else (model_type/model_family/model_sub_family, mflux_cli,
mflux_cli_tools, upstream.status, upstream.license, quants) has no single
source of truth in this repo -- pyproject.toml's [project.scripts] lists CLI
commands but not which model each one defaults to, and taxonomy/license/status
aren't recorded anywhere in-repo. Those fields live in the OVERLAY table below
and must be updated by hand when a model is added, renamed, or reclassified.

A model key present in AVAILABLE_MODELS but missing from OVERLAY fails the
run rather than publishing null taxonomy fields -- add its OVERLAY row first.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from mflux.models.common.config.model_config import AVAILABLE_MODELS  # noqa: E402

OUTPUT_PATH = REPO_ROOT / ".ci_cache" / "models_mflux.json"

CANONICAL_REPO_URL = "https://github.com/mflux-community/mflux"


def _resolve_repo_url() -> str:
    """The repo this extraction actually ran in, not necessarily canonical upstream."""
    github_repository = os.environ.get("GITHUB_REPOSITORY")
    if github_repository:
        return f"https://github.com/{github_repository}"
    try:
        remote = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return CANONICAL_REPO_URL
    if remote.startswith("git@github.com:"):
        return "https://github.com/" + remote.removeprefix("git@github.com:").removesuffix(".git")
    return remote.removesuffix(".git")


STANDARD_QUANTS = ["q3", "q4", "q5", "q6", "q8", "bf16"]

# key -> (model_type, model_family, model_sub_family, mflux_cli, mflux_cli_tools, status, quants)
OVERLAY: dict[str, tuple[str, str, str, list[str], list[str], str, list[str] | None]] = {
    "dev": ("image", "flux1", "flux1-dev", ["mflux-generate"], [], "maintenance", STANDARD_QUANTS),
    "schnell": ("image", "flux1", "flux1-schnell", ["mflux-generate"], [], "maintenance", STANDARD_QUANTS),
    "dev-kontext": ("image", "flux1", "flux1-kontext", ["mflux-generate-kontext"], [], "maintenance", STANDARD_QUANTS),
    "dev-fill": ("image", "flux1", "flux1-fill", ["mflux-generate-fill"], [], "maintenance", STANDARD_QUANTS),
    "dev-redux": ("image", "flux1", "flux1-redux", ["mflux-generate-redux"], [], "maintenance", STANDARD_QUANTS),
    "dev-depth": ("image", "flux1", "flux1-depth", ["mflux-generate-depth"], [], "maintenance", STANDARD_QUANTS),
    "dev-controlnet-canny": (
        "image",
        "flux1",
        "flux1-controlnet",
        ["mflux-generate-controlnet"],
        [],
        "active",
        STANDARD_QUANTS,
    ),
    "schnell-controlnet-canny": (
        "image",
        "flux1",
        "flux1-controlnet",
        ["mflux-generate-controlnet"],
        [],
        "active",
        STANDARD_QUANTS,
    ),
    "dev-controlnet-upscaler": (
        "image",
        "flux1",
        "flux1-controlnet",
        ["mflux-upscale-controlnet"],
        [],
        "active",
        STANDARD_QUANTS,
    ),
    "dev-fill-catvton": (
        "image",
        "flux1",
        "flux1-fill",
        ["mflux-generate-in-context-catvton"],
        [],
        "experimental",
        STANDARD_QUANTS,
    ),
    "krea-dev": ("image", "flux1", "flux1-krea", ["mflux-generate"], [], "maintenance", STANDARD_QUANTS),
    "flux2-klein-4b": (
        "image",
        "flux2",
        "flux2-klein",
        ["mflux-generate-flux2", "mflux-generate-flux2-edit"],
        [],
        "active",
        STANDARD_QUANTS,
    ),
    "flux2-klein-9b": (
        "image",
        "flux2",
        "flux2-klein",
        ["mflux-generate-flux2", "mflux-generate-flux2-edit"],
        [],
        "active",
        STANDARD_QUANTS,
    ),
    "flux2-klein-9b-kv": (
        "image",
        "flux2",
        "flux2-klein",
        ["mflux-generate-flux2", "mflux-generate-flux2-edit"],
        [],
        "active",
        STANDARD_QUANTS,
    ),
    "flux2-klein-base-4b": (
        "image",
        "flux2",
        "flux2-klein-base",
        ["mflux-generate-flux2", "mflux-generate-flux2-edit"],
        [],
        "active",
        STANDARD_QUANTS,
    ),
    "flux2-klein-base-9b": (
        "image",
        "flux2",
        "flux2-klein-base",
        ["mflux-generate-flux2", "mflux-generate-flux2-edit"],
        [],
        "active",
        STANDARD_QUANTS,
    ),
    "krea-2": ("image", "krea2", "krea2-turbo", ["mflux-generate-krea2"], [], "active", STANDARD_QUANTS),
    "krea-2-raw": ("image", "krea2", "krea2-raw", ["mflux-generate-krea2"], [], "active", STANDARD_QUANTS),
    "qwen-image": ("image", "qwen-image", "qwen-image", ["mflux-generate-qwen"], [], "active", STANDARD_QUANTS),
    "qwen-image-edit": (
        "image",
        "qwen-image",
        "qwen-image-edit",
        ["mflux-generate-qwen-edit"],
        [],
        "active",
        STANDARD_QUANTS,
    ),
    "qwen-image-2.1": (
        "image",
        "qwen-image",
        "qwen-image-2.1",
        ["mflux-generate-qwen-2.1"],
        [],
        "active",
        STANDARD_QUANTS,
    ),
    "fibo": (
        "image",
        "fibo",
        "fibo",
        ["mflux-generate-fibo"],
        ["mflux-refine-fibo", "mflux-inspire-fibo"],
        "active",
        None,
    ),
    "fibo-lite": (
        "image",
        "fibo",
        "fibo-lite",
        ["mflux-generate-fibo"],
        ["mflux-refine-fibo", "mflux-inspire-fibo"],
        "active",
        None,
    ),
    "fibo-edit": (
        "image",
        "fibo",
        "fibo-edit",
        ["mflux-generate-fibo-edit"],
        ["mflux-refine-fibo", "mflux-inspire-fibo"],
        "active",
        None,
    ),
    "fibo-edit-rmbg": (
        "image",
        "fibo",
        "fibo-edit",
        ["mflux-generate-fibo-edit"],
        ["mflux-refine-fibo", "mflux-inspire-fibo"],
        "active",
        None,
    ),
    "z-image": ("image", "z-image", "z-image", ["mflux-generate-z-image"], [], "active", STANDARD_QUANTS),
    "z-image-turbo": (
        "image",
        "z-image",
        "z-image-turbo",
        ["mflux-generate-z-image-turbo"],
        [],
        "active",
        STANDARD_QUANTS,
    ),
    "z-image-turbo-controlnet-union-2.1": (
        "image",
        "z-image",
        "z-image-controlnet",
        ["mflux-generate-z-image-controlnet"],
        [],
        "active",
        STANDARD_QUANTS,
    ),
    "ernie-image": (
        "image",
        "ernie-image",
        "ernie-image",
        ["mflux-generate-ernie-image"],
        [],
        "active",
        STANDARD_QUANTS,
    ),
    "ernie-image-turbo": (
        "image",
        "ernie-image",
        "ernie-image-turbo",
        ["mflux-generate-ernie-image-turbo"],
        [],
        "active",
        STANDARD_QUANTS,
    ),
    "ideogram-4-fp8": (
        "image",
        "ideogram4",
        "ideogram4-fp8",
        ["mflux-generate-ideogram4"],
        [],
        "active",
        STANDARD_QUANTS,
    ),
    "boogu-image-turbo": (
        "image",
        "boogu",
        "boogu-image-turbo",
        ["mflux-generate-boogu"],
        [],
        "active",
        STANDARD_QUANTS,
    ),
    "lens-turbo": ("image", "lens", "lens-turbo", ["mflux-generate-lens"], [], "active", STANDARD_QUANTS),
    "ming-image-design": (
        "image",
        "ming-image",
        "ming-image-design",
        ["mflux-generate-ming"],
        [],
        "active",
        STANDARD_QUANTS,
    ),
    "seedvr2-3b": ("video-upscale", "seedvr2", "seedvr2-3b", ["mflux-upscale-seedvr2"], [], "active", STANDARD_QUANTS),
    "seedvr2-7b": ("video-upscale", "seedvr2", "seedvr2-7b", ["mflux-upscale-seedvr2"], [], "active", STANDARD_QUANTS),
}

# Registry keys with no ModelConfig entry (standalone tools wired only via pyproject.toml scripts).
EXTRA_ENTRIES: dict[str, dict[str, Any]] = {
    "depth-pro": {
        "model_type": "depth-estimation",
        "model_family": "depth-pro",
        "model_sub_family": "depth-pro",
        "model_aliases": [],
        "upstream": {"repo": "apple/DepthPro", "license": None, "status": "active"},
        "mflux_cli": [],
        "mflux_cli_tools": ["mflux-save-depth"],
    },
}


def build_registry() -> dict[str, Any]:
    registry: dict[str, Any] = {}

    for key, config in AVAILABLE_MODELS.items():
        upstream: dict[str, Any] = {"repo": config.model_name}
        if config.controlnet_model is not None:
            upstream["controlnet_repo"] = config.controlnet_model
        if config.custom_transformer_model is not None:
            upstream["custom_transformer_repo"] = config.custom_transformer_model

        overlay = OVERLAY.get(key)
        if overlay is None:
            model_type = model_family = model_sub_family = status = None
            mflux_cli: list[str] = []
            mflux_cli_tools: list[str] = []
            quants = None
        else:
            model_type, model_family, model_sub_family, mflux_cli, mflux_cli_tools, status, quants = overlay

        upstream["license"] = None
        upstream["status"] = status

        entry: dict[str, Any] = {
            "model_type": model_type,
            "model_family": model_family,
            "model_sub_family": model_sub_family,
            "model_aliases": list(config.aliases),
            "upstream": upstream,
            "mflux_cli": mflux_cli,
            "mflux_cli_tools": mflux_cli_tools,
        }
        if quants is not None:
            entry["quants"] = quants
        registry[key] = entry

    collisions = registry.keys() & EXTRA_ENTRIES.keys()
    if collisions:
        raise ValueError(
            f"EXTRA_ENTRIES key(s) {sorted(collisions)} already present in AVAILABLE_MODELS; "
            "remove the now-redundant EXTRA_ENTRIES entry."
        )
    registry.update(EXTRA_ENTRIES)
    return dict(sorted(registry.items()))


def _value_counts(registry: dict[str, Any], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in registry.values():
        value = entry[field]
        if value is not None:
            counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def build_metadata(registry: dict[str, Any]) -> dict[str, Any]:
    unique_quants = sorted({q for entry in registry.values() for q in entry.get("quants", [])})

    return {
        "datetime_extract": datetime.now(timezone.utc).isoformat(),
        "repo": _resolve_repo_url(),
        "counts": {
            "model_count": len(registry),
            "model_type_count": _value_counts(registry, "model_type"),
            "model_family_count": _value_counts(registry, "model_family"),
            "unique_quants": unique_quants,
        },
    }


def main() -> None:
    registry = build_registry()
    missing_overlay = [k for k in AVAILABLE_MODELS if k not in OVERLAY]
    if missing_overlay:
        sys.exit(
            f"error: no OVERLAY entry for: {', '.join(sorted(missing_overlay))} "
            "-- add a row to OVERLAY in scripts/ci_extract_models.py before publishing."
        )

    document = {
        "metadata": build_metadata(registry),
        "models": [{key: entry} for key, entry in registry.items()],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(document, indent=2) + "\n")
    print(f"wrote {len(registry)} models to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
