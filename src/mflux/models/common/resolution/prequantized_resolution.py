import contextlib
import logging
import os
import sys
from collections.abc import Iterator
from pathlib import Path

from mflux.models.common.resolution.path_resolution import PathResolution

logger = logging.getLogger(__name__)


class PrequantizedResolution:
    # With -q N, a built-in model loads mflux-community/<stem>-mflux-q<N> instead of downloading the upstream bf16 weights
    # and quantizing them on every run (#938, d#794). Without -q nothing changes.
    ORG = "mflux-community"
    _enabled = True

    @staticmethod
    def weights_root(
        model_config,
        model_path: str | None,
        quantize: int | None,
        has_loras: bool = False,
        bake_lora: bool = True,
    ) -> str:
        root, reason = PrequantizedResolution.resolve(model_config, model_path, quantize, has_loras, bake_lora)
        redirected = model_path is None and root != model_config.model_name
        if redirected or (quantize is not None and getattr(model_config, "mflux_repo_stem", None) and not model_path):
            logger.info(f"Weights: {root} ({reason})")
        else:
            logger.debug(f"Weights: {root} ({reason})")
        # The sidecar records the repo the weights came from; model_path stays the command line (#705)
        from mflux.utils.generated_image import GeneratedImage

        GeneratedImage.weights_repo = root if redirected else None
        return root

    @staticmethod
    def resolve(
        model_config,
        model_path: str | None,
        quantize: int | None,
        has_loras: bool = False,
        bake_lora: bool = True,
    ) -> tuple[str, str]:
        upstream = model_config.model_name
        if model_path:
            return model_path, "path given"
        if model_config.base_model is not None:
            # A config inferred from a third-party name carries its base's fields, the stem too
            return upstream, "third-party model"
        if quantize is None:
            return upstream, "no -q"
        if not PrequantizedResolution._enabled:
            return upstream, "pre-quantized repos off for this command"
        stem = getattr(model_config, "mflux_repo_stem", None)
        if stem is None:
            return upstream, "no pre-quantized repo for this model"
        if has_loras and bake_lora:
            # A baked LoRA is folded into the bf16 weights before quantizing; a stored quantization has none to fold into
            return upstream, "LoRAs bake into the bf16 weights before quantizing"
        level = f"q{quantize}"
        if level in getattr(model_config, "mflux_repo_missing", ()):
            return upstream, f"{stem} has no {level} repo"
        repo = PrequantizedResolution.repo_id(stem, quantize)
        if PrequantizedResolution._cached(upstream) and not PrequantizedResolution._cached(repo):
            # Someone who already has the upstream weights does not download a second copy (anthonywu, d#794)
            print(
                f"ℹ️  {repo} is a smaller download of this model; the cached upstream weights are used.", file=sys.stderr
            )
            return upstream, "upstream already cached"
        return repo, f"-q {quantize}"

    @staticmethod
    def repo_id(stem: str, quantize: int) -> str:
        return f"{PrequantizedResolution.ORG}/{stem}-mflux-q{quantize}"

    @staticmethod
    @contextlib.contextmanager
    def disabled() -> Iterator[None]:
        # mflux-train and mflux-save keep the upstream weights in this release
        previous = PrequantizedResolution._enabled
        PrequantizedResolution._enabled = False
        try:
            yield
        finally:
            PrequantizedResolution._enabled = previous

    @staticmethod
    def _cached(repo_id: str) -> bool:
        # Cache only, no network. Weights sit in component folders (transformer/, vae/, ...), so the pattern recurses;
        # every index in the snapshot must have its shards on disk.
        snapshot = PathResolution._find_complete_cached_snapshot(repo_id, ["**/*.safetensors"])
        if snapshot is None:
            return False
        return all(
            PathResolution._indexed_shards_present(Path(directory))
            for directory, _, files in os.walk(snapshot, followlinks=True)
            if any(name.endswith(".safetensors.index.json") for name in files)
        )
