import contextlib
import logging
import sys
from collections.abc import Iterator

from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition

logger = logging.getLogger(__name__)


class PrequantizedResolution:
    # With -q N, a built-in model loads mflux-community/<stem>-mflux-q<N> instead of downloading the upstream bf16 weights
    # and quantizing them on every run (#938, d#794). Without -q nothing changes.
    ORG = "mflux-community"
    _enabled = True
    # Why a -q run on a model with a repo keeps the upstream weights (rules 4-6); weights_root logs these at info
    _BAKED = "LoRAs bake into the bf16 weights before quantizing"
    _MISSING = "{stem} has no {level} repo"
    _UPSTREAM_CACHED = "upstream already cached"

    @staticmethod
    def weights_root(
        model_config,
        model_path: str | None,
        quantize: int | None,
        has_loras: bool = False,
        bake_lora: bool = True,
        upstream_patterns: list[str] | None = None,
        saved_subdirs: list[str] | None = None,
    ) -> str:
        # upstream_patterns and saved_subdirs are what the loader reads: the download patterns of the upstream repo
        # and the component subdirs of an mflux-save. Without them the cache is not checked (rule 6 does not fire).
        root, reason = PrequantizedResolution.resolve(
            model_config, model_path, quantize, has_loras, bake_lora, upstream_patterns, saved_subdirs
        )
        redirected = model_path is None and root != model_config.model_name
        if redirected or PrequantizedResolution._explains_upstream(model_config, quantize, reason):
            logger.info(f"Weights: {root} ({reason})")
        else:
            logger.debug(f"Weights: {root} ({reason})")
        # The sidecar records the repo the weights came from; model_path stays the command line (#705)
        from mflux.utils.generated_image import GeneratedImage

        GeneratedImage.weights_repo = root if redirected else None
        return root

    @staticmethod
    def layout(definition, upstream_patterns: list[str] | None = None) -> dict[str, list[str]]:
        # The weights_root keyword arguments for a weight definition loaded by WeightLoader.load.
        if upstream_patterns is None:
            upstream_patterns = definition.get_download_patterns()
        saved_subdirs = list(ComponentDefinition.save_subdirs(definition.get_components()).values())
        return {"upstream_patterns": upstream_patterns, "saved_subdirs": saved_subdirs}

    @staticmethod
    def resolve(
        model_config,
        model_path: str | None,
        quantize: int | None,
        has_loras: bool = False,
        bake_lora: bool = True,
        upstream_patterns: list[str] | None = None,
        saved_subdirs: list[str] | None = None,
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
        stem = model_config.mflux_repo_stem
        if stem is None:
            return upstream, "no pre-quantized repo for this model"
        if has_loras and bake_lora:
            # A baked LoRA is folded into the bf16 weights before quantizing; a stored quantization has none to fold into
            return upstream, PrequantizedResolution._BAKED
        level = f"q{quantize}"
        if level in model_config.mflux_repo_missing:
            return upstream, PrequantizedResolution._MISSING.format(stem=stem, level=level)
        repo = PrequantizedResolution.repo_id(stem, quantize)
        upstream_cached = PrequantizedResolution._upstream_cached(upstream, upstream_patterns)
        if upstream_cached and not PrequantizedResolution._repo_cached(repo, saved_subdirs):
            # Someone who already has the upstream weights does not download a second copy (anthonywu, d#794)
            print(
                f"ℹ️  {repo} is a smaller download of this model; the cached upstream weights are used.", file=sys.stderr
            )
            return upstream, PrequantizedResolution._UPSTREAM_CACHED
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
    def _explains_upstream(model_config, quantize: int | None, reason: str) -> bool:
        missing = PrequantizedResolution._MISSING.format(stem=model_config.mflux_repo_stem, level=f"q{quantize}")
        return reason in (PrequantizedResolution._BAKED, PrequantizedResolution._UPSTREAM_CACHED, missing)

    @staticmethod
    def _upstream_cached(repo_id: str, patterns: list[str] | None) -> bool:
        # Cache only, no network, with the patterns the loader downloads: a partial snapshot (only vae/, or another
        # Anima variant's file) is not the model. Without patterns nothing can be said, so it counts as not cached.
        if not patterns:
            return False
        return PathResolution._find_complete_cached_snapshot(repo_id, patterns) is not None

    @staticmethod
    def _repo_cached(repo_id: str, saved_subdirs: list[str] | None) -> bool:
        # The pre-quantized repos are mflux-save uploads: shards and index in every component subdir.
        return PathResolution._find_complete_saved_snapshot(repo_id, saved_subdirs or []) is not None
