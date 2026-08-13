from pathlib import Path
from typing import TYPE_CHECKING

import transformers
from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError

from mflux.models.common.tokenizer.tokenizer import (
    BaseTokenizer,
    LanguageTokenizer,
    VisionLanguageTokenizer,
)

if TYPE_CHECKING:
    from mflux.models.common.weights.loading.weight_definition import TokenizerDefinition


class TokenizerLoader:
    @staticmethod
    def load(
        definition: "TokenizerDefinition",
        model_path: str,
    ) -> BaseTokenizer:
        tokenizer_path = TokenizerLoader._resolve_path(
            model_path=model_path,
            hf_subdir=definition.hf_subdir,
            fallback_subdirs=definition.fallback_subdirs,
            download_patterns=definition.download_patterns,
            tokenizer_class=definition.tokenizer_class,
        )

        raw_tokenizer = TokenizerLoader._load_raw_tokenizer(
            tokenizer_path=tokenizer_path,
            tokenizer_class=definition.tokenizer_class,
            chat_template=definition.chat_template,
        )

        return TokenizerLoader._create_tokenizer(
            raw_tokenizer=raw_tokenizer,
            definition=definition,
        )

    @staticmethod
    def load_all(
        definitions: list["TokenizerDefinition"],
        model_path: str,
        max_length_overrides: dict[str, int] | None = None,
    ) -> dict[str, BaseTokenizer]:
        max_length_overrides = max_length_overrides or {}
        result = {}
        for d in definitions:
            tokenizer = TokenizerLoader.load(d, model_path)
            if d.name in max_length_overrides:
                tokenizer.max_length = max_length_overrides[d.name]
            result[d.name] = tokenizer
        return result

    @staticmethod
    def _resolve_path(
        model_path: str,
        hf_subdir: str,
        fallback_subdirs: list[str] | None,
        download_patterns: list[str] | None,
        tokenizer_class: str,
    ) -> Path:
        # Resolution algorithm:
        # 1. Resolve a local path or an initial Hugging Face snapshot root.
        # 2. Prefer the primary tokenizer path and preserve real load failures.
        # 3. For cached HF snapshots, keep any loadable fallback layout before refreshing or failing.
        # 4. If a cached HF snapshot is incomplete, refresh it once over the network.
        # 5. Only then raise a targeted missing-tokenizer or unloadable-tokenizer error.
        patterns = download_patterns or [f"{hf_subdir}/**"]
        root_path, is_hf_repo, saw_cached_snapshot, download_error = TokenizerLoader._resolve_root_path(
            model_path=model_path,
            patterns=patterns,
        )

        if root_path is None:
            TokenizerLoader._raise_missing_tokenizer_error(
                model_path=model_path,
                hf_subdir=hf_subdir,
                fallback_subdirs=fallback_subdirs,
                is_hf_repo=is_hf_repo,
                saw_cached_snapshot=saw_cached_snapshot,
                download_error=download_error,
            )
        assert root_path is not None

        tokenizer_path, primary_load_error, primary_has_tokenizer_artifacts = TokenizerLoader._inspect_primary_path(
            root_path=root_path,
            hf_subdir=hf_subdir,
            tokenizer_class=tokenizer_class,
        )
        if tokenizer_path is not None:
            return tokenizer_path

        allow_cached_fallback = is_hf_repo and saw_cached_snapshot

        # If an alternate cached layout already works, keep the offline-first behavior.
        if allow_cached_fallback or not primary_has_tokenizer_artifacts:
            fallback_path = TokenizerLoader._resolve_fallback_path(root_path, fallback_subdirs, tokenizer_class)
            if fallback_path is not None:
                return fallback_path

        if is_hf_repo and saw_cached_snapshot:
            refreshed_root_path, download_error = TokenizerLoader._refresh_hf_root_path(
                model_path=model_path,
                patterns=patterns,
            )
            if refreshed_root_path is not None:
                root_path = refreshed_root_path
                tokenizer_path, primary_load_error, primary_has_tokenizer_artifacts = (
                    TokenizerLoader._inspect_primary_path(
                        root_path=root_path,
                        hf_subdir=hf_subdir,
                        tokenizer_class=tokenizer_class,
                    )
                )
                if tokenizer_path is not None:
                    return tokenizer_path

                if allow_cached_fallback or not primary_has_tokenizer_artifacts:
                    fallback_path = TokenizerLoader._resolve_fallback_path(root_path, fallback_subdirs, tokenizer_class)
                    if fallback_path is not None:
                        return fallback_path

        if primary_load_error is not None and primary_has_tokenizer_artifacts and download_error is None:
            TokenizerLoader._raise_unloadable_tokenizer_error(
                model_path=model_path,
                tokenizer_path=root_path / hf_subdir,
                is_hf_repo=is_hf_repo,
                load_error=primary_load_error,
            )

        TokenizerLoader._raise_missing_tokenizer_error(
            model_path=model_path,
            hf_subdir=hf_subdir,
            fallback_subdirs=fallback_subdirs,
            is_hf_repo=is_hf_repo,
            saw_cached_snapshot=saw_cached_snapshot,
            download_error=download_error,
        )

    @staticmethod
    def _resolve_root_path(
        model_path: str,
        patterns: list[str],
    ) -> tuple[Path | None, bool, bool, Exception | None]:
        expanded = Path(model_path).expanduser()
        if expanded.exists():
            return expanded, False, False, None
        if TokenizerLoader._is_huggingface_repo_id(model_path):
            root_path, saw_cached_snapshot, download_error = TokenizerLoader._resolve_hf_root_path(
                model_path=model_path,
                patterns=patterns,
            )
            return root_path, True, saw_cached_snapshot, download_error
        raise FileNotFoundError(
            f"Model not found: '{model_path}'. "
            f"If local path, make sure it exists. "
            f"If HuggingFace repo, use 'org/model' format."
        )

    @staticmethod
    def _is_huggingface_repo_id(model_path: str) -> bool:
        return "/" in model_path and model_path.count("/") == 1 and not model_path.startswith(("./", "../", "~/"))

    @staticmethod
    def _resolve_hf_root_path(
        model_path: str,
        patterns: list[str],
    ) -> tuple[Path | None, bool, Exception | None]:
        try:
            root_path = Path(
                snapshot_download(
                    repo_id=model_path,
                    allow_patterns=patterns,
                    local_files_only=True,
                )
            )
        except LocalEntryNotFoundError:
            root_path, download_error = TokenizerLoader._download_hf_root_path(
                model_path=model_path,
                patterns=patterns,
            )
            return root_path, False, download_error

        return root_path, True, None

    @staticmethod
    def _download_hf_root_path(
        model_path: str,
        patterns: list[str],
    ) -> tuple[Path | None, Exception | None]:
        try:
            root_path = Path(
                snapshot_download(
                    repo_id=model_path,
                    allow_patterns=patterns,
                )
            )
        except Exception as exc:  # noqa: BLE001
            return None, exc
        return root_path, None

    @staticmethod
    def _refresh_hf_root_path(
        model_path: str,
        patterns: list[str],
    ) -> tuple[Path | None, Exception | None]:
        return TokenizerLoader._download_hf_root_path(
            model_path=model_path,
            patterns=patterns,
        )

    @staticmethod
    def _inspect_primary_path(
        root_path: Path,
        hf_subdir: str,
        tokenizer_class: str,
    ) -> tuple[Path | None, Exception | None, bool]:
        tokenizer_path = root_path / hf_subdir
        is_primary_loadable, primary_load_error = TokenizerLoader._probe_tokenizer_path(
            tokenizer_path,
            tokenizer_class,
        )
        if is_primary_loadable:
            return tokenizer_path, None, True

        primary_has_tokenizer_artifacts = TokenizerLoader._has_tokenizer_artifacts(tokenizer_path)
        return None, primary_load_error, primary_has_tokenizer_artifacts

    @staticmethod
    def _is_tokenizer_loadable(path: Path, tokenizer_class: str) -> bool:
        is_loadable, _ = TokenizerLoader._probe_tokenizer_path(path, tokenizer_class)
        return is_loadable

    @staticmethod
    def _probe_tokenizer_path(path: Path, tokenizer_class: str) -> tuple[bool, Exception | None]:
        if not path.exists():
            return False, None

        TokenizerLoader._get_tokenizer_class(tokenizer_class)

        try:
            TokenizerLoader._load_raw_tokenizer(
                tokenizer_path=path,
                tokenizer_class=tokenizer_class,
            )
        except Exception as exc:  # noqa: BLE001
            return False, exc
        return True, None

    @staticmethod
    def _has_tokenizer_artifacts(path: Path) -> bool:
        if not path.exists():
            return False

        # Broad markers used only to distinguish an alternate layout from a broken tokenizer.
        tokenizer_artifacts = [
            "added_tokens.json",
            "merges.txt",
            "sentencepiece.bpe.model",
            "special_tokens_map.json",
            "spiece.model",
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "vocab.json",
            "vocab.txt",
        ]
        return any((path / filename).exists() for filename in tokenizer_artifacts)

    @staticmethod
    def _resolve_fallback_path(
        root_path: Path,
        fallback_subdirs: list[str] | None,
        tokenizer_class: str,
    ) -> Path | None:
        if not fallback_subdirs:
            return None

        for subdir in fallback_subdirs:
            fallback_path = root_path if subdir == "." else root_path / subdir
            if TokenizerLoader._is_tokenizer_loadable(fallback_path, tokenizer_class):
                return fallback_path

        return None

    @staticmethod
    def _raise_missing_tokenizer_error(
        model_path: str,
        hf_subdir: str,
        fallback_subdirs: list[str] | None,
        is_hf_repo: bool,
        saw_cached_snapshot: bool,
        download_error: Exception | None = None,
    ) -> None:
        primary_location = "repository root" if hf_subdir in ("", ".") else hf_subdir
        checked_locations = [repr(primary_location)]
        if fallback_subdirs:
            checked_locations.extend(
                repr("repository root" if subdir in ("", ".") else subdir) for subdir in fallback_subdirs
            )
        checked_summary = ", ".join(checked_locations)

        if is_hf_repo:
            if saw_cached_snapshot and download_error is not None:
                message = (
                    f"Incomplete Hugging Face tokenizer cache for '{model_path}'. "
                    f"No usable tokenizer files were found in {checked_summary}. "
                    f"Re-run with network access or clear/redownload the cache."
                )
            elif download_error is not None:
                message = (
                    f"Failed to download tokenizer files for Hugging Face repo '{model_path}'. "
                    f"Checked {checked_summary}."
                )
            else:
                message = (
                    f"No usable tokenizer files were found for Hugging Face repo '{model_path}'. "
                    f"Checked {checked_summary}."
                )
        else:
            message = (
                f"No usable tokenizer files were found for local model path '{model_path}'. Checked {checked_summary}."
            )

        if download_error is not None:
            raise FileNotFoundError(message) from download_error
        raise FileNotFoundError(message)

    @staticmethod
    def _raise_unloadable_tokenizer_error(
        model_path: str,
        tokenizer_path: Path,
        is_hf_repo: bool,
        load_error: Exception,
    ) -> None:
        location_type = "Hugging Face repo" if is_hf_repo else "local model path"
        raise RuntimeError(
            f"Found tokenizer files for {location_type} '{model_path}' at '{tokenizer_path}', but failed to load them."
        ) from load_error

    @staticmethod
    def _load_raw_tokenizer(
        tokenizer_path: Path,
        tokenizer_class: str,
        chat_template: str | None = None,
    ):
        cls = TokenizerLoader._get_tokenizer_class(tokenizer_class)

        # Workaround for Qwen2Tokenizer bug in transformers 5.0.0rc0
        # The tokenizer doesn't properly load vocab/merges from files, need to pass them directly
        if tokenizer_class == "Qwen2Tokenizer":
            tokenizer = TokenizerLoader._load_qwen2_tokenizer_workaround(tokenizer_path, cls)
        else:
            tokenizer = cls.from_pretrained(
                pretrained_model_name_or_path=str(tokenizer_path),
                local_files_only=True,
            )

        # Apply chat_template from definition if provided and not already set
        if chat_template and not getattr(tokenizer, "chat_template", None):
            tokenizer.chat_template = chat_template

        return tokenizer

    @staticmethod
    def _get_tokenizer_class(tokenizer_class: str):
        if hasattr(transformers, tokenizer_class):
            return getattr(transformers, tokenizer_class)
        raise ValueError(f"Unknown tokenizer class: {tokenizer_class}")

    @staticmethod
    def _load_qwen2_tokenizer_workaround(tokenizer_path: Path, cls):
        import json

        from tokenizers import AddedToken

        vocab_file = tokenizer_path / "vocab.json"
        merges_file = tokenizer_path / "merges.txt"
        config_file = tokenizer_path / "tokenizer_config.json"

        if not vocab_file.exists() or not merges_file.exists():
            # Fall back to standard loading if files don't exist
            return cls.from_pretrained(
                pretrained_model_name_or_path=str(tokenizer_path),
                local_files_only=True,
            )

        # Load vocab as dict[str, int]
        with open(vocab_file, encoding="utf-8") as f:
            vocab = json.load(f)

        # Load merges as list of tuples (pair of strings)
        # The merges.txt file has format: "token1 token2" per line (skip header if present)
        merges = []
        with open(merges_file, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()
                # Skip empty lines and the optional version header (e.g., "#version: 0.2")
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) == 2:
                    merges.append((parts[0], parts[1]))

        # Load tokenizer config for special tokens and chat template
        config_kwargs = {}
        chat_template = None
        if config_file.exists():
            with open(config_file, encoding="utf-8") as f:
                config = json.load(f)

            # Extract added_tokens_decoder to add special tokens
            added_tokens_decoder = config.get("added_tokens_decoder", {})
            if added_tokens_decoder:
                # Convert to the format expected by the tokenizer
                config_kwargs["added_tokens_decoder"] = {
                    int(k): AddedToken(
                        content=v["content"],
                        lstrip=v.get("lstrip", False),
                        rstrip=v.get("rstrip", False),
                        single_word=v.get("single_word", False),
                        normalized=v.get("normalized", False),
                        special=v.get("special", True),
                    )
                    for k, v in added_tokens_decoder.items()
                }

            # Extract chat_template if present
            chat_template = config.get("chat_template")

        tokenizer = cls(vocab=vocab, merges=merges, **config_kwargs)

        # Set chat_template after initialization (not a constructor param)
        if chat_template:
            tokenizer.chat_template = chat_template

        return tokenizer

    @staticmethod
    def _create_tokenizer(
        raw_tokenizer,
        definition: "TokenizerDefinition",
    ) -> BaseTokenizer:
        encoder_class = definition.encoder_class

        if encoder_class is VisionLanguageTokenizer:
            if definition.processor_class is None:
                raise ValueError("VisionLanguageTokenizer requires processor_class in definition")
            processor = definition.processor_class(tokenizer=raw_tokenizer)
            return VisionLanguageTokenizer(
                tokenizer=raw_tokenizer,
                processor=processor,
                max_length=definition.max_length,
                template=definition.template,
                image_token=definition.image_token,
            )
        elif encoder_class is not None and encoder_class is not LanguageTokenizer:
            return encoder_class(
                tokenizer=raw_tokenizer,
                max_length=definition.max_length,
            )
        else:
            # Default to LanguageTokenizer for all text-only cases
            return LanguageTokenizer(
                tokenizer=raw_tokenizer,
                max_length=definition.max_length,
                padding=definition.padding,
                template=definition.template,
                use_chat_template=definition.use_chat_template,
                chat_template_kwargs=definition.chat_template_kwargs or {},
                add_special_tokens=definition.add_special_tokens,
            )
