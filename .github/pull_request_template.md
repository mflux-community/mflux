## What

<!-- One short paragraph: what changes and why. Link issues: Fixes #... -->

## Checklist (definition of done)

- [ ] Tests added/updated run in CI by default (selector: `-m "not slow and not high_memory_requirement"`, same as `just test`). Only mark `@pytest.mark.slow` or `@pytest.mark.high_memory_requirement` when a test exceeds CI's time or memory budget (typically weight downloads / image generation).
- [ ] `ruff check` and `ruff format` are clean (`uv run ruff` uses the version pinned in the dev dependencies of `pyproject.toml`, which is the single source of truth for pre-commit and CI; `pre-commit run -a` covers it locally).
- [ ] `CHANGELOG.md`: entry under `Unreleased` referencing this PR number.
- [ ] Docs updated where behavior changed — README examples/table rows are part of the API contract (see `AGENTS.md`).
- [ ] New model: shared config wiring (aliases, default steps, mflux-save dispatch, capabilities, completions), thin CLI entrypoint, and `src/mflux/models/<name>/README.md`.
- [ ] New/changed CLI: ignored/rejected options declared (`IGNORED_OPTIONS`/`REJECTED_OPTIONS`) and `warn_ignored_options` actually called in `main()` — `mflux-capabilities` must stay truthful.

## Verification

<!-- Commands you ran and what you observed. Include generated images/screenshots for model-affecting changes. -->
