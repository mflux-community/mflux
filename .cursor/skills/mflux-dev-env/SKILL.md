---
name: mflux-dev-env
description: Set up and work in the mflux dev environment (arm64 expectation, uv, justfile recipes, lint/format/test).
---
# mflux dev environment

This repo expects macOS arm64 and prefers `uv` + justfile recipes.

## When to Use

- You’re setting up the repo locally or diagnosing environment/setup issues.
- You need the canonical way to run lint/format/check/build/test.

## Instructions

- Prereq: `just` ≥ 1.50 (`brew install just`). CI lints the justfile with `just --fmt --check`; run `just fmt-justfile` to auto-fix formatting.
- Prefer justfile recipes:
  - Install: `just install`
  - Lint: `just lint`
  - Format: `just format`
  - Pre-commit suite: `just check`
  - Build: `just build`
- Prefer `uv run ...` for running Python commands to ensure the correct environment.
- When running tests, keep `MFLUX_PRESERVE_TEST_OUTPUT=1` enabled (the justfile test recipes already do this).

## Type checking (ty)

- The type checker is astral's `ty`, pinned exactly in `pyproject.toml` dev deps (mirrored in `.pre-commit-config.yaml` and run in CI). Run it with `just typecheck` (or `uv run ty check`).
- `[tool.ty.rules]` in `pyproject.toml` carries a one-time migration baseline (mypy → ty): every rule listed there had pre-existing violations and is set to `"ignore"`, with violation counts in comments.
- **Planned burn-down**: we intend to re-enable these rules piecemeal — delete one ignore line, fix all resulting `ty check` errors, commit, repeat. Prefer starting with low-count rules. Do not add new ignore lines; new code must pass all rules not in the baseline.

