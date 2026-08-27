# mflux – Agent Rules (Project Rules)

These rules exist to make agent work in this repo **predictable, verifiable, and low-drama**.

## Commands / environment

- **Install `just` ≥ 1.50** (`brew install just`): all repo workflows run through the justfile — bare `just` (or `just --list`) shows every recipe. The version floor is set by `just --fmt` (stabilized in 1.50), which CI enforces via `just lint-justfile`; fix formatting with `just fmt-justfile`.
- **Always use uv** for dependency management and running code.
  - Run scripts/binaries with `uv run <command>`.
  - Prefer `uv run python -m ...` for local modules.
  - Manage deps with `uv add <pkg>` / `uv remove <pkg>`.
- **Tool installs (CLI executables)**:
  - When you need to (re)install the local checkout as a `uv tool` (e.g. after changing CLI code), prefer an **editable install**:
    - `uv tool install --force --editable --reinstall .`
- **Prefer justfile recipes** when they exist (they encode project-specific setup):
  - `just install`, `just lint`, `just format`, `just test-fast`, `just test`, `just build`.

## Tests (goldens / image output)

- **Always preserve test outputs** (for visual inspection): run tests with `MFLUX_PRESERVE_TEST_OUTPUT=1` (the justfile test recipes already do this).
- **Do not update reference (“golden”) images** unless explicitly asked.
- Prefer faster scopes first (`just test-fast` → `just test-slow` → `just test`).
- For the full playbook (how to handle failures and golden diffs), use the `mflux-testing` skill.

## Lint / format

- Use the justfile recipes for repo workflows: `just lint`, `just format`, `just check`.

## CI / GitHub Actions

- **Pin every `uses:` to a full commit SHA** with the release version as an inline comment, e.g. `uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262  # v4.4.0`. Never reference moving tags/branches (`@v4`, `@release/v1`) — tag repointing is a supply-chain attack vector, especially in the trusted-publishing release job.
- When adding or updating an action, pin the SHA of the **latest release tag within the currently-used major** (resolve annotated tags down to the commit). Bump majors deliberately in their own PR. **Exception:** when the current major's node runtime is deprecated (e.g. the `setup-python` v4/node16 → v7/node24 and `setup-uv` v4/node20 → v10/node24 jumps), go straight to the latest major — freezing on an EOL runtime is the bigger risk.
- Pinned tool versions have a **canonical home with pointing mirrors**: `ruff` is declared in `pyproject.toml` dev deps; `.pre-commit-config.yaml` and the CI lint job mirror it with pointer comments (pre-commit must pin its own rev, so perfect single-sourcing is impossible), and the justfile derives its ruff version from `pyproject.toml` at runtime. `uv` itself is pinned in CI/release via `setup-uv`'s `version` input so `uv.lock` rendering is reproducible across machines.

## Code style

- Avoid docstrings; prefer clear naming and focused helpers.
- Add comments only when logic is non-obvious; keep them short.
- Consider deeper modules and clear APIs over over-fragmented tiny functions; weigh the tradeoff.
- Prefer composition over inheritance when practical; avoid deep class hierarchies.
- Static methods are fine when they clarify stateless helpers.
- Avoid free-standing helper functions in Python modules; prefer placing helpers on proper classes (even if static). Thin module-level entrypoints like CLI `main()` functions are fine.
- Keep private methods (leading underscore) at the bottom of classes; public APIs at the top.
- Use type hints consistently for public APIs.
- Keep CLI entrypoints thin: parse args, resolve config, construct the model, register callbacks, run, save.
- Prefer extending existing shared abstractions over adding model-specific one-off paths unless the model truly needs a new abstraction.
- Reuse shared plumbing before adding bespoke code paths: `CommandLineParser`, `ModelConfig`, `CallbackManager`, `DimensionResolver`, shared schedulers, and existing save/metadata helpers.
- Keep public model APIs small and obvious; prefer a clear top-level method like `generate_image(...)` with implementation details pushed into private helpers.
- Keep one clear source of truth for model defaults and reflect those defaults consistently in CLI behavior, Python API examples, and README docs.
- Treat README examples as part of the API contract. If behavior, defaults, or flags change, update the relevant examples in the same pass.

## Releases

- When preparing a release, prefer the `mflux-release` skill.
- Tagging/publishing is handled by an external GitHub Action.

## Git safety

- **Never push or force-push** to the remote repository without explicit user approval for each push.
- Committing locally is fine and encouraged for progress tracking.
- Before any major squash/rewrite of local branch history, create a **local backup branch** at the current `HEAD` first (for example `backup/<branch>-pre-squash-YYYYMMDD-HHMM` so repeated backups stay unique to the minute).
- That backup branch is intentionally **local-only** for safety and rollback; do not push it unless the user explicitly asks.

## Agent workflow norms (modern agent best practices)

- For multi-file or high-risk work, **start with a short plan** (bullets: goals, constraints, files to touch, how you’ll verify).
- **Plan Mode Enforcement**: For any non-trivial task or high architectural risk, save your plan to `.agents/plans/YYYY-MM-DD-feature-name.md` and ask for approval before coding.
- Keep changes tight, and prefer **verifiable goals** (tests/lint/build) over speculation.
- If the task scope changes materially, stop and re-align rather than continuing in a confused state.
- When users ask for CLI usage (e.g., “Can you help me generate an image using z-image?”), use the `mflux-cli` skill.
- For new models or major feature additions, the definition of done usually includes shared config wiring, a thin CLI entrypoint, verification coverage, and a README/example update that matches existing model docs.

## Bug/behavior reporting format (chat)

- When reporting bugs or behaviors, always use a simple story format.
- Use separate "Scenario" sections for separate issues.
- Each scenario must be step-by-step and end with a one-line "Fix:".

Example:
Scenario — Late preview crash
1) User starts edit training without data/preview.*.
2) Run crashes on first preview step.
Fix: Validate preview image at config load.

## Skills

- For image generation requests, **always use** `mflux-cli` to find the right command and flags.
- Use `mflux-cli` for CLI capability discovery and usage help.
- Use `mflux-dev-env` for setup, uv usage, and justfile recipes.
- Use `mflux-testing` for running tests and handling golden images.
- Use `mflux-manual-testing` for validating CLI outputs manually.
- Use `mflux-debugging` for MLX vs PyTorch/diffusers comparisons.
- Use `mflux-model-porting` when porting models into MLX.
- Use `mflux-release` for release preparation steps.
- Use `mflux-pr` for preparing clean PRs.

