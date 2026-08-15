# Justfile Guide

### What & Why
Developers use command runners like make and just as CLI macros... named shortcuts for repeatable development tasks. 

[**Just**](https://github.com/casey/just) is simpler for task automation: it has cleaner syntax, clearer errors, and avoids [**Make**](https://en.wikipedia.org/wiki/Make_(software))’s build-system quirks. [**Make**](https://en.wikipedia.org/wiki/Make_(software)) is better when you need dependency-based incremental builds; [**Just**](https://github.com/casey/just) is better for commands like `test`, `lint`, and `release`.

[@anthonywu](https://github.com/anthonywu) proposed JustFile in a [MFlux-Community Discussion](https://github.com/orgs/mflux-community/discussions/585#discussion-10610831) in August 2026: _"This is the modern alternative to Makefiles, the syntax is much more readable to humans. In my two years working with just, AI assistants big and small have had no trouble helping me write these files."_


## Installing

<sub>as MFlux is targetting - we presume you are running on M-Series Mac - ie `aarch64`</sub>

```
brew install just
```


## Common MFLux Workflow

```sh
just install
just lint
just test-fast
```

`just install` creates `.venv`, synchronizes dependencies, and installs
pre-commit hooks. If `pre-commit` is not installed, the recipe installs it as
a uv tool.

## Recipes

| Command | Purpose |
| --- | --- |
| `just` | List public recipes. |
| `just all` | Install dependencies, then run the default test selection. |
| `just install` | Create the environment, synchronize dependencies, and install pre-commit hooks. |
| `just venv-init` | Create a clean Python 3.13 virtual environment after checking the platform and uv. |
| `just lint` | Run Ruff checks without changing files. |
| `just lint-justfile` | Verify that `just --fmt` would not reformat the justfile. |
| `just format` | Run Ruff's formatter and show a summary of changed files. |
| `just check` | Run all pre-commit hooks, including auto-fixes and formatters. Review changes afterwards. |
| `just test` | Run the default pytest selection, which excludes slow model tests. |
| `just test-fast` | Run tests marked `fast`; these do not generate images. |
| `just test-slow` | Run tests marked `slow`; these generate images. |
| `just test-all` | Run all tests except those marked `high_memory_requirement`. This can download model weights. |
| `just build` | Build sdist/wheel into `dist/`, then report artifact sizes and flag any oversized files in the sdist (a check that image assets weren't accidentally bundled). |
| `just clean` | Remove `.venv`. Run `just install` to recreate it. |

Test recipes first synchronize the locked environment with all extras, then
run pytest with `MFLUX_PRESERVE_TEST_OUTPUT=1` so generated outputs remain
available for inspection. They do not update golden images.

`just build` removes any stale `dist/mflux-*` artifacts, builds fresh sdist and
wheel packages with `uv build`, then reports their sizes (expected under 1MB)
and lists the five largest files inside the sdist — a quick sanity check that
no image outputs got swept into the package by mistake.

## Internal Recipes
`expect-arm64`, `expect-uv`, `ensure-pre-commit`, and `_test-run` are private
helpers used by public recipes. They are not intended to be invoked directly.

<sub>Saturday 15 August -  v0.1 &nbsp; | &nbsp; by [@ianscrivener](https://github.com/ianscrivener), GPT 5.6 Terra & Claude Sonnet 5
