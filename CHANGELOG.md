# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### 🐛 Bug Fixes

- **SeedVR2 upscaling runs on mlx 0.32.2, and its window attention is correct for more than one batch element**: the window attention expanded per-batch values to per-window ones with `mx.repeat(x, mx.array(window_counts), axis=0)`, but `mx.repeat` only takes a scalar repeat count. Up to mlx 0.32.1 pybind11 coerced a *one-element* array to an int, so a single image happened to work; mlx 0.32.2 dropped that coercion and every `mflux-upscale-seedvr2` run now dies before the first step with `TypeError: repeat(): incompatible function arguments`, inside the `mlx>=0.32.0,<0.33.0` range mflux itself declares (the dev group pins 0.32.0, which is why CI never saw it). A second batch element made `window_counts` longer than one and raised the same TypeError on *every* mlx version tested back to 0.31.2, so this was always a latent bug that a batch of one was hiding. All four call sites now gather with a window→batch index built once per attention call, which is `np.repeat(x, counts, axis=0)` semantics and needs no coercion; output for a single batch element is bit-identical to before. (#690)
- **Every Flux generation variant can save, and dev-redux is saveable end-to-end**: conversion tooling driving the model classes got `AttributeError: '...' object has no attribute 'save_model'` from every Flux variant but `Flux1`/`Flux1Controlnet`, and `mflux-save --model dev-redux` was dispatched to plain `Flux1`, whose base weights can never come from the Redux adapter repo. All Flux variants now have `save_model`, and the redux save writes a self-contained checkpoint (base FLUX.1 components plus the siglip image encoder and redux embedder) that `mflux-generate-redux` reads back from disk instead of the hub. Along the way this repairs `mflux-generate-redux` on default arguments, which crashed loading base weights from the adapter repo (it has no vae/transformer/text encoders — the base now resolves from FLUX.1-dev unless a checkpoint path says otherwise). (#667, #680)

### ✨ Improvements

- **Training loss monitoring no longer taxes the run**: the loop computed a loss every step and discarded it, then paid up to 10 extra forward passes per plot tick for a smoothed re-read of training samples; with `plot_frequency: 1` that tripled wall time (5.4 vs 17.5 s/step on a Krea 2 Raw QLoRA, the 80 vs 232 hour run of #671). The already-computed value now feeds the loss curve every step for free, the batch metric became an optional second series at `plot_frequency` ticks, and `plot_frequency` is optional with a default of 20. Checkpoint loss statistics carry both series; a legacy checkpoint's single series loads as the batch metric it was. (#671, #672)

## [0.19.1] - 2026-08-20

### 🔒 Security

- **Locked dependencies updated beyond all reported vulnerable ranges**: raises the minimum supported versions of Pillow, Requests, PyTorch, Transformers, and urllib3, refreshes affected transitive dependencies, and removes the now-unused cryptography dependency from the lock. This addresses 35 Dependabot findings: 19 high, 12 medium, and 4 low. (#655)

### 🐛 Bug Fixes

- **`--instruction` exclusivity is declared, and visible to `mflux-capabilities`**: in-context-edit's prompt/instruction mutual exclusion was hand-checked after parsing, so the capabilities dump published `--instruction` as an independent flag and `--prompt-file --instruction` slipped through entirely. `--instruction` now joins the `--prompt`/`--prompt-file` argparse group; the "at least one" requirement stays post-parse. Also fixes `--width`'s help interpolating the HEIGHT default, and documents why fibo-edit's matte writes no sidecar. (#578, #669)
- **Baking a LoRA into a sub-8-bit quantized model silently dropped the adapter**: the fold requantized the merged weight at the base layer's own precision, and below 8 bits the quantization step is coarser than a typical LoRA delta (on Krea 2 at q4 the group step measures ~12x the delta rms), so `-q 4 --lora X` with baking on, which is the default, generated the base image while the logs reported success. Sub-8-bit layers now requantize at q8 when a delta is folded in, the same escape the fp8 path takes, with a console line reporting how many layers moved; the per-layer loader already reconstructs such mixed saves from stored shapes. Applies to runtime baking and to `mflux-save` of a quantized model with `--lora`. (#665, #668)
- **Foreign `--model` values are rejected on the boogu, z-image, flux2 and ideogram4 CLIs**: all of them silently ignored a builtin name from another family and ran their own model instead (`mflux-generate-boogu --model dev` ran Boogu; `mflux-generate-ideogram4 --model qwen-image` ran Ideogram). They now resolve through the same restriction as the krea2/lens/ernie/z-image-turbo CLIs: aliases of the command's own model (or, for flux2, any registry `flux2-` entry, and for z-image, its distilled sibling) are accepted, anything else errors during CLI startup before any weights move, and paths or repo ids keep loading through `model_path` as before. flux2-edit additionally enforces the same distilled-checkpoint `--guidance 1.0` rule as flux2-generate. (#578, #650)
- **Shell completions cover every installed command**: the generator's hand-maintained list had drifted six commands behind pyproject (`mflux-generate-boogu`, `mflux-generate-lens`, both `mflux-generate-ernie-image` commands, `mflux-generate-ideogram4`, `mflux-capabilities` never completed). Commands are now discovered from the installed console scripts and, where a CLI exposes `build_parser()`, completions are generated from the CLI's real parser instead of a hand-copied recipe of it. (#578, #651)
- **`mflux-capabilities` publishes wire types, not Python converter names**: options validated by a named converter leaked the function name as their type (`--vae-tile-size` claimed type `vae_tile_size`, `--mlx-cache-limit-gb` claimed `positive_float`, every Path option claimed `PosixPath`). Named converters now map to what they yield: `int`, `float`, `path`, and `int-or-scale` for values that accept a pixel count, a `2x` factor or `auto`. The dump's type field is pinned to that closed vocabulary by a test. (#578, #652)
- **The metadata sidecar of an in-context result can reproduce the run**: `get_right_half()` dropped `negative_prompt`, `image_paths` and the redux fields when cloning metadata onto the cropped half, and the `--save-full-image` composite was saved without a sidecar even under `--metadata`. `redux_image_paths` was also serialized as a Python repr (`"['a.png', 'b.png']"`) instead of a list, so `--config-from-metadata` could never read it back. The unreachable restore branches for `controlnet_save_canny` and `image_outpaint_padding`, keys no released writer has ever emitted, are removed along with a stray debug `print()` in the outpaint parse block. (#578, #649)
- **`mflux-generate-redux --config-from-metadata` can replay a sidecar**: #649 made `redux_image_paths` a JSON list, but the redux CLI still declared the flag argparse-required and `supports_metadata_config=False`, so a sidecar-only rerun exited 2 before restore ran. `-C` now restores the paths and strengths, `--redux-image-paths` is required after restore rather than during parsing, and a missing sidecar path errors at parse time naming the sidecar. Old repr-string sidecars are not treated as path lists. (#578, #663)

### 📝 Documentation

- **Lens and Boogu Image documented**: both models join the README's supported-models table, each with a per-model README covering usage and options. (#659)

### 🧰 DX & Maintenance

- **Machine-readable model registry extract**: a new `scripts/ci_extract_models.py` (with a `ci-extract` workflow and `just` recipe) dumps the supported-model registry as JSON, needed by CI to auto-build MFlux models for Hugging Face. (#658)
- **PyPI publish gated behind a protected `pypi` environment**: the release workflow now runs under the `pypi` deployment environment, so its reviewer-approval and branch-protection rules apply before anything reaches PyPI, and the trusted publisher is pinned to that environment name. A fast-fail step also rejects dispatches from any ref other than `main`. (#646)
- **Lens DiT loads through the shared `WeightDefinition` seam**: replaces its bespoke weight-loading path with the mechanism the other models use. (#654)
- **Flaky Gemma 2 causality test fixed**: the test could pick an out-of-vocab token and fail spuriously. (#653)

## [0.19.0] - 2026-08-18

### ⚠️ Behavior Changes

- **A metadata sidecar no longer stacks its LoRAs on top of the command line's**: `--config-from-metadata` restored `lora_paths`/`lora_scales` by concatenation, while every other key it restores defers to explicitly given arguments. The sidecar's adapters therefore always applied and no spelling could replace them — `--config-from-metadata prior.metadata.json --lora mine.safetensors 0.9` ran both. Since only one of the two lists usually came from the command line, the counts then disagreed and `resolve_scales` padded or truncated the difference behind a warning: `--lora-paths cli.safetensors` left the new adapter with no scale of its own, so the sidecar's went to the sidecar's adapter and the new one was padded to 1.0, and `--lora-scales 0.8` on a one-adapter sidecar generated at the sidecar's scale with the 0.8 discarded. `--lora`/`--lora-paths` now replaces the sidecar's adapters rather than adding to them, `--lora-paths` with no values re-runs without any, and `--lora-scales` on its own keeps the sidecar's adapters and applies the new strengths. Anything scripted around the stacking behaviour has to name every adapter it wants. Sidecars also record each adapter's resolved absolute path, so one carried between machines failed on a path that was nowhere on the command line — and failed even when the user named their own copy, since the dead path was merged in and resolved anyway; that now works, and when there is no replacement the error names the sidecar it came from and both ways out. (#577, #618)
- **`--guidance` and `--scheduler` now take effect on `mflux-upscale-controlnet`**: both were registered by the parser and then never passed to `generate_image`, so whatever you typed was replaced by `Flux1Controlnet`'s own defaults. They are forwarded now, which means an invocation that passed either flag produces a different image than it did before. Omitting `--guidance` still gives the 4.0 the upscaler has always run at, rather than `ui_defaults.GUIDANCE_SCALE` (3.5), so nothing changes for a command that never set it. (#577)
- **Abbreviated long options are rejected**: every mflux CLI now requires flags to be written in full, so `--prom` no longer stands in for `--prompt`. Option provision is detected by scanning `argv`, which cannot see abbreviations, and an abbreviation that is unambiguous today starts resolving elsewhere the moment a new flag is added. Anything scripted with a shortened flag needs the full spelling. (#499)

### ✨ Improvements

- **twine removed from mflux's dependencies**: it only existed to support the release script, but every `pip install mflux` pulled in twine plus ~24 transitive packages (`cryptography`, `keyring`, `rich`, `readme-renderer`, ...). The release script's token-mode upload now shells out to `uv publish` — `--check-url` replicates twine's `skip_existing`, the token travels via `UV_PUBLISH_TOKEN` so it never appears in logged commands, and the transient-retry semantics are unchanged — while `twine check dist/*` survives via `uv tool run twine` in an ephemeral environment. CI's trusted-publishing path is untouched (`pypa/gh-action-pypi-publish` still uploads). Also fixes the script invoking uv as `python -m uv`, which requires the uv PyPI package that the release environment does not install; it now calls the `uv` binary from PATH. Adds a `just release` recipe that triggers the release workflow via `gh`. (#644)
- **Type checking migrated from mypy to astral's ty** (pinned `ty==0.0.72` in dev deps, pre-commit and CI): full-project checks in ~0.4s vs mypy's 2–9s. Pre-existing violations are baselined as ignored rules in `[tool.ty.rules]` (with counts) to be re-enabled and fixed piecemeal; new code must pass all non-baselined rules. Adds `just typecheck`. (#597)
- **Quieter LoRA loading**: stacking a second adapter printed a `🔀` line for every layer it fused — hundreds of lines between the file's own `🔧` line and its `✅ Applied to N layers` summary, which already says how many landed. Those three per-layer messages are now `logger.debug`, like the resolution rules' tracing. (#577)
- **Silently dropped CLI options now warn**: a command that accepts an option it cannot honour says so at parse time instead of ignoring it, for example `--guidance` on a guidance-distilled model. Adds `mflux-capabilities`, a machine-readable dump of every image-generating command with its options, defaults, types and the status of each option (honored, ignored, conditional or rejected), in JSON, YAML or Markdown. (#499)

### ✨ Improvements

- **`--vae-tiling` and `--vae-tile-size` flags**: Restore user-facing control over tiled VAE decoding, decoupled from `--low-ram` (previously the only way to enable it). `--vae-tiling` enables tiled decode with the default 512px tiles; `--vae-tile-size 256` shrinks the tiles to further reduce peak decode memory and implies `--vae-tiling`. Both compose with `--low-ram`, whose implicit tiling defaults they override. The original `--vae-tiling`/`--vae-tiling-split` flags were removed in the Z-Image refactor (#284); this restores the capability on top of the generalized `VAETiler`. (#311, #407)
### 🐛 Bug Fixes

- **Saved FLUX ControlNet weights were ignored on load**: `FluxInitializer.init_controlnet` always fetched the ControlNet component from the remote repo, never consulting the `model_path` it honors for every other component. Reloading a checkpoint written by `mflux-save` silently swapped its ControlNet weights for the hub's — a `-q` save came back with that component unquantized, a fine-tune was discarded, and an otherwise complete local checkpoint still wanted the network. Only reachable since #607 made `mflux-save` actually write the ControlNet component. A `model_path` holding `transformer_controlnet/*.safetensors` now loads it from disk, the branch `ZImageInitializer` already had; a builtin name or a directory without that component keeps the hub path. (#610, #642)
- **A repo id shared by two registry entries resolved to the ControlNet variant**: several roots share a `model_name` with a ControlNet derivative, and the resolver returned the first match in `priority` order — which put the Z-Image ControlNet ahead of plain turbo, so `--model Tongyi-MAI/Z-Image-Turbo` built a ControlNet config in every resolver caller (the generate CLIs and `mflux-save` dispatch alike). Ties on a bare repo id now prefer the base entry, on the rule that a derived variant is always addressed by its own key or alias; keys and aliases resolve exactly as before. (#609, #641)
- **Failed LoRAs were reported as applied**: every step between resolving an adapter and folding it into the weights had a path that printed a warning and carried on, and the run then announced `✅ All LoRA weights applied successfully`. An adapter that could not be resolved was dropped from the list — generating from the untouched base model, and, with several adapters, shifting the scales onto the wrong ones as `resolve_scales` padded the shortened list back to length. A file that was missing or unreadable at load time printed `❌` and returned. A target layer the model does not have, or one that turned out not to be linear, printed `❌` per layer and left the adapter applied to only part of the model. And baking caught every exception — including a base-weight/delta shape mismatch — printed `⚠️` and returned the layer **unbaked**, so `mflux-save` wrote a checkpoint advertised as merged that held the original base weights, and ordinary generation (which bakes by default) quietly ran without the adapter. All of these now raise. Bake errors name the layer they failed at (`blocks.0.attn.wq`), and `mflux-save` bakes every component before it writes a single file, so a mismatch aborts instead of leaving half a checkpoint on disk. Anything scripted around a mistyped or missing adapter path now exits instead of generating an image without it. (#577)
- **boogu and FIBO accepted `--lora` and then discarded it**: `mflux-generate-boogu`, `mflux-generate-fibo` and `mflux-generate-fibo-edit` all declared the LoRA flags, so the parser resolved every adapter — downloading it from HuggingFace when the name was a repo id — and the initializers then dropped it on the floor, while `mflux-capabilities` reported `"lora": true` and nothing warned at any point. mflux implements no LoRA mapping for either architecture, so the flags are gone rather than ignored: the three CLIs no longer take them, matching `mflux-generate-lens`, and `--help` and the capabilities dump now agree with the code. `BooguInitializer`/`FIBOInitializer` and the `BooguImage`/`FIBO`/`FIBOEdit` constructors lost their unused `lora_paths`/`lora_scales`/`bake_lora` parameters so the Python API stops accepting an adapter it ignores, and `mflux-save --model boogu --lora ...`, which filtered the same kwargs away by signature and wrote an unmodified checkpoint, now exits with a message. (#577, #615)
- **`--model` silently ignored on 4 CLIs**: `mflux-generate-krea2`, `mflux-generate-z-image-turbo`, `mflux-generate-ernie-image` and `mflux-generate-ernie-image-turbo` accepted `--model` and then constructed a hard-coded config, so `--model dev` silently ran Krea-2-Turbo (etc.) while `mflux-capabilities` reported the option as honored. Each now validates `--model` through a shared `ConfigResolution.resolve_restricted` (the lens pattern): a builtin registry name must be an alias of the CLI's own model, anything else errors. Local checkpoint paths and HuggingFace repo ids — which `parse_args` routes through `model_path` — are untouched: they keep the CLI's own config and load weights from the path, as they always have. Validation compares registry entries by identity rather than `model_name`, so `--model z-image-controlnet` — whose entry shares the `Tongyi-MAI/Z-Image-Turbo` repo id — is correctly rejected by the plain turbo CLI; `lens_generate`'s inline check, which compared `model_name`, now uses the same helper. (#577, #614)
- **`mflux-completions` crashed on every invocation**: the `mflux-upscale-controlnet` parser added its LoRA arguments twice, so every build path died with `argparse.ArgumentError: conflicting option string: --lora-style` (exit 1) before writing a single completion. A regression test now builds the parser for every command mflux-completions knows about. (#577, #613)
- **Qwen Image Edit conditioning resolution**: Encode the transformer's image-conditioning latents at the edit target resolution, not the vision-language conditioning resolution (≈384px by area), preventing patchy/tiled artifacts in edit outputs. This changes edit output at every quantization level, not only `-q 4`. (#420)
- **Qwen Image Edit default dimensions**: Preserve the first input image dimensions by default; explicit `--width`/`--height` values or scale factors such as `2x` still opt into resizing. (#420)
- **Qwen Image Edit CLI scheduler**: Forward `--scheduler` to the Qwen edit pipeline (previously ignored). (#420)
- **Inferred model configs lost their settings**: `ConfigResolution` rebuilt inferred configs field by field, so anything resolved from a local path or variant name (`/models/qwen-image-edit-q4`) silently fell back to generic defaults. It now carries every field and rewrites only identity. This restores the Qwen edit sigma schedule (`0.9`/`8192`/`0.02` rather than `1.15`/`4096`/`None`), ERNIE's LoRA training guidance, and `supports_kv_cache` on `flux2-klein-9b-kv`. (#420)
- **Default `--steps` ignored the model on 10 CLIs**: every image CLI that picks its model in `main()` rather than at parse time — boogu, krea2, lens, z-image, z-image-turbo, flux2, flux2-edit, qwen, qwen-edit, ideogram4 — resolved `--steps` against a `None` model and silently inherited FLUX.1-dev's 25. That is 6x the work on a 4-step distillation (boogu, lens, klein) and half the work on a 50-step base model (z-image). The step table is now keyed by canonical model rather than by alias and resolved through `AVAILABLE_MODELS`, so `--model klein-4b` and `--model flux2-klein-4b` no longer disagree, and a CLI declares its own model to the parser via `add_model_arguments(default_model=...)`. Krea 2's `DEFAULT_STEPS = 8` fallback, unreachable because the parser always supplied 25, is gone. Explicit `--steps` is unaffected. (#580)
- **`--model <name>` failed for most built-in models**: the names the CLI accepted were a hand-maintained list (`ui_defaults.MODEL_CHOICES`) that had drifted from `AVAILABLE_MODELS`, and anything missing from it was treated as a local checkpoint directory — `mflux-generate-lens --model lens-turbo` died with `Model not found: 'lens-turbo'` on the canonical name of the only model that CLI runs. 16 registry keys and roughly 40 aliases were affected, among them `krea-2-raw`, `qwen-image`, and every `klein-*`, `boogu-*`, `zimage*` and `fiboedit*` spelling. The accepted names are now derived from the registry, so a new model or alias needs no second edit. `--base-model` likewise dropped its stale argparse `choices=` list, which rejected valid names such as `qwen-image`; it is now validated against `ConfigResolution`'s own list of root models, and the check also covers a `base_model` restored from a metadata sidecar. (#577)
- **`mflux-save` wrote checkpoints under the wrong architecture**: the model class was chosen by a substring chain over the raw `--model` string ending in `else: Flux1`, so every name it had not been taught was saved as a FLUX.1 model — `lens`, `lens-turbo`, `klein-4b`/`9b`/`9b-kv` and `seedvr2*` all were — while `fibo-edit*` was saved by the txt2img FIBO class and `z-image-controlnet*` by plain `ZImage`. Nothing raised; the mismatch surfaced later as an unloadable checkpoint. Dispatch now goes through the registry (`ConfigResolution.resolve_key`), so canonical keys, aliases, HuggingFace repo ids and `--base-model` spellings all land on the class that owns those weights, an unrecognisable name exits with a message instead of defaulting to Flux1, and the two models with no save path (Lens, SeedVR2) say so. Every rejection also lists the models `mflux-save` can write, which `--help` does not show — it lists models that cannot be saved. A drift test fails if a model is added to `AVAILABLE_MODELS` without a save class. Two consequences worth noting: `--model dev-controlnet-canny`/`dev-controlnet-upscaler`/`schnell-controlnet-canny` now save through `Flux1Controlnet` (ControlNet weights included) rather than as a bare Flux.1, and `mflux-save --model boogu` no longer dies with a `TypeError` on `bake_lora`, which `BooguImage` does not accept. (#607)
- **ERNIE multi-frame latents**: `pack_latents` indexed the VAE's temporal axis away, so anything but a single frame was silently discarded; it now fails loudly. (#577)
- **`ernie-image-turbo` guidance contract**: the model declared `supports_guidance: true` while its CLI exits on any guidance other than 1.0, contradicting itself in the `mflux-capabilities` dump. (#578)
- **Partial Krea 2 Turbo downloads passed as complete**: a cached snapshot was checked against its download patterns only when the model needed no subdirectory. Krea 2 Turbo keeps its transformer at the repo root as `turbo.safetensors`, so a download that fetched `vae/` and `text_encoder/` and then stopped reported itself complete. Hugging Face's repair download never ran, and the load died later with `Missing specified weight files in <snapshot>: ['turbo.safetensors']`, which reads like a corrupt cache rather than an interrupted transfer. Root-level `*.safetensors` patterns are now checked in the subdirectory branch too. (#577, #593)
- **Saved checkpoints loaded whatever was in the directory**: `mflux-save` writes a `model.safetensors.index.json` naming every shard, and the reload globbed the directory instead of reading it. Saving `-q 4` over an existing `-q 8` checkpoint in the same folder left the q8 tail behind, and those stale tensors overwrote the ones just written while the metadata still reported 4. A checkpoint missing a shard the index names loaded the rest and came up short. The index now decides which shards get read, and a named shard that is not on disk raises with the filenames. Checkpoints with no index, or with a damaged one, keep loading from the directory. (#577)
- **Metadata sidecars did not restore dimensions, the negative prompt or init images**: every run embeds `height`, `width`, `negative_prompt` and `image_paths`, but `--config-from-metadata` read none of them back, so a sidecar-only rerun of a 1536x768 generation came out at the 1024x1024 default with the negative prompt dropped — a different image on any CFG model, not just a differently sized one. The edit CLIs could not rerun a sidecar at all: `--image-paths` was `required=True`, which argparse enforces before the restore block runs, so `mflux-generate-qwen-edit --config-from-metadata prior.metadata.json` exited 2 demanding images the sidecar was carrying. All four keys are restored now, each deferring to the command line option by option, and `--image-paths` is required after the restore rather than during parsing. A sidecar's init image that does not exist on this machine now errors at parse time naming the sidecar, instead of loading the whole model first and dying on a bare `FileNotFoundError`. (#577, #634)
- **The upscale commands were missing from `mflux-capabilities`**: `COMMAND_PREFIXES` covered `mflux-generate*` and `mflux-concept*` only, so neither `mflux-upscale-controlnet` nor `mflux-upscale-seedvr2` appeared in the dump — which is how the `--metadata` bug above went unnoticed by the contract checks that exist to catch exactly that. Both are now published with full coverage, which meant adopting the `build_parser()` convention and declaring what their hardcoded configs cannot honour: `--base-model` is ignored by both, and on `mflux-upscale-controlnet` `--negative-prompt` is ignored while `--model` is conditional (a path or repo id loads weights, a built-in model name does nothing, since the config is always dev-controlnet-upscaler). Both commands now warn at parse time when an ignored option is passed. (#577)
- **`mflux-upscale-seedvr2 --metadata` never wrote the sidecar**: the CLI accepted the flag and then saved without forwarding it, so the upscale ran to completion and no JSON appeared. Its sibling `mflux-upscale-controlnet` passes `export_json_metadata` on the equivalent line. (#577)
- **Ideogram 4 quantization**: `mflux-save -q` now actually quantizes Ideogram 4. Every weight-bearing linear in the model is an `Fp8Linear`, which defined no `to_quantized` and whose components were marked `skip_quantization`, so `-q 4` wrote an FP8 checkpoint stamped `quantization_level: 4` and only the VAE was touched. Adds `Fp8Linear.to_quantized`, drops the skip flags, derives bits and group size from the stored shapes when rebuilding a saved checkpoint instead of assuming q8/group-64, and rebuilds quantized embeddings as embeddings rather than linears. `-q 8` is 26 GB and `-q 4` is 14 GB, both visually indistinguishable from FP8. (#559)

### 📝 Documentation

- **Ideogram 4 gated weights**: Document the `HF_TOKEN` / `hf auth login` step alongside the existing note that access must be approved on the model card — authenticating is the half people miss, and without it an approved account still fails with a bare `401`/`403`. Adds a quantization section covering `mflux-save -q` and loading a saved checkpoint with `--model-path`. (#559)

### 🧰 DX & Maintenance

- **Dev workflow: Makefile replaced by justfile**: `make <target>` is gone; use `just <target>` instead (`just install`, `just lint`, `just format`, `just check`, `just test-fast`, …). Running bare `just` (or `just --list`) shows all recipes with descriptions. Semantics follow the post-#576 Makefile: `uv sync` installs and ruff is pinned from the `pyproject.toml` dev dependency. The CI lint job also lints the justfile itself via `just --fmt --check`, using the pinned `just-setup` composite action. Requires [`just`](https://github.com/casey/just) ≥ 1.50 locally. (#590)

## [0.18.1] - 2026-08-07

### 🎨 New Model Support

- **Krea 2**: Add text-to-image support for `krea/Krea-2-Turbo` — a single-stream MMDiT built on the Qwen-Image stack (Qwen-Image VAE + a 12-layer Qwen3-VL-4B text-encoder tap). Includes the `mflux-generate-krea2` CLI (live progress, `--metadata`, stepwise output), `er_sde` and Euler samplers, and `mflux-save` quantization caching.
- **Krea 2 Raw + LoRA training**: Add `krea/Krea-2-Raw` as a trainable base and `mflux-train` LoRA training for Krea 2 (flow-matching velocity, QLoRA over the quantized base, gradient checkpointing across the 28 blocks). The transformer also loads from the diffusers `transformer/` shard layout in addition to the native single-file checkpoint, and the official `krea/Krea-2-LoRA-*` adapters load as-is. Train on Raw, run the adapter on Turbo (Krea's recommended workflow). (#462)

### ✨ Improvements

- **Atomic `--lora` and `--image` flags**: Pair each path with its value on a single, repeatable flag — `--lora A.safetensors 0.7 --lora B.safetensors` (scale defaults to `1.0`) and `--image photo.jpg 0.6` (strength defaults to the model default). This removes the positional-alignment footgun of the parallel `--lora-paths`/`--lora-scales` and `--image-path`/`--image-strength` lists, which remain fully supported and are marked deprecated in `--help`. (#438)

### 🐛 Bug Fixes

- **Ideogram 4 stepwise output**: Fix `--stepwise-image-output-dir` for Ideogram 4 by routing unpacked latents to `vae.decode` instead of `decode_packed_latents` when channels are already VAE-ready. (#444)

### 📝 Documentation

- **Related projects**: Add [mflux-paint](https://github.com/Amo643/mflux-paint) to the Related projects list. (#471)

### 🧰 DX & Maintenance

- **PyPI trusted publishing**: Publish releases via GitHub Actions OIDC (`pypa/gh-action-pypi-publish`) instead of a long-lived `PYPI_API_TOKEN` secret.

### 👩‍💻 Contributors

- **@Amo643**
- **@anthonywu**
- **@filipstrand**
- **@plz12345**

---

### ✨ Features

- **LyCORIS LoKr adapters (FLUX.1 and FLUX.2)**: Load community LyCORIS LoKr safetensors through the existing `--lora-paths` / `lora_paths` API. Supports direct and factorized (`lokr_w1_a`/`lokr_w1_b`, `lokr_w2_a`/`lokr_w2_b`, optional `lokr_t2`) tensors, observed LyCORIS key layouts, alpha scaling for decomposed factors, optional `dora_scale`, multi-adapter fusion with classic LoRA, and baking into non-quantized base weights as well as quantized layers (via dequantization and re-quantization). Inference applies the Kronecker product without materializing full dense deltas for standard (non-DoRA) LoKr layers.

## [0.18.0] - 2026-06-07

### 🎨 New Model Support

- **ERNIE-Image & ERNIE-Image-Turbo**: Port Baidu's ERNIE-Image models with text-to-image CLI entrypoints, LoRA inference and training, and `mflux-save` support.
- **Ideogram 4 FP8**: Add Ideogram 4 FP8 text-to-image support with JSON caption handling, FP8 safetensors loading, and a dedicated CLI entrypoint.

### ✨ Improvements

- **FLUX.2 Klein 9B KV-cache**: Add KV-cache support for `flux2-klein-9b-kv` with roughly 2.4× speedup on multi-reference edit workloads.

### 🐛 Bug Fixes

- **FLUX.2 Klein Edit guidance**: Allow `--guidance > 1.0` for FLUX.2 Klein edits by checking the resolved FLUX.2 model config instead of requiring a base model name; defaults remain unchanged.
- **FLUX.2 Klein `mflux-generate`**: Fix `FileNotFoundError: text_encoder_2` by routing Klein models through `Flux2Klein` and skipping the unused T5/`text_encoder_2` weight path.
- **Memory management**: Evict the text encoder after encoding and clear the MLX cache between seeds on multi-seed runs to prevent OOM on large models such as FLUX.2 Klein 9B.

### 📝 Documentation

- **Related projects**: Add mlx-taef and mlx-teacache to the Related projects list.

### 👩‍💻 Contributors

- **@azrahello**
- **@c2p-cmd**
- **@IonDen**
- **@lpalbou**
- **@michaeltrefry**
- **@omercelik**
- **@plz12345**

---

## [0.17.5] - 2026-04-10

### 🐛 Bug Fixes

- **Qwen Image Edit `mflux-save`**: Route Qwen edit model names to `QwenImageEdit` and save through the same path as inference so VisionTransformer (`encoder.visual`) weights are written. Saving with `QwenImage` previously omitted those weights and led to random vision encoders after reload.
- **Battery saver callback**: Harden Apple Silicon battery detection when `system_profiler` is missing and resolve the helper script via absolute paths.

### 📝 Documentation

- **Related projects**: Clarify that MindCraft Studio is a macOS app built on mflux.

### 🧰 DX & Maintenance

- **Dependencies**: Relax the `protobuf` upper bound to allow current 7.x releases while keeping a safe ceiling below 8.0.

### 👩‍💻 Contributors

- **@anthonywu**
- **@f-gibellini**
- **@JiwaniZakir**

---

## [0.17.4] - 2026-03-28

### 🐛 Bug Fixes

- **Z-Image PEFT/ModelScope LoRA keys**: Extend the Z-Image LoRA mapping with `.default` tensor name variants so adapters in PEFT/ModelScope layouts (for example Tongyi-MAI exports) resolve and apply correctly instead of matching zero weights.

### 👩‍💻 Contributors

- **@filipstrand**

---

## [0.17.3] - 2026-03-27

### 🐛 Bug Fixes

- **FLUX.2 edit guidance metadata**: Preserve the requested guidance value for FLUX.2 Klein base image-edit runs so `mflux-info` and saved metadata report the actual guidance used instead of always showing `1.0`.

### 👩‍💻 Contributors

- **@filipstrand**

---

## [0.17.2] - 2026-03-23

### 🐛 Bug Fixes

- **Shared tokenizer cache resolution**: Fix Hugging Face tokenizer resolution when a repo is only partially cached locally, preserving offline-first behavior for valid cached layouts while retrying ambiguous cached primaries once before surfacing real load errors.

### 🧰 DX & Maintenance

- **Tokenizer resolution coverage**: Expand shared tokenizer-resolution regression tests to cover root-layout tokenizers, fallback edge cases, and refresh failure handling.

### 👩‍💻 Contributors

- **@filipstrand**

---

## [0.17.1] - 2026-03-22

### 🐛 Bug Fixes

- **Hugging Face tokenizer dependencies**: Declare `protobuf` so minimal installs (including `uv tool install mflux`) include packages Transformers may require when loading tokenizers, fixing failures such as `mflux-generate-fibo` when the tokenizer falls back off the fast path.

### 👩‍💻 Contributors

- **@filipstrand**

---

## [0.17.0] - 2026-03-20

### 🎨 New Model Support

- **FIBO Edit**: Add image-editing support for the FIBO model family.
- **FIBO Edit remove-background workflow**: Support the dedicated remove-background edit path for FIBO.

### ✨ Improvements

- **Training image scaling**: Scale training images by area rather than longest side for more consistent preprocessing.
- **MLX 0.31.x**: Allow MLX 0.31.x in dependency ranges.
- **FLUX.2 LoRA mapping**: Expand LoRA key mapping coverage for FLUX.2.

### 🐛 Bug Fixes

- **Training optimizer state**: Evaluate optimizer state after each training step as intended.
- **Local tokenizer loading**: Fix loading tokenizers from local paths.
- **Dynamic-resolution image edit**: Restore correct behavior for image edit when using dynamic resolution.

### 👩‍💻 Contributors

- **@filipstrand**
- **@icelaglace**
- **@TheOrsa**
- **@waldheinz**

---

## [0.16.9] - 2026-03-07

### ✨ Improvements

- **Broader LoRA compatibility for FLUX.2 and Z-Image**: Expand LoRA mapping coverage so more adapter key layouts resolve cleanly for FLUX.2 and Z-Image models.

### 👩‍💻 Contributors

- **@filipstrand**

---

## [0.16.8] - 2026-03-06

### ✨ Improvements

- **Local-model LoRA training**: Allow LoRA training to work when the base model is supplied from a local path, including the FLUX.2 and Z-Image training adapters.

### 📝 Documentation

- **Distilled-model step defaults**: Clarify CLI guidance so examples prefer model default inference steps unless the user intentionally overrides them.

### 👩‍💻 Contributors

- **@waldheinz**

---

## [0.16.7] - 2026-03-02

### 🎨 New Model Support

- **FIBO-Lite support**: Add support for the FIBO-Lite model variant.

### 🐛 Bug Fixes

- **FLUX.2 edit downsampling extents**: Fix downsampling in FLUX.2 edit paths so image extents are preserved.

### 👩‍💻 Contributors

- **@filipstrand**

---

## [0.16.6] - 2026-02-20

### ✨ Improvements

- **SeedVR2 7B support**: Add support for the SeedVR2 7B upscaler variant.
- **Qwen-Image parity with diffusers**: Align Qwen-Image behavior more closely with the diffusers reference implementation.
- **FIBO scheduler default**: Default FIBO `generate_image` to `flow_match_euler_discrete`.

### 🧰 DX & Maintenance

- **Repo tooling cleanup**: Remove unused Cursor command wrappers from the repository.
- **SeedVR2 7B test coverage**: Add image test support for the new SeedVR2 7B path.

### 👩‍💻 Contributors

- **@ciaranbor**
- **@icelaglace**
- **@filipstrand**

---

## [0.16.5] - 2026-02-17

### ✨ Improvements

- **FLUX.2 Klein img2img CLI parity**: Add `--image-path` and `--image-strength` to `mflux-generate-flux2`, enabling init-image driven generation with the same CLI pattern used in other generators.
- **MLX cache control**: Add `--mlx-cache-limit-gb` to cap MLX cache usage without requiring full `--low-ram` mode.

### 📝 Documentation

- **Common CLI docs**: Document `--mlx-cache-limit-gb` behavior and usage in the shared model README.

### 👩‍💻 Contributors

- **@terribilissimo**
- **@icelaglace**

---

## [0.16.4] - 2026-02-15

### 🐛 Bug Fixes

- **Training preview stability**: Always offload optimizer state during preview generation to avoid memory pressure and improve preview reliability.
- **Apple Silicon compile guard**: Narrow the M1/M2 compile fallback so it excludes Max and Ultra variants, preserving expected optimized behavior on those chips.

---

## [0.16.3] - 2026-02-14

### 🐛 Bug Fixes

- **Z-Image training preview guidance**: Fix Z-Image (non-turbo) training previews so they use the configured guidance value instead of defaulting to 0.0, ensuring preview quality matches actual CFG behavior.
- **FLUX.2 training preview guidance**: Fix FLUX.2 training previews (txt2img and edit) so they use the configured guidance value instead of forcing 1.0.

---

## [0.16.2] - 2026-02-12

### 🐛 Bug Fixes

- **Edit training preview fallback**: Fix edit auto-discovery runs (`*_in/*_out`) with monitoring enabled so fallback preview prompts use an available input image instead of requiring explicit `data/preview.*` files.

### 📝 Documentation

- **FLUX.2 training guide**: Expand the FLUX.2 LoRA training example documentation with richer guidance and examples.

---

## [0.16.1] - 2026-02-11

### 🐛 Performance regression fixes

- **M1/M2 inference performance fallback**: Disable model-level `mx.compile` prediction wrappers for Z-Image and FLUX.2 on Apple M1/M2 to avoid observed 0.16 regressions on older Apple Silicon while preserving compiled paths on newer chips.

---

## [0.16.0] - 2026-02-11

### ✨ Improvements

- **Completely rewritten training system**: Rebuild LoRA training end-to-end, replacing the DreamBooth-specific implementation with a new common training stack (dataset, state, optimizer, runner, and statistics) shared across model families.
- **New base-model support for training and inference**: Add support for `flux2-klein-base-4b`, `flux2-klein-base-9b`, and `z-image` (in addition to `z-image-turbo`) with dedicated FLUX.2 and Z-Image training adapters.
- **Performance tuning**: Improve core scheduler/model execution paths used by FLUX.2 and Z-Image.

### 🐛 Bug Fixes

- **FLUX.2 Klein 9B text encoder overrides**: Fix override resolution/application in the FLUX.2 initializer/config flow.

### 🧰 DX & Maintenance

- **FLUX.1 legacy cleanup**: Remove legacy FLUX.1 image-generation tests/resources and retire unused helper tools.
- **Dependency alignment**: Update install guidance for stable `transformers` 5.0 and refresh lockfile/dependency metadata.

### 📝 Documentation

- **Training docs refresh**: Expand and update training docs/README sections for common training, FLUX.2, and Z-Image.
- **Install troubleshooting**: Add troubleshooting guidance for `hf_transfer` installation issues.

### 👩‍💻 Contributors

- **Filip Strand (@filipstrand)**
- **Xin (@q3g)**

---

## [0.15.5] - 2026-01-26

### ✨ Improvements

- **SeedVR2 directory input**: Allow passing a folder to `--image-path` to upscale all images inside.

### 🧰 DX & Maintenance

- **Model porting guidance**: Require model README entries in the porting workflow.

### 📝 Documentation

- **SeedVR2 usage**: Document directory upscaling with CLI and Python API examples.
- **CLI docs**: Add Python API sections and improve Z-Image Turbo entry-point links.

---

## [0.15.4] - 2026-01-20

### ✨ Improvements

- **Flux2 LoRA aliasing**: Add key aliases for `base_model` prefixes to improve LoRA resolution across configs.

### 📝 Documentation

- **Agent guidance**: Clarify skill references for Cursor agents.

---

## [0.15.3] - 2026-01-19

### 🐛 Bug Fixes

- **Flux2 Klein local path**: Fix errors when using a local FLUX.2-klein-9B path in `mflux-save` and `mflux-generate-flux2`.

---

## [0.15.2] - 2026-01-19

### 🐛 Bug Fixes

- **Flux2 edit (low-ram)**: Normalize tiled VAE latents to 4D before patchifying to avoid shape errors.

---

## [0.15.1] - 2026-01-18

### 🐛 Bug Fixes

- **PyPI metadata**: Removed invalid architecture classifier that blocked uploads (`Architecture :: AArch64`).

---

## [0.15.0] - 2026-01-18

### 🎨 New Model Support

- **Flux2 Klein (4B/9B)**: Full MLX port of Flux2 Klein (including multi-image editing support).
- **New command**: `mflux-generate-flux2` for Flux2 Klein image generation.
- **New command**: `mflux-generate-flux2-edit` for Flux2 Klein image editing.

### 🔧 Improvements

- **Qwen3-VL shared module**: Extracted `qwen3_vl` into `models/common_models/` for reuse across model families (Flux2 and Fibo etc).
- **Experimental CUDA support**: Added initial CUDA backend support as an experimental feature.
- **Test Infrastructure**: Image tests are pinned to MLX v0.30.3.

### 📝 Documentation

- **README reorganization**: Reorganized the main README for better structure and readability.

---

## [0.14.2] - 2026-01-13

### 📊 Improved Metadata Handling

- **Enhanced IPTC & XMP Support**: Significant improvements to metadata reading and writing, ensuring better compatibility with professional image editing tools.
- **Robust Metadata Extraction**: Refined logic for extracting generation parameters from previously generated images.
- **New Metadata Tests**: Added comprehensive test suite for IPTC metadata building and original image info utilities.

### 🤖 DX & Maintenance

- **Cursor AI Workflows**: Introduced standardized Cursor commands and agent rules in `.cursor/` for improved development consistency and automation.
- **SeedVR2 & ControlNet Tweaks**: Minor refinements to SeedVR2 and ControlNet model implementations.
- **Documentation Updates**: Updated README and added AGENTS.md for better contributor onboarding.

---

## [0.14.1] - 2026-01-01

### 🔧 SeedVR2 Improvements

- **Enhanced Color Correction**: Implemented precise LAB histogram matching with wavelet reconstruction for superior color consistency between input and upscaled images.
- **Configurable Softness**: Added a new `--softness` parameter (0.0 to 1.0) to control input pre-downsampling, allowing for smoother upscaling results when desired.
- **RoPE Alignment**: Fixed RoPE dimension mismatch (increased to 128) to perfectly match the reference 3B transformer architecture.

### 🤖 DX & Maintenance

- **Updated `.cursorrules`**: Added standard procedure for test output preservation and release management.
- **Updated Test Infrastructure**: Updated SeedVR2 reference images and fixed dimension-related test failures.

---

## [0.14.0] - 2025-12-31

### 🎨 New Model Support

- **SeedVR2 Diffusion Upscaler**: Added support for [SeedVR2](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler), a powerful diffusion-based image upscaler.
- **New command**: `mflux-upscale-seedvr2` for high-quality image upscaling.
- **Tiling support**: Tiling is enabled by default for SeedVR2 to support high-resolution upscaling on standard memory configurations.

### 🔧 Improvements

- **Global VAE Tiling Support**: Introduced a unified VAE tiling system (`VAETiler`) that supports both tiled encoding and decoding.
- **Low-RAM Mode Enhancements**: Enabling `--low-ram` now automatically activates VAE tiling across all model families (Flux, Qwen, FIBO, Z-Image), significantly reducing memory pressure for high-resolution generation on Apple Silicon.
- **Robust Offline Cache Handling**: Improved logic for detecting complete cached models on HuggingFace Hub, handling symlinks and missing files more reliably to prevent runtime errors during offline use.
- **Selective Weight Loading**: Support for loading specific weight files, enabling more flexible model configurations and better resource sharing between related models.
- **CLI UX Improvements**:
  - Multi-image generation (multiple seeds or input images) now automatically appends suffixes (`_seed_{seed}` or `_{image_name}`) to output filenames to prevent accidental overwrites.
  - Better model configuration resolution with a priority-based system for resolving ambiguous model names.
- **Enhanced Shell Completions**: Significant updates to shell completion generation to support new commands and properly handle positional arguments and subparsers.
- **Qwen Test Hardening**: Updated Qwen image generation and edit tests to use 8-bit quantization for more robust and faster testing.
- **Test Infrastructure**: Added automatic MLX version pinning (v0.29.2) in `make test-fast` to ensure consistent test environments across different development setups.

### 📝 Documentation

- Added information about pre-quantized models available on HuggingFace for easier access.

---

## [0.13.3] - 2025-12-06

### 🐛 Bug Fixes

- **LoRA save bloat prevention**: Bake and strip LoRA wrappers before sharding to avoid exploding shard counts/sizes when saving quantized models with multiple/mismatched LoRAs (see [issue #217 comment](https://github.com/filipstrand/mflux/issues/217#issuecomment-3615321206)).
- **Regression test hardening**: LoRA model-saving tests now include size guardrails (5% tolerance) while using the bundled local LoRA fixtures to catch shard bloat regressions early.

---

## [0.13.2] - 2025-12-05

### ✨ Improvements

- **Better error messages for multi-file LoRA repos**: When a HuggingFace LoRA repo contains multiple `.safetensors` files, the error message now displays copy-paste ready options instead of a raw list
- **Z-Image LoRA format support**: Added support for Kohya and ComfyUI LoRA naming conventions, enabling compatibility with more community LoRAs.

---

## [0.13.1] - 2025-12-03

### 🐛 Bug Fixes

- **FIBO VLM chat template not loaded**: Fixed issue where the FIBO VLM tokenizer's chat template was not being loaded with `transformers` v5, causing `apply_chat_template()` to fail. The tokenizer loader now properly extracts and sets the chat template from the tokenizer config.

---

## [0.13.0] - 2025-12-03

# MFLUX v.0.13.0 Release Notes

### 🎨 New Model Support

- **Z-Image Turbo Support**: Added support for [Z-Image Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo), a fast distilled Z-Image variant optimized for speed
- **New command**: `mflux-generate-z-image-turbo` for rapid image generation (with LoRA support, img2img, and quantization)

### ✨ New Features

- **FIBO VLM Quantization Support**: The FIBO VLM commands (`mflux-fibo-inspire`, `mflux-fibo-refine`) now support quantization via the `-q` flag (3, 4, 5, 6, or 8-bit)

- **Unified `--model` argument**: The `--model` flag now accepts local paths, HuggingFace repos, or predefined model names
  - Local paths: `--model /Users/me/models/fibo-4bit` or `--model ~/my-model`
  - HuggingFace repos: `--model briaai/Fibo-mlx-4bit`
  - Predefined names: `--model dev`, `--model schnell`, `--model fibo`
  - This mirrors how LoRA paths work for a consistent UX

- **Scale Factor Dimensions for Img2Img**: Generalized the scale factor feature (e.g., `2x`, `0.5x`, `auto`) from upscaling to all img2img commands
  - Specify output dimensions relative to input image: `--width 2x --height 2x`
  - Use `auto` to match input image dimensions: `--width auto --height auto`
  - Mix scale factors with absolute values: `--width 2x --height 512`
  - Supported in: `mflux-generate`, `mflux-generate-z-image-turbo`, `mflux-generate-fibo`, `mflux-generate-kontext`, `mflux-generate-qwen`
- **DimensionResolver utility**: New `DimensionResolver.resolve()` for consistent dimension handling across commands

### 🔧 Architecture Improvements

- **Unified Resolution System**: New `resolution/` module for consistent parameter resolution across all models
  - `PathResolution`: Resolves model paths from local paths, HuggingFace repos, or predefined names
  - `LoRAResolution`: Handles LoRA path resolution from all supported formats
  - `ConfigResolution`: Centralizes configuration resolution logic  
  - `QuantizationResolution`: Determines quantization from saved models or CLI args
- **Unified Weight Loading System**: Complete rewrite of weight handling with declarative mappings
  - New `WeightLoader` with single `load(model_path)` interface
  - `WeightDefinition` classes define model structure per model family
  - `WeightMapping` declarative mappings replace imperative weight handlers
  - Removed all per-model `weight_handler_*.py` files in favor of unified system
- **Unified Tokenizer System**: New common tokenizer module
  - `TokenizerLoader.load_all()` with unified `model_path` interface
  - Removed model-specific tokenizer handlers (`clip_tokenizer.py`, `t5_tokenizer.py`, etc.)
- **Unified LoRA API**: Simplified LoRA loading to a single `lora_paths` parameter
  - All LoRA formats now resolved through `LoRALibrary.resolve_paths()`:
    - Local paths: `/path/to/lora.safetensors`
    - Registry names: `my-lora` (from `LORA_LIBRARY_PATH`)
    - HuggingFace repos: `author/model`
    - **New**: HuggingFace collections: `repo_id:filename.safetensors`
  - Simplified model initialization: just pass `lora_paths` and everything resolves automatically
- **Unified Latent Creator Interface**: Standardized `unpack_latents(latents, height, width)` signature across all model families
  - `FluxLatentCreator`, `ZImageLatentCreator`, `FiboLatentCreator`, and `QwenLatentCreator` now share the same interface
  - Moved `FIBO._unpack_latents` to `FiboLatentCreator.unpack_latents` for consistency
- **StepwiseHandler Refactor**: Fixed `StepwiseHandler` to work with all model types by accepting a `latent_creator` parameter
  - Previously hardcoded to `FluxLatentCreator`, now model-agnostic
  - Each command passes its appropriate latent creator to `CallbackManager.register_callbacks()`
- **CLI Reorganization**: Moved CLI entry points to model-specific directories (e.g., `mflux/models/flux/cli/`)

### 🔄 Breaking Changes

- **Simplified `generate_image()` API** (programmatic users only):
  - Removed `Config` class - parameters are now passed directly to `generate_image()`
  - Removed `RuntimeConfig` class - internal complexity eliminated
  - Added `Flux1` export to main `mflux` module for cleaner imports
- **LoRA API simplified** (programmatic users only):
  - Removed `lora_names` and `lora_repo_id` parameters from all model classes (`Flux1`, `QwenImage`, `QwenImageEdit`, etc.)
  - Removed `--lora-name` and `--lora-repo-id` CLI arguments
  - Removed `LoRAHuggingFaceDownloader` class

### 🔄 Breaking Changes (CLI)

- **`--path` flag removed**: The deprecated `--path` flag for loading models has been removed. Use `--model` instead for local paths, HuggingFace repos, or predefined model names.

### 📦 Dependency Updates

- **Updated `huggingface-hub`** from `>=0.24.5,<1.0` to `>=1.1.6,<2.0`
  - v1.1.6 includes fix for incomplete file listing in `snapshot_download` which could cause cache corruption
  - Removed explicit `accelerate` and `filelock` dependencies (pulled in as transitive dependencies)
- **Updated `transformers`** from `>=4.57,<5.0` to `>=5.0.0rc0,<6.0`
  - Required for `huggingface-hub` 1.x compatibility
  - Added workaround for `Qwen2Tokenizer` bug in transformers 5.0.0rc0 where vocab/merges files are not loaded correctly via `from_pretrained()`

### 🐛 Bug Fixes

- **Qwen empty negative prompt crash**: Fixed crash when running Qwen models without a `--negative-prompt` argument. Empty prompts now use a space as fallback to ensure valid tokenization.

- **`--model` flag not working**: Fixed bug where the `--model` argument wasn't being used for loading models from HuggingFace or local paths. All CLI commands now correctly use `--model` for model path resolution.
- **Model Saving Index File**: Fixed issue where locally saved models (via `mflux-save`) would fail to load when uploaded to HuggingFace, due to missing `model.safetensors.index.json`. The model saver now generates this index file alongside the safetensor shards, ensuring compatibility with both mflux and standard HuggingFace loading paths. (see [#285](https://github.com/filipstrand/mflux/issues/285))

### 🧪 Test Infrastructure

- **Test markers**: Added `fast` and `slow` pytest markers to categorize tests
  - Fast tests: Unit tests that don't generate images (parsers, schedulers, resolution, utilities)
  - Slow tests: Integration tests that generate actual images and compare to references
- **New Makefile targets**:
  - `make test-fast` - Run fast tests only (quick feedback during development)
  - `make test-slow` - Run slow tests only (image generation tests)
  - `make test` - Run all tests (unchanged)
- Run specific test categories: `pytest -m fast` or `pytest -m slow`
- **GitHub Actions CI**: Fast tests now run automatically on PRs and pushes to main

### 🔧 Internal Changes

- Simplified `WeightLoader.load()` to take a single `model_path` parameter instead of separate `repo_id` and `local_path`
- Simplified `TokenizerLoader.load_all()` with the same unified `model_path` interface
- Renamed `local_path` parameter to `model_path` in all model constructors for clarity
- Removed `quantization_util.py` - quantization now handled through `QuantizationResolution`
- Removed `lora_huggingface_downloader.py` - downloading integrated into `LoRAResolution`
- Added comprehensive test coverage for resolution modules

### 👩‍💻 Contributors

- **Filip Strand (@filipstrand)**: Z-Image Turbo support, architecture improvements, core development

---

## [0.12.1] - 2025-11-27

### 🐛 Bug Fixes

- **FIBO VLM Tokenizer Download**: Fixed an issue where the FIBO VLM tokenizer files would not download automatically when the model weights were cached but tokenizer files were missing. The initializer now properly checks for tokenizer file existence and downloads them if needed.

---

## [0.12.0] - 2025-11-27

# MFLUX v.0.12.0 Release Notes

### 🎨 New Model Support

- **Bria FIBO Support**: Added support for [FIBO](https://huggingface.co/briaai/FIBO), the first open-source JSON-native text-to-image model from [Bria.ai](https://bria.ai)
- **Three operation modes**: Generate (text-to-image with VLM expansion), Refine (structured prompt editing), and Inspire (image-to-prompt extraction)
- **New commands**:
  - `mflux-generate-fibo` - Generate images from text prompts with VLM-guided JSON expansion
  - `mflux-refine-fibo` - Refine images using structured JSON prompts for targeted attribute editing
  - `mflux-inspire-fibo` - Extract structured prompts from reference images for style transfer and remixing
- **VLM-guided JSON prompting**: Automatically expands short text prompts into 1,000+ word structured schemas using a fine-tuned Qwen3-VL model

### 🔧 Restructure and 🔄 Breaking Changes

- **Common module reorganization**: Moved shared functionality to `models/common/` for better code reuse
  - Unified latent creators across model families
  - Centralized scheduler implementations
  - Common quantization utilities
  - Shared model saving functionality

### 👩‍💻 Contributors

- **Filip Strand (@filipstrand)**: FIBO model implementation, architecture, core development

---

## [0.11.1] - 2025-11-13

# MFLUX v.0.11.1 Release Notes

### 🎨 New Model Support

- **Qwen Image Edit Support**: Added support for the Qwen Image Edit model, enabling natural language image editing capabilities
- **New command**: `mflux-generate-qwen-edit` for image editing with text instructions
- **Multiple image support**: Edit images using multiple reference images via `--image-paths` parameter
- **Model**: Uses `Qwen/Qwen-Image-Edit-2509` for high-quality image editing
- **Quantization support**: Full support for quantized models (8-bit recommended for optimal quality)

### 🔧 Improvements

- **Dedicated Qwen Image command**: Added `mflux-generate-qwen` as a dedicated command for Qwen Image model generation. The `mflux-generate` command now only supports Flux models.
- **Image comparison utility refactoring**: Refactored `image_compare.py` into a cleaner class-based structure with static methods
- **Error handling**: Moved `ReferenceVsOutputImageError` to the main exceptions module for better organization

### 🔄 Breaking Changes

⚠️ **Qwen Image Command Change**: The Qwen Image model now requires using the dedicated `mflux-generate-qwen` command instead of `mflux-generate --model qwen`. This provides better separation between Flux and Qwen model families and improves command clarity.

### 👩‍💻 Contributors

- **Filip Strand (@filipstrand)**: Qwen Image Edit model implementation, code refactoring

---

## [0.11.0] - 2025-10-14

# MFLUX v.0.11.0 Release Notes

### 🎨 New Model Support

- **Qwen Image Support**: Added support for the Qwen Image text-to-image model, enabling a new generation of visual content creation
- **New command**: `mflux-generate` now supports Qwen models for image generation
- **Qwen-specific features**: Full LoRA support with Qwen naming conventions, img2img support, and optimized weight handling
- **Qwen-Image-mflux-6bit Model**: Added [filipstrand/Qwen-Image-mflux-6bit](https://huggingface.co/filipstrand/Qwen-Image-mflux-6bit) quantized model to HF

### 🏗️ Major Architecture Improvements

- **Package Restructure**: Complete reorganization of the codebase to support multiple model architectures
  - Moved from flat structure to organized `models/` hierarchy (`models/flux/`, `models/qwen/`, `models/depth_pro/`)
  - Better separation of concerns with dedicated model, variant, tokenizer, and weight handler modules
  - Improved maintainability and extensibility for future model additions
- **Namespace Package**: Converted mflux to a namespace package (in preparation for mflux.mcp extension)
- **Common Module**: Extracted shared functionality into `models/common/` for better code reuse
  - Unified LoRA handling across different model types
  - Shared attention utilities
  - Common download and weight management utilities

### 📊 Metadata Enhancements

- **XMP/IPTC Metadata Support**: Added comprehensive metadata support for professional workflows
  - Write XMP and IPTC metadata to generated images
  - Industry-standard metadata formats for better compatibility with professional image tools
  - Enhanced metadata reading and writing capabilities
- **New `mflux-info` command**: Display detailed metadata information from generated images
  - View generation parameters, model information, and settings
  - Extract metadata from any mflux-generated image
  - Professional-grade metadata inspection

### 🔧 Scheduler System

- **Scheduler Interface**: Introduced a new scheduler abstraction for better extensibility
  - Clean interface for implementing custom sampling schedulers
  - Foundation for future scheduler additions (Euler, DPM++, etc.)
  - Current implementation: Linear scheduler (existing behavior preserved)
- **Scheduler Selection**: Added `--scheduler` command-line argument for choosing schedulers

### 🐛 Bug Fixes

- **Non-Quantized Model Loading**: Fixed critical bug where locally saved non-quantized models failed to load properly
- **Model Weight Handling**: Improved weight loading reliability for edge cases

### 🔧 Developer Experience

- **MLX 0.29.2 Support**: Updated MLX dependency to support the latest version (mlx>=0.27.0,<0.30.0)
- **Python 3.13 Support**: Unblocked sentencepiece and torch dependencies for Python 3.13
  - Updated dependency specifications for better Python 3.13 compatibility
  - Ensured smooth experience on latest Python versions
- **Test Improvements**: Enhanced image comparison logic to allow similar images that are "close enough"
  - More robust test suite that accommodates minor numerical differences
  - Reduced false positives in image generation tests
- **CI Updates**: Removed Claude CI agent (replacement coming soon)

### 🔄 Breaking Changes

⚠️ **Import Path Changes**: Due to the package restructure, some internal import paths have changed. If you're using mflux as a library and importing internal modules directly, you may need to update your imports:
- Flux modules moved from `mflux.flux.*` to `mflux.models.flux.*`
- Common utilities moved to `mflux.models.common.*`
- CLI tools remain unchanged and fully backward compatible

### 👩‍💻 Contributors

- **Filip Strand (@filipstrand)**: Qwen model support, package restructure, core development
- **Alessandro Rizzo (@azrahello)**: XMP/IPTC metadata support, info command implementation
- **Anthony Wu (@anthonywu)**: Scheduler interface, namespace package conversion, Python 3.13 improvements, bug fixes

---

## [0.10.0] - 2025-08-04

# MFLUX v.0.10.0 Release Notes

### 🎨 Model Improvements

- **FLUX.1 Krea [dev] Support!**
- **FLUX.1-Krea-dev-mflux-4bit Model**: Added [filipstrand/FLUX.1-Krea-dev-mflux-4bit](https://huggingface.co/filipstrand/FLUX.1-Krea-dev-mflux-4bit) quantized model to HF
- **FLUX.1-Kontext-dev-mflux-4bit Model**: Added [akx/FLUX.1-Kontext-dev-mflux-4bit](https://huggingface.co/akx/FLUX.1-Kontext-dev-mflux-4bit) quantized model to HF, contributed by @akx

### ✨ New Features

- **5-bit Quantization Support**: Added support for 5-bit quantization as a new option alongside existing 3, 4, 6, and 8-bit quantization levels

### 🔧 Improvements

- **Enhanced Default Inference Steps**: Increased default inference steps for dev models from 14 to 25 for improved image quality
- **Multiple Model Aliases Support**: Improved model configuration system to properly support multiple aliases per model, making model selection more flexible and robust

### 🐛 Bug Fixes

- **LoRA Resume Training**: Fixed critical bug where adapters created after training interruption would fail to load for generation with `AttributeError: 'list' object has no attribute 'weight'`. The issue occurred because the resume loading logic wasn't properly handling layers that are legitimately lists in the transformer architecture (like `attn.to_out`). (see [#224](https://github.com/filipstrand/mflux/issues/224))

### 🔧 Technical Requirements

- **MLX Compatibility**: This release assumes MLX 0.27.0 and upwards for optimal performance and compatibility
- **MLX Compatibility for test**: Fix MLX version to 0.27.1 for image generation tests
- **Non-strict Weight Updates**: Explicitly added non-strict mode (`strict=False`) for weight updates to maintain compatibility with later MLX versions that enforce stricter weight validation by default

### 👩‍💻 Developer Experience

- **Streamlined Release Process**: Removed TestPyPi publishing step from release workflow for simplified deployment

### 🙏 Contributors

- **[@filipstrand](https://github.com/filipstrand)** - FLUX.1 Krea [dev] model support, 5-bit quantization, enhanced defaults, and various improvements
- **[@akx](https://github.com/akx)** - Added 4-bit quantized Kontext model to HF

---

## [0.9.6] - 2025-07-20

# MFLUX v.0.9.6 Release Notes

### 🔧 Technical Details

- Cap the upper MLX dependency to a known working version (0.26.1) to avoid compatibility issues with newer MLX releases that enforce stricter weight validation (see [#238](https://github.com/filipstrand/mflux/pull/238))

## [0.9.5] - 2025-07-17

# MFLUX v.0.9.5 Release Notes

### 🐛 Bug Fixes

- **Fixed faulty imports**: Corrected import issues in the mflux module to ensure proper package initialization and functionality

## [0.9.4] - 2025-07-17

# MFLUX v.0.9.4 Release Notes

### 🛠️ Dependency Updates

- Expanded MLX dependency range from `mlx>=0.22.0,<=0.26.1` to `mlx>=0.22.0,<0.27.0` to support newer MLX versions

### 🔧 Developer Experience

- Refactor the release script into a reusable Python module for better maintainability

## [0.9.3] - 2025-07-08

# MFLUX v.0.9.3 Release Notes

### 😖 Revert "Offline Resilience" change

On a "cold start" where user has not previously downloaded the requested model, the workflow does not successfully request the download of all the expected files, blocking the image generation workflow for first time users. The feature will be re-evaluated carefully after this hot fix.

## [0.9.2] - 2025-07-08

# MFLUX v.0.9.2 Release Notes

### 🏗️ Build System Improvements

- **Updated build backend**: Migrated from setuptools to modern `uv build` backend for faster and more reliable package builds
- **Enhanced artifact exclusion**: Optimized distribution packages by excluding documentation assets (~27MB) and example images (~5MB) from published packages
- **New `make build` command**: Added development build command for testing distribution packages and validating sizes

### 🗃️ Offline Resilience

- **Local-first behavior**: Implemented cache-first downloading to improve resilience when HuggingFace Hub or network connectivity is unavailable
- **Graceful fallback**: System automatically uses cached model files when available, falling back to downloads only when necessary
- **Improved reliability**: Enhanced model loading reliability in environments with unstable internet connections

### 🔧 Developer Experience

- **Release script improvements**: Enhanced release automation with better error handling and duplicate version detection
- **Build system fixes**: Fixed minor typos in Makefile that could cause build issues

## Contributors

- **Anthony Wu (@anthonywu)**: Build system modernization, offline resilience implementation
- **Filip Strand (@filipstrand)**: Release automation improvements, build fixes

---

## [0.9.1] - 2025-07-04

# MFLUX v.0.9.1 Release Notes

### 🛠️ Dependency Fixes

- Restricted MLX dependency upper bound to **0.26.1** (`mlx>=0.22.0,<=0.26.1`) to prevent incompatibility issues with MLX 0.26.2.

### 🎨 Inpaint Mask Tool Improvements

- Enhanced interactive inpaint masking tool with additional shape options (ellipse, rectangle, and free-hand drawing).
- Added eraser mode for precise mask corrections.
- Implemented undo/redo history for non-destructive editing when crafting masks.

### 👩‍💻 Developer Experience

- Introduced initial `mypy` static-type checking configuration and performed a first round of type-hint clean-up across the codebase.
- Upgraded *pre-commit* hooks and addressed newly surfaced lint warnings for a cleaner commit experience.

## Contributors

- **Filip Strand (@filipstrand)**
- **Anthony Wu (@anthonywu)**

---

## [0.9.0] - 2025-06-28

# MFLUX v.0.9.0 Release Notes

## Major New Features

### 📸 FLUX.1 Kontext

- **Added FLUX.1 Kontext support**: Official Black Forest Labs model for character consistency, local editing, and style reference
- **New command**: `mflux-generate-kontext` for image-guided generation with text instructions
- **Advanced image editing capabilities**: Sequential editing, style transfer, character consistency, and local modifications
- **Comprehensive documentation**: Detailed prompting guide with tips, templates, and best practices
- **Automatic model handling**: Uses `dev-kontext` model configuration with optimized defaults

### 🖼️ Scale Factor Support for Image Upscaling

- **Enhanced upscaling dimensions**: Added support for scale factors (e.g., `2x`, `1.5x`) in addition to absolute pixel values
- **Mixed dimension types**: Ability to combine scale factors and absolute values (e.g., `--height 2x --width 1024`)
- **Auto dimension handling**: Use `auto` to preserve original image dimensions
- **Safety warnings**: Automatic warnings when requested dimensions exceed recommended limits
- **Pixel-perfect scaling**: Scale factors automatically align to 16-pixel boundaries for optimal results

### ⌨️ Shell Completions

- **ZSH completion support**: Full tab completion for all mflux CLI commands and arguments
- **Smart completions**: Context-aware completions for model names, quantization levels, LoRA styles, and file paths
- **Easy installation**: Simple `mflux-completions` command for automatic setup
- **Dynamic generation**: Completions stay in sync with code changes and new commands
- **Comprehensive coverage**: Supports all 15+ mflux commands with proper argument validation

### 🗂️ Cache Management Improvements

- **Platform-native caching**: Uses `platformdirs` for macOS-idiomatic cache locations (`~/Library/Caches/mflux/`)
- **Automatic migration**: Seamless migration from legacy `~/.cache/mflux` to new platform-appropriate locations
- **Environment variable support**: `MFLUX_CACHE_DIR` for custom cache locations
- **Improved organization**: Separate cache directories for different types of data (models, LoRAs, etc.)
- **Backward compatibility**: Automatic symlink creation for legacy path compatibility

## Breaking Changes

### 🔧 Python API Class Naming Standardization

- **Class rename**: `FluxInContextFill` is now `Flux1InContextFill` to follow consistent naming convention
- **Class rename**: `FluxConceptFromImage` is now `Flux1ConceptFromImage` to follow consistent naming convention
- **Breaking change for library users**: If you import these classes directly in Python code, you may need to update your imports
- **CLI tools unaffected**: All command-line tools (`mflux-generate-*`) continue to work without changes

## Contributors

Contributors:
- **Anthony Wu (@anthonywu)**: Scale factor support, shell completions, cache refactor
- **Filip Strand (@filipstrand)**: Kontext support, class naming standardization, core development

## [0.8.0] - 2025-06-14

# MFLUX v.0.8.0 Release Notes

## Experimental AI Features

### 👗 CatVTON (Virtual Try-On)
- **[EXPERIMENTAL]** Added virtual try-on capabilities using in-context learning via `mflux-generate-in-context-catvton`
- Support for person image, person mask, and garment image inputs for comprehensive virtual clothing try-on
- Automatic prompting for virtual try-on scenarios with optimized default prompts
- Side-by-side generation showing garment product shot alongside styled result
- AI-powered virtual clothing fitting with realistic lighting and fabric properties

### ✏️ IC-Edit (In-Context Editing)
- **[EXPERIMENTAL]** Added natural language image editing capabilities via `mflux-generate-in-context-edit`
- Natural language image editing using simple text instructions like "make the hair black" or "add sunglasses"
- Automatic diptych template formatting for optimal editing results
- Optimal resolution auto-sizing for 512px width (the resolution IC-Edit was trained on)
- Specialized LoRA automatically downloaded and applied for enhanced editing capabilities

## Enhanced Generation Control

### 🔎 Image Upscaling
- **Built-in upscaling capabilities**: Enhanced image quality and resolution enhancement for generated images
- Seamless integration with existing generation workflow
- Professional-grade upscaling for production-ready outputs

## Interpretability research

### 🧠 Concept Attention
- **Enhanced image generation control**: Fine-grained control over image generation focus areas using attention-based concepts
- Improved composition and subject handling for more precise artistic direction
- Advanced attention mechanisms for better understanding of prompt concepts

## Workflow & Performance Improvements

### 🪫 Battery Saver
- **Power management**: Automatic power optimization during extended generation sessions
- Configurable power-saving modes specifically designed for laptop users
- Smart resource management for long-running batch operations

### 📝 Prompt File Support
- **File-based prompt input**: Batch operations via `--prompt-file` for large-scale generation projects
- Dynamic prompt updates for large batch generation workflows
- Support for external prompt management and automation systems

### 🔄 Redux Function Balancing
- **Enhanced Redux capabilities**: Improved control over image-to-image transformation strength
- Better quality variations with adjustable parameters for more predictable results
- Refined Redux algorithm for more natural image variations

### 📥 Stdin Prompt Support
- **LLM Integration Ready**: Added support for providing prompts via stdin using `--prompt -`
- Enables seamless integration with LLMs and other text generation tools
- Supports both single-line and multi-line prompts through stdin
- Perfect for automation workflows and dynamic prompt generation
- Example usage: `echo "A beautiful landscape" | mflux-generate --prompt -`

## Developer Experience

### 🔧 LORA_LIBRARY_PATH Improvements
- **Unix-style resource discovery**: Enhanced LoRA library path handling for better organization
- Improved path handling for LoRA weight discovery across multiple directories
- Better cross-platform compatibility for LoRA management

### 🧪 Testing & Documentation
- New command-line arguments for both experimental features with comprehensive help
- Comprehensive argument parser tests for new functionality
- Updated documentation with experimental feature warnings and usage guidelines
- Added note about upcoming FLUX.1 Kontext model from Black Forest Labs

## Architecture Improvements

### 📚 Documentation Structure
- Refactored "In-Context LoRA" section to "In-Context Generation" with clear subcategories
- Enhanced documentation structure for better organization and user navigation
- Improved categorization of experimental vs stable features

### 🔄 Code Architecture Changes
- **Class rename**: `Flux1InContextLora` is now `Flux1InContextDev` to better reflect the dev model variant
- **Module reorganization**: Moved from `mflux.community.in_context_lora.flux_in_context_lora` to `mflux.community.in_context.flux_in_context_dev`
- **Breaking change for library users**: If you import the class directly, update your imports accordingly


### ⚡ Performance Optimizations
- Updated MLX dependency to latest version for improved performance and stability
- Removed PyTorch dependency for DepthPro model, significantly reducing installation requirements
- Streamlined dependencies for faster installation and reduced disk usage

## Experimental Notice

⚠️ **Important**: CatVTON and IC-Edit features are experimental and may be removed or significantly changed in future updates. These features represent cutting-edge AI capabilities that are still under active development.

## Contributors

Special thanks to the following contributors for their exceptional work since v0.7.1:
- **Anthony Wu (@anthonywu)**: Battery Saver implementation, Prompt File Support, Stdin Prompt Support, LORA_LIBRARY_PATH improvements
- **Alessandro (@azrahello)**: Redux Function Balancing enhancements
- **Filip Strand (@filipstrand)**: Core development, experimental features integration, infrastructure improvements

## [0.7.1] - 2025-05-06

# MFLUX v.0.7.1 Release Notes

## New Features

### 🎭 Multi-LoRA Support
- **Multiple LoRA Loading**: Added support for loading multiple LoRA adapters simultaneously when using the in-context feature
- Enhanced creative flexibility by combining multiple artistic styles in a single generation
- Reference: [GitHub Issue #178](https://github.com/filipstrand/mflux/issues/178)

## [0.7.0] - 2025-04-25
# MFLUX v.0.7.0 Release Notes

## Major New Features

### 🖌️ FLUX.1 Tools | Fill

- Added support for the FLUX.1-Fill model for inpainting and outpainting
- Introduced `mflux-generate-fill` command-line tool for selective image editing
- Implemented interactive mask creation tool to easily mark areas for regeneration
- Added outpainting capabilities with customizable canvas expansion
- Includes helper tools for creating outpaint image canvases and masks

### 🔍 FLUX.1 Tools | Depth

- Added support for the FLUX.1-Depth model for depth-conditioned image generation
- Implemented Apple's ML Depth Pro model in MLX for state-of-the-art depth map extraction
- Added `mflux-generate-depth` and `mflux-save-depth` command-line tools
- Added ability to use either auto-generated depth maps or custom depth maps

### 🔄 FLUX.1 Tools | Redux

- Added Redux tool as a new image variation technique
- Implemented a different approach compared to standard image-to-image generation
- Uses image embedding joined with T5 text encodings for more natural variations
- Added Redux-specific weight handlers and initialization

## New Models

### 🔎 Apple ML Depth Pro

- Added native MLX implementation of Apple's ML Depth Pro model for both separate use, and as a part of the Depth tool functionality

### 🖼️ Google SigLIP Vision Transformer

- Added SigLIP vision model for the Redux functionality

## Architecture Improvements

### 💾 Weight Management Improvements

- Added support for saving MFLUX version information in model metadata

### 🧠 Memory Optimization

- Additional improvements to the `--low-ram` option
- Better memory management for image generation models

## Contributors

- @anthonywu 
- @ssakar 
- @akx 

## [0.6.2] - 2025-03-13

# MFLUX v.0.6.2 Release Notes

## Bug Fixes

### 💾 Model Saving Fix
- **Fixed local model saving**: Resolved bug preventing users from saving models locally with `mflux-save`
- Restored full functionality for local model storage and management

## [0.6.1] - 2025-03-11

# MFLUX v.0.6.1 Release Notes

## Bug Fixes

### 🛑 Image Generation Interruption
- **Fixed interruption flow**: Properly handles interruptions during image generation, ensuring graceful stops even when no callbacks are registered
- **Keyboard interrupt handling**: Ensures image generation can be stopped via Ctrl+C in all diffusion model variants (standard Flux, ControlNet, and In-Context LoRA)
- Relocated `StopImageGenerationException` from stepwise handler to main generation functions for more robust interruption system

## Test Stability Improvements

### 🧪 Test Reliability
- **Fixed sporadic test failures**: Resolved intermittent failures in auto-seeds test case when using random seed count of 1
- Improved test consistency and reliability

## Code Quality Improvements

### 🔧 Code Standards
- **Formatting and linting fixes**: Fixed various formatting issues that were missed in the v0.6.0 release
- Enhanced code consistency and maintainability

## [0.6.0] - 2025-03-05
# MFLUX v.0.6.0 Release Notes

## Major New Features

### 🌐 Third-Party HuggingFace Model Support
- Comprehensive ModelConfig refactor to support compatible HuggingFace dev/schnell models
- Added ability to use models like `Freepik/flux.1-lite-8B-alpha` and `shuttleai/shuttle-3-diffusion`
- New `--base-model` parameter to specify which base architecture (dev or schnell) a third-party model is derived from
- Maintains backward compatibility while opening up the ecosystem to community-created models

### 🎭 In-Context LoRA
- Added support for In-Context LoRA, a powerful technique that allows you to generate images in a specific style based on a reference image without requiring model fine-tuning
- Introduced a new command-line tool: `mflux-generate-in-context`
- Includes 10 pre-defined styles from the Hugging Face ali-vilab/In-Context-LoRA repository
- Detailed documentation on how to use this feature effectively with prompting tips and best practices

### 🔌 Automatic LoRA Downloads
- Added ability to automatically download LoRAs from Hugging Face when specified by repository ID
- Simplifies workflow by eliminating the need to manually download LoRA files before use

### 🧠 Memory Optimizations
- Added `--low-ram` option to reduce GPU memory usage by constraining the MLX cache size and releasing text encoders and transformer components after use
- Implemented memory saver for ControlNet to reduce RAM requirements
- General memory usage optimizations throughout the codebase

### 🗜️ Enhanced Quantization Options
- Added support for 3-bit and 6-bit quantization (requires mlx > v0.21.0)
- Expanded quantization options now include 3, 4, 6, and 8-bit precision

## ⚠️Breaking changes

Previously saved quantized models will not work for v.0.6.0 and later.  See #149 for more details.

## Interface Improvements

### 🔧 Modified Parameters

- The previous `--init-image-path` parameter is now `--image-path` 
- The previous `--init-image-strength` parameter is now `--image-strength` 

### 🖼️ Image Generation Enhancements
- Added `--auto-seeds` option to generate multiple images with random seeds in a single command
- Added option to override previously saved test images
- Added `--controlnet-save-canny` option to save the Canny edge detection reference image used by ControlNet
- Improved handling of edge cases for img2img generation

### 🔄 Callback System
- Implemented a general callback mechanism for more flexible image generation pipelines
- Added support for before-loop callbacks to accept latents
- Enhanced StepwiseHandler to include initial latent

## Architecture Improvements

### 🏗️ Code Refactoring
- Removed 'init' prefix for a more general interface
- Removed `ConfigControlnet` - the `controlnet_strength` attribute is now on `Config`
- Simplified quantization by removing unnecessary class predicates 
- Refactored model configuration system
- Refactored transformer blocks for better maintainability
- Unified attention mechanism in single and joint attention blocks
- Added support for variable numbers of transformer blocks
- Optimized with fast SDPA (Scaled Dot-Product Attention)
- Added PromptCache for small optimization when generating with repeated prompts

### 🧰 Developer Tools
- Added Batch Image Renamer tool as an isolated uv run script
- Added descriptive comments for attention computations

## Compatibility Updates
- Updated to support the latest mlx version
- Fixed compatibility issues with HuggingFace dev/schnell models

## Bug Fixes
- Fixed handling of edge cases for img2img generation
- Various small fixes and improvements throughout the codebase


## Contributors

- @anthonywu
- @ssakar
- @azrahello
- @DanaCase

## [0.5.1] - 2024-12-23

# MFLUX v.0.5.1 Release Notes

## Bug Fixes

### 🔧 LoRA Loading Fix
- **Quantized model LoRA compatibility**: Fixed critical bug where locally saved quantized models failed to set LoRA weights
- Users can now successfully combine local quantized models with external LoRA adapters
- Improved reliability for advanced workflows combining quantization and LoRA fine-tuning

## [0.5.0] - 2024-12-22

# MFLUX v.0.5.0 Release Notes

## Major New Features

### 🎛️ DreamBooth Fine-tuning
- **DreamBooth support**: Introduced V1 of fine-tuning support in MFLUX
- Enables custom model training for personalized image generation
- Full fine-tuning pipeline with training configuration options

## Architecture Improvements

### 🔧 Weight Management Overhaul
- **Rewritten LoRA handling**: Completely rewritten LoRA weight handling system
- Improved performance and reliability for LoRA operations
- Better support for complex LoRA workflows

## Developer Experience

### 🧪 Testing & Quality
- **Enhanced test coverage**: Added comprehensive tests for new and existing features
- Multi-LoRA testing support
- Local model saving test coverage

### 📊 New Dependencies
- **Matplotlib integration**: Added matplotlib for visualizing training loss during fine-tuning
- **TOML support**: Added TOML library for better handling of MFLUX version metadata
- Enhanced configuration management

## [0.4.1] - 2024-10-29

# MFLUX v.0.4.1 Release Notes

## Bug Fixes

### 🐛 Image Generation Fixes
- **Img2img resolution fix**: Fixed img2img functionality for non-square image resolutions
- Improved compatibility with various aspect ratios

## [0.4.0] - 2024-10-28

# MFLUX v.0.4.0 Release Notes

## Major New Features

### 🖼️ Image-to-Image Generation
- **Img2Img Support**: Introduced the ability to generate images based on an initial reference image
- Transform existing images using AI-powered generation techniques
- Control the strength of transformation to balance between original image preservation and creative generation
- Perfect for iterating on designs and creating variations of existing artwork

### 📊 Metadata-Driven Generation
- **Image Generation from Metadata**: Added support to generate images directly from provided metadata files
- Streamlined workflow for recreating images with specific parameters
- Enhanced reproducibility for professional and research workflows
- Automated parameter loading from previously generated images

### 🔍 Real-time Generation Monitoring
- **Progressive Step Output**: Optionally output each step of the image generation process for real-time monitoring
- Visual feedback during generation for better understanding of the AI process
- Debug and fine-tune generation parameters by observing intermediate steps
- Educational tool for understanding diffusion model progression

## Developer Experience Improvements

### 🛠️ Enhanced Command-Line Interface
- **Improved argument handling**: Enhanced parsing and validation for command-line arguments
- Better error messages and user guidance for parameter configuration
- More intuitive command structure for complex generation workflows

### 🧪 Testing & Quality Assurance
- **Automated Testing**: Added comprehensive automatic tests for image generation and command-line argument handling
- Improved reliability and stability for all generation modes
- Continuous integration testing for better code quality

### 🔧 Development Workflow
- **Pre-Commit Hooks**: Integrated pre-commit hooks with `ruff`, `isort`, and typo checks for better code consistency
- Enhanced developer experience with automated code quality checks
- Streamlined contribution process for open source development

## [0.3.0] - 2024-09-24

# MFLUX v.0.3.0 Release Notes

## Major New Features

### 🕹️ ControlNet Support
- **ControlNet Canny support**: Added Canny edge detection ControlNet functionality for precise image control
- Enhanced control over image generation with edge-guided conditioning

## Model Export Improvements

### 📦 Advanced Model Export
- **Quantized model export with LoRA**: Added ability to export quantized models with LoRA weights baked in
- Streamlined deployment for fine-tuned models

## Developer Experience

### 🛠️ Development Tools
- **Enhanced development workflow**: Improved developer experience with uv, ruff, makefile, pre-commit hooks
- Better code quality tools and automated checks
- Streamlined contribution process

## Legal & Licensing

### ⚖️ Open Source License
- **Official MIT license**: Established clear open source licensing for the project
- Legal clarity for users and contributors

## [0.2.1] - 2024-09-14

# MFLUX v.0.2.1 Release Notes

## Improvements

### 🔧 LoRA Enhancements
- **Enhanced LoRA support**: Improved compatibility and performance for LoRA weight loading
- Better integration with existing workflows
- Refined handling of LoRA adapters

## [0.2.0] - 2024-09-07

# MFLUX v.0.2.0 Release Notes

## Major Milestone

### 🚀 Official PyPI Release
- **First official PyPI release**: `pip install mflux` - making MFLUX easily installable for everyone
- Big thanks to @deto for letting us have the "mflux" name on PyPI!

## New Features

### 🎨 Core Image Generation
- **Command-line tools**: Introduced dedicated commands for better user experience
  - `mflux-generate` for generating images
  - `mflux-save` for saving quantized models to disk
- **🗜️ Quantization support**: Added support for quantized models with 4-bit and 8-bit precision
- **LoRA weights**: Added support for loading trained LoRA (Low-Rank Adaptation) weights
- **Automatic metadata**: Images now automatically save metadata when generated

## Developer Experience

### 📦 Distribution
- Official packaging and distribution through PyPI
- Simplified installation process for end users
- Professional project structure and naming
