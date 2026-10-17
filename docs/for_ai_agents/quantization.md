# Quantization in MFlux (reference for AI agents)

Scope: diffusion transformers (DiT) and text encoders. This file does not cover depth, upscale (SeedVR2), Redux, or ControlNet internals, except where noted.

Verified against: MLX 0.32.0 (`pyproject.toml` pins `mlx>=0.32.0,<0.33.0` on macOS), git HEAD `02bbe7b`. Paths are relative to `src/mflux/` unless noted.

## 1. Short answers

| Question | Answer |
|---|---|
| Are all weights quantized equally? | No. A module must expose `to_quantized` and pass the model-family predicate. `nn.Linear` and `nn.Embedding` are examples. A predicate can return a per-layer override (Qwen-Image keeps `img_mod_linear` at 8-bit when the request is 4-bit). |
| Is there activation quantization (W4A8 style)? | No. MFlux does weight-only quantization. `quantize_input` stays `False` in every call. `Fp8Linear.to_quantized` raises `ValueError` if `quantize_input=True`. |
| Which text encoders stay in BF16? | Qwen-Image 1.x, Qwen-Image-2.1 txt2img, Krea 2. These set `skip_quantization=True` on the component. Section 6 has the full table. Section 11 has the per-model matrix. |
| Why do they stay in BF16? | Code comments only: "Quantization causes significant semantic degradation" (Qwen) and "quantizing the TE degrades conditioning" (Krea 2). No measurement or issue number is in the repo for these flags. |
| Who decides? | The model author, in the `ComponentDefinition` of the family. The user cannot override it with a flag. |
| Lower limit on bits? | CLI accepts `-q 3, 4, 5, 6, 8`. No code blocks a text encoder from Q3. MLX itself accepts 2, 3, 4, 5, 6, 8 bits. See section 4. |
| Only affine quantization? | MFlux code paths use `mode="affine"` (the MLX default). One exception: the Lens GPT-OSS encoder reads `mode` from its own checkpoint config (MXFP4 experts). |
| NVFP4 support? | No. `nvfp4` appears nowhere in `src/`. MLX 0.32.0 supports it, but MFlux has no flag, no code, and no layout inference for it. |

## 2. Core mechanism

All quantization goes through `nn.quantize(model, class_predicate=..., bits=..., group_size=...)` from MLX. The wrapper is `WeightApplier` in `models/common/weights/loading/weight_applier.py`.

- `WeightApplier.apply_and_quantize(weights, models, quantize_arg, weight_definition)` handles multi-component models.
- `WeightApplier.apply_and_quantize_single(...)` handles one component. Lens and the Z-Image ControlNet use it. The Qwen-Image-2.1 ControlNet calls `nn.quantize` directly.
- `nn.quantize` quantizes a module only if it exposes `to_quantized` and the predicate accepts it. `nn.Linear` and `nn.Embedding` are examples.
- `nn.Linear` and `nn.Embedding` have `to_quantized`. `nn.Conv2d`, `LayerNorm` and `RMSNorm` do not (checked in the project venv).
- A quantized layer stores `weight` (packed `uint32`), `scales`, and `biases` (affine). A linear layer's own `bias` stays in the model dtype.
- `group_size` defaults to 64. Only `BooguWeightDefinition` overrides it (`quantization_group_size = 32`, because its 3360-wide linears do not divide by 64).
- The call sites never pass `mode=`. MLX defaults to `affine`.
- `mx.quantize` needs the last dimension divisible by `group_size`. A layer that does not divide makes the load fail, unless the predicate skips it (section 5).
- Quantization is lazy. Parameters are read and quantized when a forward pass first needs them (see `Qwen21Initializer.load_components`).

### 2.1 Entry points

| Source | Where |
|---|---|
| CLI `--quantize` / `-q` | `cli/parser/parsers.py` lines 156 and 253. `choices=ui_defaults.QUANTIZE_CHOICES`. |
| Choices list | `cli/defaults/defaults.py:64`, `QUANTIZE_CHOICES = [3, 5, 4, 6, 8]` |
| Python API | `quantize=<int>` on each model class, for example `ZImageTurbo(quantize=8)`. MFlux does not check the value. MLX raises if bits is not in 2, 3, 4, 5, 6, 8. |
| Save | `mflux-save` (`models/common/cli/save.py`) loads with `quantize=args.quantize`, then calls `ModelSaver.save_model`. |
| Training | `training_spec.quantize` goes to the model constructor (`models/common/training/runner.py`). The base is quantized and only the LoRA trains (QLoRA). |

## 3. Resolution of stored versus requested bits

`models/common/resolution/quantization_resolution.py`. `QuantizationResolution.resolve(stored, requested)` returns `(bits, warning)`.

| stored | requested | Rule | Result |
|---|---|---|---|
| None | None | `none` | No quantization. |
| None | N | `on_the_fly` | Quantize at N bits. |
| S | None | `pre_quantized` | Use S. |
| S | N | `conflict` | Use S. If S differs from N, print `Model is pre-quantized at S-bit. Ignoring -q N flag.` |

`stored` comes from safetensors metadata (`quantization_level`, a string) or from `model.safetensors.index.json` (`weight_loader.py` around line 265). A save writes one level per component directory (`model_saver.py:_save_weights`).

### 3.1 Three load flows in `apply_and_quantize`

1. `bits is None`: `_set_weights` only. Weights load as saved (BF16, or FP8 for Ideogram).
2. `stored is None` and `bits` set (on-the-fly): `_set_weights` first, then `_quantize`. The full-precision weights load, then MLX quantizes them.
3. `stored` set (pre-quantized checkpoint): `_quantize` first on the empty skeleton, then `_set_weights`. The skeleton must have the same layer types as the file, so the file's packed tensors fit.

### 3.2 Per-layer layout recovery (`_stored_layer_predicate`)

For a pre-quantized checkpoint, the predicate does not trust the global level. It reads each layer's stored tensors:

- `bits = stored_weight.shape[-1] * 32 / input_dims`
- `group_size = input_dims / scales.shape[-1]`
- If `{bits, group_size}` is in `{2, 3, 4, 5, 6, 8}` x `{32, 64, 128}`, it returns `{"bits": b, "group_size": g}` for that layer.
- If the layer has no `scales` in the file, the layer stays unquantized (`return False`).
- If the numbers do not match a known pair, it prints a warning and falls back to the requested level. `update()` may then reject the shapes.

This is how mixed-precision files reload. The metadata holds one number. The shapes hold the truth per layer.

The inference assumes the affine layout. It does not read or pass `mode`. An MXFP4 or NVFP4 layer saved by MFlux would not round-trip (MFlux never saves one).

## 4. Bit-width limits

| Layer | Limit | Source |
|---|---|---|
| CLI | 3, 4, 5, 6, 8 | `QUANTIZE_CHOICES` |
| Resolution tests | 3, 4, 5, 6, 8 | `tests/resolution/test_quantization_resolution.py` |
| Layout inference | 2, 3, 4, 5, 6, 8 bits and group 32, 64, 128 | `_INFERABLE_BITS`, `_INFERABLE_GROUP_SIZES` |
| MLX affine | 2, 3, 4, 5, 6, 8 | Verified: `mx.quantize(..., bits=1)` and `bits=7` raise. Bits 2, 3, 5 work. |

- No code forces a text encoder to stay at 8-bit or above. If a text encoder is not skipped, it receives the same `bits` as the transformer.
- 2-bit is reachable only through the Python API. The CLI rejects it, and no test covers it.
- Quality guidance is in READMEs only. `models/qwen/README.md` says 6-bit or below can degrade the image much more than FLUX.
- Ming Image is the exception: `--text-encoder-quantize` sets the text encoder level apart from `-q` (`ming_image_initializer.py` line 77). If you omit it, `-q` applies.
- Otherwise the same level applies to all quantized components in a run. `Qwen21Initializer.load_components` raises `ValueError` if two components resolve to different levels (`Conflicting component quantization levels`).

## 5. Which layers inside a component are quantized

A layer is quantized only if it has `to_quantized` and the family predicate accepts it. `nn.Linear` and `nn.Embedding` are examples. Default predicate: `hasattr(module, "to_quantized")`.

Always unquantized: convolutions, norms, positional or rotary buffers, scalar parameters, and the `bias` vector of a linear layer. Embeddings are quantized (`embed_tokens` in a text encoder) unless the predicate or the component skip blocks them.

| Family | Predicate behavior | File |
|---|---|---|
| FLUX.1, FLUX.2 Klein, Z-Image, Ernie, Boogu, Lens DiT, Ideogram 4 | Default. Every Linear and Embedding. | `*_weight_definition.py` |
| Qwen-Image 1.x (and edit) | Default, plus a 3-argument form `predicate(path, module, bits)`. If `bits == 4` and `".img_mod_linear"` is in the path, return `{"bits": 8}`. This is mixed precision. The reason is in the comment: adaLN modulation error grows over the denoising steps at 4-bit (issue #484). | `qwen/weights/qwen_weight_definition.py` |
| Qwen-Image-2.1 txt2img (`Qwen21WeightDefinition`) | Default. | `qwen21/weights/qwen21_weight_definition.py` |
| Qwen-Image-2.1 edit and ControlNet (`QwenImage21WeightDefinition`) | Linear/Embedding only if `weight.shape[-1] % 64 == 0` and the path does not contain `modulation`, `time_text_embed`, or `norm_out`. | `qwen21/weights/qwen_image21_weight_definition.py` |
| Krea 2 | Skip any layer whose last dimension is not divisible by 64 (for example the `txtfusion` projector, Linear 12 to 1). | `krea2/weights/krea2_weight_definition.py` |
| FIBO | Skip Conv2d, shape `(1152, 4304)`, and last dimension not divisible by 64. | `fibo/weights/fibo_weight_definition.py` |
| FIBO VLM | Skip any path containing `lm_head`. | `fibo_vlm/weights/fibo_vlm_weight_definition.py` |
| Ming Image | Default. Routers stay full precision because gate modules have no `to_quantized`. | `ming_image/weights/ming_image_weight_definition.py` |
| Redux | Skip Conv2d, shape `(1152, 4304)`, and last dimension not divisible by 64. | `flux/weights/flux_weight_definition.py` |

The 3-argument predicate form is detected with `inspect.signature` in `WeightApplier._predicate_with_bits`. Keyword-only third parameters do not count.

A predicate may return `True`, `False`, or a dict such as `{"bits": 8}` (and `group_size`). MLX treats a dict as the per-layer override.

### 5.1 VAE note (from code reading, not run)

Flux, Flux2, and Z-Image pass `"vae"` in the `models` dict. Their VAE attention blocks use `nn.Linear(512, 512)` (`flux_vae/common/attention.py`, `flux2_vae/common/attention.py`, `z_image_vae/common/attention.py`). The default predicate accepts them, and 512 divides by 64. So these few linear layers appear to be quantized with the DiT. All VAE convolutions stay unquantized. `QwenImage21WeightDefinition` sets `skip_quantization=True` for its VAE and uses FP32.

## 6. Component-level skip (`skip_quantization`)

Defined in `ComponentDefinition` (`models/common/weights/loading/weight_definition.py`, default `False`).

| Component | Skipped? | Comment in code |
|---|---|---|
| Qwen-Image 1.x text encoder (Qwen2.5-VL) | Yes | "Quantization causes significant semantic degradation" |
| Qwen-Image-2.1 txt2img text encoder (Qwen3-VL) | Yes | Same comment |
| Krea 2 text encoder | Yes | "quantizing the TE degrades conditioning" |
| Ming Image `heads` | Yes | "~30M params that feed the DiT directly" |
| Qwen-Image-2.1 edit/ControlNet VAE | Yes | none |
| FLUX.1 T5 and CLIP | No | |
| FLUX.2 Klein text encoder (Qwen3) | No | |
| Z-Image text encoder | No | |
| Ernie text encoder | No | |
| Boogu text encoder (Qwen3-VL language model) | No | |
| Ming Image text encoder (`mllm`) and connector | No | |
| FIBO text encoder | No | |
| Ideogram 4 text encoder (FP8 layers) | No. `Fp8Linear.to_quantized` converts it. | |
| Qwen-Image-2.1 **edit** text/vision encoder | No. `reference/README.md` says eligible encoder layers quantize. | |
| Lens GPT-OSS encoder | Not applicable. It never enters `WeightApplier`. | |

Inconsistency to know: Qwen-Image-2.1 txt2img keeps the text encoder in BF16, but the 2.1 edit and ControlNet definitions do not set the flag. `qwen21/README.md` says the text encoder "is never quantized". `qwen21/reference/README.md` says eligible text/vision encoder layers quantize. The code follows the second statement for edit.

Why the skip exists: the repo gives no measurement. Both `qwen/README.md` and `qwen21/README.md` state that `-q` does not shrink the encoder (about 14 to 17.5 GB stays resident). The flag came with the Z-Image and Qwen work (`git log -S skip_quantization` shows `37d202d` for Qwen).

### 6.1 Skip with a pre-quantized checkpoint

`_quantize` and `apply_and_quantize_single` treat `skip_quantization` as "do not quantize BF16 weights on load". They do not block a checkpoint that already ships quantized tensors for that component:

- Component has stored weights: the stored-layout predicate builds quantized modules where `scales` exist. Layers without `scales` stay Linear.
- Component has no weights in the file: the skip holds.
- `mflux-save -q 8` on Qwen writes the text encoder in BF16. It reloads as BF16.

Test: `tests/weights/test_skip_quantization_prequantized.py`.

## 7. Special paths

### 7.1 Ideogram 4 (FP8 source)

- Components load with `loading_mode="fp8_safetensors"`. Layers are `Fp8Linear` (`ideogram4/model/ideogram4_transformer/fp8_linear.py`): `uint8` weight plus a per-row `float32` `weight_scale`, dequantized in every forward with `mx.from_fp8`.
- `Fp8Linear.to_quantized(group_size, bits, mode="affine", quantize_input=False)` makes `nn.quantize` reach these layers. Weights already loaded: dequantize through the row scale, then `mx.quantize`. Weights not loaded (skeleton for a saved checkpoint): return an empty `QuantizedLinear`.
- `Ideogram4Initializer._rebuild_q8_folded_layers` rebuilds layers that a LoRA bake or a save already turned into quantized form, inferring bits and group size from shapes.
- On-disk size: FP8 26 GB, `-q 8` 26 GB, `-q 4` 14 GB (`ideogram4/README.md`).
- Tests: `tests/test_ideogram4_quantization.py`, `tests/image_generation/test_ideogram4_quantized_roundtrip.py`.

### 7.2 Lens

- DiT: `apply_and_quantize_single` with `LensWeightDefinition.quantization_predicate`. The `-q` flag affects only this part.
- Text encoder: `LensGptOssEncoder` (`lens/model/text_encoder/lens_gpt_oss_encoder.py`) loads `mlx-community/gpt-oss-20b-MXFP4-Q8`. It reads `config["quantization"]`, then calls `nn.quantize(..., group_size, bits, mode=quantization.get("mode", "affine"))`. A path listed in the config uses its own settings (MXFP4 experts, Q8 elsewhere). `-q` has no effect on it.
- This is the only place in `src/` where a non-affine mode can run, and it follows the checkpoint, not an MFlux option.

### 7.3 ControlNets

- Qwen-Image-2.1 ControlNet: `nn.quantize(model.controlnet, bits=model.bits, class_predicate=QwenImage21WeightDefinition.quantization_predicate)`. A saved ControlNet with a different level than the base raises `ValueError`.
- Z-Image ControlNet: `apply_and_quantize_single` with its own predicate.

### 7.4 LoRA on quantized bases

- Runtime LoRA layers wrap `nn.QuantizedLinear` (`lora/layer/linear_lora_layer.py`). The base stays quantized. The delta stays dense.
- Bake (default) merges the delta into the weight. For a quantized base, `LoRASaver` dequantizes, merges, and requantizes (`lora/mapping/lora_saver.py`).
- Bake into a 4-bit (or lower) layer upgrades that layer to 8-bit with group size 64. Reason: at 4-bit the quantization step is larger than a small delta, so the delta disappears (issue #665). Test: `tests/lora/test_bake_low_bit_quantized.py`. Order test: `tests/lora/test_bake_folds_before_quantizing.py`.
- Result: after a bake, a "4-bit" model can hold some 8-bit layers. The layout recovery in section 3.2 handles this on reload.

### 7.5 Compute precision

`ComputePrecision` (`models/common/compute_precision/compute_precision.py`) casts attention and feed-forward parameters to float16. It runs after quantization and LoRA. Float activations against BF16 quantization scales would promote `quantized_matmul` to float32. `ComputePrecision.ensure_savable` blocks `mflux-save` for such a model. It is separate from bit-width quantization.

## 8. Saving and reloading

- `ModelSaver.save_model(model, bits, base_path, weight_definition)` writes each component in 2 GB shards (`{i}.safetensors`) with metadata `quantization_level` and `mflux_version`. It also writes `model.safetensors.index.json` with the same metadata.
- Saved parameters are the post-quantization tree: packed `weight`, `scales`, `biases` for quantized layers, plain tensors for the rest. A skipped component is saved in BF16 but is stamped with the same `quantization_level`.
- Loading a saved model with `-q N` ignores N and prints the conflict warning (section 3).
- Models saved with MFlux before v0.6.0 do not load. Re-save them (`models/common/README.md`).

## 9. Invariants for agents who change this code

1. Add a new family: write `quantization_predicate` in its `*WeightDefinition`. Skip layers where `weight.shape[-1] % group_size != 0`. Otherwise the load fails inside `mx.quantize`.
2. Add a text encoder skip: set `skip_quantization=True` on its `ComponentDefinition`. Add a comment with the evidence. Pre-quantized files still work (section 6.1).
3. Use a dict return (`{"bits": 8}`) for per-layer precision. Use the 3-argument predicate form if it depends on the requested level. Saves from before the change still load because of section 3.2.
4. Do not pass `mode=` or `quantize_input=True` to `nn.quantize` without changing `_stored_layer_predicate`, `Fp8Linear`, and `LoRASaver`. All three assume affine, weight-only layouts.
5. To add NVFP4 or MXFP4, add a `mode` option to the CLI, `WeightApplier._quantize`, the layout inference, the metadata, `LoRASaver`, and the tests. Nothing of this exists now. The scale layout differs from affine: MLX 0.32.0 gives `uint8` scales and no `biases`. NVFP4 uses group size 16. MXFP4 uses group size 32.
6. `QUANTIZE_CHOICES` is the only CLI gate on bits. The Python API has no gate besides MLX.

## 10. Quick lookup

| Task | Where |
|---|---|
| Change the allowed `-q` values | `cli/defaults/defaults.py:64` |
| Find why a layer is not quantized | The family predicate in `models/<family>/weights/*_weight_definition.py` |
| Find why a component is not quantized | `skip_quantization` in the same file |
| Change group size for a family | `quantization_group_size` class attribute (Boogu example) |
| Resolve a stored versus requested conflict | `models/common/resolution/quantization_resolution.py` |
| Debug a shape error on reload | `WeightApplier._stored_layer_predicate` warning text |
| Fast tests | `tests/resolution/test_quantization_resolution.py`, `tests/weights/test_skip_quantization_prequantized.py`, `tests/test_qwen_mixed_q4.py`, `tests/test_ideogram4_quantization.py` |

## 11. Per-model matrix (canonical copy: `docs/quantization_strategies.md`)

`docs/quantization_strategies.md` holds the maintained table (one row per model that `mflux-save` can save, FLUX.1 and SeedVR2 rows removed by request). Columns: TE Q, DiT Q, GS (group size), text encoder, HS (hidden size), tapped hidden states. Summary of what is in it:

| Text encoder | Models | HS | Quantized by `-q`? | Hidden states read |
|---|---|---|---|---|
| Qwen2.5-VL 7B | qwen-image, qwen-image-edit, qwen-image-edit-2511 | 3584 | No (skip) | Final |
| Qwen3-VL (8B-class) | qwen-image-2.1 | 4096 | No for txt2img (skip). Yes for the ControlNet-union definition. | Final |
| Qwen3-VL-4B | krea-2, krea-2-raw | 2560 | No (skip) | 12 taps: 2, 5, ... 35 |
| Qwen3-VL (FP8) | ideogram-4-fp8 | 4096 | Yes (FP8 converted to affine) | 13 taps: 0, 3, ... 33, 35 |
| Qwen3-VL | boogu-image-turbo | 4096 | Yes, group size 32 | Final |
| Qwen3-4B | flux2-klein-4b, flux2-klein-base-4b | 2560 | Yes | Layers 9, 18, 27 |
| Qwen3-8B-sized | flux2-klein-9b, 9b-kv, base-9b | 4096 | Yes | Layers 9, 18, 27 |
| Qwen3 | z-image, z-image-turbo, z-image-turbo-controlnet-union-2.1 | 2560 | Yes | Second from last |
| Ministral-3 3B | ernie-image, ernie-image-turbo | 3072 | Yes | Second from last |
| SmolLM3-3B | fibo, fibo-lite, fibo-edit, fibo-edit-rmbg | 2048 | Yes | Second from last |
| Ling-mini-2.0 MoE | ming-image-design | 2048 | Yes, with its own `--text-encoder-quantize` | Layer 20 (final) plus layers 5, 12, 20 (DirectVLM) |
| T5-XXL + CLIP-L | FLUX.1 family (dev, schnell, krea-dev, kontext, fill, depth, redux, controlnets) | 4096 / 768 | Yes (both) | T5 final after final layer norm. CLIP pooled at the EOS token. |
| GPT-OSS 20B (MXFP4-Q8, pre-quantized) | lens-turbo (not saveable) | n/a | Not applicable | Layers 5, 11, 17, 23, no final norm |

Hidden sizes come from constructor defaults in each encoder class, except the FLUX.1 values (standard T5-XXL and CLIP-L sizes, not read from the repo). Qwen3-4B and Qwen3-8B names for Klein are inferred from `text_encoder_overrides` in `model_config.py` (hidden 2560/inter 9728 and hidden 4096/inter 12288). The repo does not name those sizes.

## 12. Hidden-state tap styles

Three styles exist in the encoders:

1. **Final**: last layer output (Qwen-Image 1.x, Qwen-Image-2.1, Boogu, T5).
2. **Single intermediate**: second from last layer (Z-Image `all_hidden_states[-2]`, Ernie layer `num_hidden_layers - 2`, FIBO `hidden_states_list[-2]`).
3. **Multi-layer stack**: several layers concatenated or stacked (Klein 9, 18, 27. Krea 2 twelve taps. Ideogram 4 thirteen taps. Lens four taps. Ming 5, 12, 20).

Relevance to quantization:
- The skip flag does not follow the tap style. Qwen 1.x and 2.1 (final) and Krea 2 (12 taps) are skipped. Klein, Ideogram (multi-tap) and Z-Image (single) are quantized.
- Hypothesis, not tested in this repo: quantization error accumulates with depth, so any style that reads late layers sees the sum of all earlier layers' error. Multi-tap styles also expose early-layer error directly.
- When you tap layers `k < N`, layers after the last tap run but their output is unused for that tap (Z-Image, Ernie, FIBO read layer `N-2`, so the last layer's weights affect nothing). Quantizing them gains memory and loses nothing. This is an inference from the code, not a measured result.

## 13. Ernie text encoder identity and drop-in rules

Findings from the repo and the web (checked 2026-10-08):

- `ErnieMistralTextEncoder` (`ernie_image/model/ernie_text_encoder/text_encoder.py`) is a text-only decoder: 26 layers, hidden 3072, 32 heads, 8 KV heads, head dim 128, MLP 9216, vocab 131072, rope theta 1e6, max positions 262144.
- `ErnieWeightDefinition` loads `text_encoder/` with `weight_prefix_filters=["language_model.model."]` and no key mapping. The vision tower in the file is ignored.
- `baidu/ERNIE-Image/text_encoder/config.json`: architecture `Mistral3Model`, `model_type: mistral3` (text `ministral3`, vision `pixtral`), hidden 3072, 26 layers, 32 heads, intermediate 9216, vocab 131072, RoPE type **YARN** with theta 1e6, bfloat16, vision tower 24 layers, no `_name_or_path`.
- ComfyUI docs list `ministral-3-3b.safetensors` (repo `Comfy-Org/ERNIE-Image`) as the ERNIE-Image text encoder.
- Baidu's model card says nothing about the text encoder. No source states whether the weights are stock, base or instruct, or fine-tuned.

Drop-in rules for a Ministral-3 VLM checkpoint:
1. Shapes must match the constants above, or the load fails.
2. Text keys must sit under `language_model.model.`. Other layouts need a key mapping (none exists).
3. The tokenizer must match. Ernie ships its own `tokenizer/` folder.
4. The DiT was trained on one set of features. Any other fine-tune (base versus instruct) gives off-distribution features even when shapes match.
5. Possible mismatch in MFlux: the config says YARN RoPE but the class computes plain RoPE (`inv_freq = 1 / theta^(...)`, lines 139 to 144). Effect on short prompts is untested.

Cheap verification that loads no model: compare safetensors headers (key names and shapes), then tensor hashes, against the `mistralai` Ministral-3 3B release.

## 14. Group size summary

| Case | Group size |
|---|---|
| Default | 64 (`_quantize` passes 64. `apply_and_quantize_single` passes none, so MLX uses 64) |
| Boogu | 32 (whole model) |
| LoRA bake into layer < 8-bit | 64, layer becomes 8-bit |
| LoRA bake into 8-bit layer | the layer's own group size |
| Pre-quantized checkpoint | stored size, inferred as 32, 64 or 128 |
| Lens GPT-OSS encoder | from checkpoint `config.json` (value not checked) |

MFlux never sets a per-layer group size when quantizing. Mixed group sizes arise only from loaded checkpoints, after a bake.

## 15. Ming Image: separate text-encoder level

`MingImage(quantize=..., text_encoder_quantize=...)` and CLI `--text-encoder-quantize`. `ming_image_initializer.py` line 77 passes `text_encoder_quantize` if set, otherwise `quantize`. This is the only model with two independent levels. README example: `-q 8 --text-encoder-quantize 4` (8-bit DiT and connector, 4-bit MoE encoder). The `heads` component (about 30M parameters) is skipped.

## 16. Known gaps and inconsistencies (found by code reading)

1. Qwen-Image `img_mod_linear` protection is `bits == 4` only. At `-q 3` these layers stay at 3-bit.
2. Qwen-Image-2.1 txt2img skips the text encoder. The 2.1 edit and ControlNet definition does not. `qwen21/README.md` says the encoder "is never quantized". `qwen21/reference/README.md` says eligible encoder layers quantize.
3. The VAE attention `nn.Linear` layers (FLUX.1, FLUX.2, Z-Image, and SeedVR2) are in the `models` dict and the default predicate accepts them, so they appear to be quantized. All VAE convolutions stay in full precision.
4. `_INFERABLE_BITS` includes 2, but the CLI rejects `-q 2`. The Python API passes 2 to MLX, which accepts it. No test covers it.
5. Layout recovery assumes affine tensors. A non-affine layer (MXFP4, NVFP4) would not round-trip. Only Lens loads such layers, through its own code.
6. A skipped component in a saved model is stamped with the model's `quantization_level` although it is BF16.
7. Ernie RoPE: YARN in the checkpoint config, plain RoPE in the class (section 13).

## 17. How this document was verified

Read from source: all `*_weight_definition.py` files, `weight_applier.py`, `quantization_resolution.py`, `model_saver.py`, `lora_saver.py`, `fp8_linear.py`, the Lens encoder, initializers, encoder classes (taps), `model_config.py` overrides, READMEs.

Executed in the project venv (no model weights loaded):
- `inspect.signature` of `nn.quantize` and `mx.quantize`.
- `mx.quantize` on a random 128x128 array: affine group 64 bits 4 gives `uint32` weight, bfloat16 scales and biases. `mxfp4` (group 32), `nvfp4` (group 16) and `mxfp8` (group 32) give `uint8` scales and no biases. Bits 1 and 7 raise. Bits 2, 3, 5 work.
- `hasattr(cls, "to_quantized")`: true for `nn.Linear` and `nn.Embedding`, false for `Conv2d`, `LayerNorm`, `RMSNorm`, `QuantizedLinear`.

Not done: no model load, no inference, no image comparison, no measurement of quality at any bit width. Statements about quality come from READMEs or code comments only.

Web sources: [baidu/ERNIE-Image](https://huggingface.co/baidu/ERNIE-Image), its [text_encoder config.json](https://huggingface.co/baidu/ERNIE-Image/raw/main/text_encoder/config.json), [ComfyUI ERNIE-Image docs](https://docs.comfy.org/tutorials/image/ernie-image).

Re-verify commands:
- `grep -rn "skip_quantization" src/mflux/models`
- `grep -rn "quantization_predicate" src/mflux/models`
- `grep -rn "nvfp4\|mxfp" src` (expect only Lens comments)
- `sed -n 64p src/mflux/cli/defaults/defaults.py` (expect `QUANTIZE_CHOICES`)
