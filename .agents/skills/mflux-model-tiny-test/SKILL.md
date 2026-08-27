---
name: mflux-model-tiny-test
description: Write a hermetic "tiny" model-saving test for an mflux model — a fast twin of the slow save/load test that runs the real ModelSaver/WeightLoader/WeightApplier seam on real component classes at toy dimensions, with no downloads or image generation. Use when asked to "make tiny test for <model>".
---
# mflux tiny model-saving test

A tiny test proves a model's quantized checkpoint survives a save/load roundtrip byte-for-byte, in about a second, without downloading anything. It builds the model's **real** component classes at toy dimensions and pushes them through the **real** save/load code path.

## When to Use

- You're asked to "make tiny test for `<model>`" or add a fast twin of a slow save/load test.
- You ported a new model (see `mflux-model-porting`) and want cheap checkpoint coverage.

Related: `mflux-testing` for running the suite and golden-image rules.

## Reference implementations

Read these before writing a new one — they are the source of truth:

- `tests/model_saving/tiny_checkpoint_helper.py` — the shared harness
- `tests/model_saving/test_tiny_model_saving_ernie_image.py` — components take explicit kwargs
- `tests/model_saving/test_tiny_model_saving_ideogram4.py` — components take a Config dataclass

Do not trust code quoted in this skill over those files. The pattern landed as an RFC (PR #599, "proposal + demo on 2 models") and is expected to evolve.

**Models that already have tiny tests:** `ernie_image`, `ideogram4`.

## What you write

One file, no `src/` changes:

```text
tests/model_saving/test_tiny_model_saving_<model>.py
```

The whole test is a single call into `TinyCheckpointRoundtrip`. The real work is figuring out how to build that model's components small. Match the two examples exactly:

- class `TestTiny<Model>ModelSaving`
- method `test_tiny_quantized_checkpoint_roundtrips_exactly`, decorated `@pytest.mark.fast`
- a `@staticmethod _tiny_components()` returning the component dict
- a comment naming the slow twin it mirrors

## Instructions

### 1. Read the weight definition — it is the contract

```text
src/mflux/models/<model>/weights/<model>_weight_definition.py
```

`get_components()` is authoritative: it gives the exact set of component **names** you must produce, and how many. While there, note two things:

- **`quantization_group_size`** — a class attribute on the weight definition. Defaults to 64 but is not always 64 (`boogu` uses 32). Every tiny dimension must be a multiple of *this model's* value. Check with `grep -n "quantization_group_size" src/mflux/models/<model>/weights/*.py`.
- **`skip_quantization=True`** on any component (`krea2` and `qwen` each have one). In production, `WeightApplier._quantize` honors this flag and leaves the component unquantized. **`TinyCheckpointRoundtrip._quantize` does not** — it calls `nn.quantize` on every saved component regardless of the flag, so a `skip_quantization` component comes out of the "saved" side quantized (with `scales`/`biases`) while the reloaded "fresh" side stays unquantized (no `scales`/`biases`). The helper's exact key-set comparison (`saved_weights.keys() == fresh_weights.keys()`) will then fail for that component. This is a real gap in `tiny_checkpoint_helper.py`, not a quirk to work around in the test file — if you're writing a tiny test for `krea2`, `qwen`, or any other model with a `skip_quantization` component, stop and fix `TinyCheckpointRoundtrip._quantize` to skip those components (mirroring `WeightApplier._quantize`) rather than special-casing it in the test.

### 2. Find each component's real class and read its `__init__`

Derive the shrink knobs from the actual constructor. Do not pattern-match one example — they differ deliberately: ERNIE's components take explicit kwargs (`hidden_size=`, `num_layers=`), Ideogram 4's transformer takes an `Ideogram4Config` dataclass.

Look under `src/mflux/models/<model>/model/`. Shrink layer counts to 2, hidden and intermediate sizes to 128, head counts to 2, vocab to 128.

### 3. Use `TinyVAEStandIn` for the VAE

The real VAEs are fixed-size (~170M param) architectures with no dimension knobs, so they cannot be shrunk. `ModelSaver`/`WeightLoader` treat every component as an opaque parameter tree, so the stand-in walks the identical code path. Don't spend a loop trying to shrink a real VAE.

### 4. Respect interlocking dimension constraints

Dimensions are not independent. Look for constraints in the component's `__init__` and attention/rope code:

- `rope_axes_dim` must **sum to** `head_dim` (ERNIE: `[16, 24, 24]` → 64)
- `head_dim` must be consistent with `hidden_size / num_attention_heads`
- `num_key_value_heads` must divide `num_attention_heads`

### 5. Choose `tensors_per_shard`

Start with `tensors_per_shard=8`. It forces shard boundaries so the `index.json` / `weight_map` multi-shard paths get exercised — the production size-based split (≥1 GB) never shards test-sized tensors.

The helper asserts `len(shard_files) > component_count`. If a model's tiny components have too few parameter tensors to clear that bar, **lower the number until it passes** — do not drop the argument, that silently removes multi-shard coverage.

### 6. Name the slow twin — after checking it exists

The examples carry `Fast twin of tests/model_saving/test_model_saving_<model>.py`. Several models have weight definitions but **no** slow save test. Verify with `ls tests/model_saving/` and adapt the comment rather than pointing at a file that isn't there.

## Constraints that will bite you

- **Key the dict by `component.name`, not `model_attr`.** The helper does a bare `components[component.name]` lookup, so a missing or misnamed key is a raw `KeyError`, not a helpful message. Most common first-try failure. Every component in `get_components()` needs an entry.
- **Do not collapse the two seeds.** The harness randomizes saved components with `seed=0` and fresh ones with `seed=1`. That difference is exactly what makes the final equality assertion prove values came off disk. Unifying them leaves the test green and meaningless.
- **Do not edit `WeightDefinitionType`.** The helper types `weight_definition` as plain `type` on purpose — that `TYPE_CHECKING` union has drifted and omits several definitions. Widening a src type alias is out of scope; see the helper's comment.
- **Do not modify `src/`.** A tiny test is a test-only addition. If a model appears to genuinely require a src change to be testable, stop and report that rather than making it.

## Verify

```sh
uv run --no-sync python -m pytest tests/model_saving/test_tiny_model_saving_<model>.py -v
```

Then confirm nothing else broke with `just test-fast` (see `mflux-testing`).

The test should run in roughly a second. If it takes noticeably longer, or the network is touched, something is loading a real checkpoint — investigate rather than accepting it.

### Do you need to check that quantization actually happened?

**No.** Verified empirically: MLX raises loudly on a bad dimension rather than silently skipping —

```text
ValueError: [quantize] The last dimension of the matrix needs to be divisible
by the quantization group size 64. However the provided matrix has shape (100,100)
```

so a dimension that isn't a clean multiple of the group size fails the test outright. It cannot leave a green-but-weak twin.

Relatedly, when inspecting a quantized tree you'll see some `.weight` entries with no matching `.scales`. Confirmed benign: those are 1-D norm weights and `Conv2d` layers, which `nn.quantize` correctly leaves alone. Only `Linear`-style layers quantize. Don't chase these.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `KeyError: '<name>'` | `_tiny_components()` is missing a component, or keyed it by `model_attr` instead of `name` |
| `ValueError: [quantize] ... divisible by the quantization group size` | A dimension isn't a multiple of this model's `quantization_group_size` |
| `assert len(shard_files) > component_count` | Lower `tensors_per_shard` (step 5) |
| Values differ after reload | A real save/load bug — report it, do not weaken the assertion |
| Shape assertion inside a component's `__init__`/forward | An interlocking dimension constraint (step 4) |
