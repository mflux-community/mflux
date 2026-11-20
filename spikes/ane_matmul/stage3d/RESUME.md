# Stage 3d — RESUME

## CLOSED 2026-11-20: no shippable gain. Do not run Steps 2b or 3.

**Verdict: the row split does not overlap through an MLX primitive on MLX 0.32.4.** Shippable path
0.97–1.07× against the GPU across every form and env setting; threaded path 1.17–1.20× but illegal
under `mx.compile`. Full write-up: report **§9** (`3D-ROWSPLIT-MLP.md`).

The prize is real and quantified: `ff` at 4224 rows 86.22 ms vs 44.02 ms at the 2176 GPU rows, staging
+ epilogue 0.89 ms, ANE predict ~52 ms ⇒ ~1.25× block-level if the ANE starts at ~0 ms. It starts at
**96.5 ms** (baseline) or **29.7 ms** (best setting, `MLX_MAX_OPS_PER_BUFFER=1` + `MLX_METAL_FAST_SYNCH=1`),
and in-task `wait()` is 0.0 ms — the data is ready, the *task start* is late. Five hypotheses were
tested and falsified: dedicated stream, fork ordering, CPU-stream staging into a reused host surface,
command-buffer granularity, and per-dispatch `notify_new_task`. Nothing moved the delay below 29.7 ms.

Accuracy, corrected: A2 (`real_primitive_gate.py`) passes primitive fidelity on all 34 real blocks ×
3 gains (cosine gap 1.4e-6 vs the pointer API, 0 non-finite, 1 thread, 0 copybacks). But Step 1's
~0.99998 was an **fp32** figure — at the bf16 resolution generation uses the split is min **0.99969**
(`layers.27`) and only **60/102** rows clear 0.9999. The pointer path scores the same.

Keep if any ANE primitive is revisited: the fence (`wait()` inside the task), `add_temporary` input
lifetimes, QoS `USER_INTERACTIVE`, the reused host staging surface, `wl_timeline`/`wl_mark`, and the
A2 gate. Do not advertise `--ane-mlp` as a speedup.

Only untried lever: compile in two halves per block with `mx.async_eval` between them (mailbox inbox
option 2). It targets the same commit-boundary mechanism `ops=1` already probed, which cut the delay
3× without buying speed — so it is not worth more budget without a new idea.

Everything below is **history**: the state and instructions as they stood before Plan A. The Step 2b/3
image commands are superseded by the closure above and were never run.

---

This file exists because a reboot ends the session.
Take every number and hash from tool output, never from memory.

## Where things stand (historical)

- Worktree: `/Users/ianscrivener/MFlux/mflux-ane-stage3`
- Branch: `experiment/zit-on-ane-stage3d`
- HEAD: `git log --oneline -1`. Plan A ended at `f7c7e414`; before it, `3f549884` (Step 1, accepted)
  and `52b4b0dd` (Step 2 wiring + smoke test).
- Nothing has been pushed. **Ask Scriv before any `git push`. First question to ask: "Do we run
  CodeRabbit locally first?"**
- Report: `spikes/ane_matmul/stage3d/3D-ROWSPLIT-MLP.md` (§9 is the closing verdict)
- PRDs: `spikes/ane_matmul/PRD-stage3d.md`, `PRD-stage3d-finish.md`
- Reports go to `spikes/ane_matmul/MAILBOX/outbox.md`, not chat (inbox instruction).

| commit | what |
| --- | --- |
| `f9231fc5` | spike imported |
| `4599aaa9` | fused MLP wired into the bridge |
| `f4474b6e` | memoised quantisation (see the leak caveat below) |
| `ce1cb694` | 128x gain fix, scale-aware gate, `BenchLock` |
| `cbf0281f` | report + probes + result JSONs |
| `f4010249` `16dc9cd0` `f922049a` | probes A, B, C (synthetic data) |
| `53f1aa26` | real-weight gate: NO-GO, ANE lost to q8 in 0/90 rows |
| `7c43f8c7` | range + accuracy mirror over 34 real blocks |
| `3f549884` | **Step 1: a+b range fix on the ANE — accepted by Scriv** |
| `52b4b0dd` | Step 2 wiring + weight-free smoke test |
| `b6e645d5` | **Step 2a: T4 re-timed with the range fix → 1.186× at 34 blocks** |
| `22bb44ed` | Step 2b prep: record the real token count, since share is not fixed |
| `856b6efd` | swap delta split into load phase vs timed window |
| `fb74544f` | **`wl_fused` as an MLX primitive — works under `mx.compile`** |
| `3df900ee` | primitive reads inputs correctly on either MLX device |
| `470014e6` | **option (a) complete — row split wired, eager == compiled** |

## Read this first: the ANE arm was structurally broken, and is now fixed

The first real generation run with `--ane-mlp` did not fail from memory. It failed at the first step:

```
ValueError: [eval] Attempting to eval an array during function transformations like compile or vmap is not allowed.
```

mflux wraps the denoise step in `mx.compile` (`src/mflux/models/z_image/variants/z_image.py:282`), and
the spike's `RowSplitFFN` synchronised to the host with `np.array()` to feed Core ML. The standalone
benches never caught it because they call the chain eagerly. So every T2/T3/T4 speedup number was
measured on a code path that **cannot run inside generation**.

Fixed by option (a): `wl_fused` is now an `mx::Primitive` (`WLFused`) with two fp16 outputs
(`y_rows`, `h_max_out`), and `PrimitiveSplitFFN`/`patch_transformer_compilable` use it with no host
sync. Verified on scaled dims:

- primitive vs pointer API: **bit-identical** on both outputs (maxdiff 0.000000)
- inside `mx.compile`: works; eager and compiled give the **same** cos 0.9999350 / relerr 0.01141
- input device placement is irrelevant: identical on `mx.gpu` and `mx.cpu` (MLX 0.32.4 has no
  `mx.to_device`; placement is set by `mx.set_default_device`)
- the rowmax restore is **explicitly fp32**. In fp16 it reintroduces the overflow this exists to
  remove (real `h_max` 5064; restored product 5.06e5 vs a 65504 ceiling).

`patch_model(mode=)` selects the form: `"compiled"` (default, for generation) or `"bench"` (the
threading path the standalone benches need in order to time phases).

## Still open before any timing claim

1. **T2/T3/T4 and the 1.265×/1.186× numbers came from the bench path, not the compiled path.**
   Whether MLX's own scheduling overlaps the ANE call as well as my worker thread did is unmeasured.
   The honest expectation is that the compiled arm may be slower. Re-measure in Step 2b before
   quoting a speedup for generation.
2. **Swap is not clean at 1024², for any arm.** Measured: bf16 load pushed +5612 MB and the window
   pulled back −5426 MB; q8 +4795 / −4717. Negative window deltas mean page-ins happened *inside*
   the timed region, so those step times are contaminated. `--low-ram` helps (8.71 vs 10.34 s/step,
   and ~0.74 GB paged instead of ~2.86 GB) but cannot help the load, which is when the paging occurs.
   **Reboot before Step 2b/3**, and expect the load phase to page regardless — say so rather than
   presenting a contaminated number as a result.
3. `free-memory` only frees ~1.00 GB against ~26.8 GB already available; it is not the lever.

## STOP: the timing case is dead too, so Steps 2b and 3 should not be run

Measured on `3a92a975` after this reboot, same arguments, differing only in `--patch-mode` (report §7,
commit `8533d9b5`):

| path | rung 8 | rung 16 | rung 34 | swap | peak RSS |
| --- | --- | --- | --- | --- | --- |
| threaded (cannot ship) | 1.1116× | 1.1676× | 1.0882× | 0 → 0 MB | 5.96 GB |
| primitive (shippable) | 1.0336× | **0.9191×** | **0.9692×** | 0 → 1287 MB | 9.08 GB |

The 1.265× and 1.186× figures came from a host-syncing Python worker thread that `mx.compile` forbids.
Given MLX's own scheduler the split is slower than the GPU at 16 and 34 blocks, and macmon recorded ANE
power at median 0.00 W (max 2.55 W) during that run against 4.03 W in a back-to-back burst. The overlap
was the thread's, not the ANE's. Rungs 8 and 16 had `discarded=0` with swap flat, so this is not
contamination. Accuracy already failed independently (0 of 90 rows vs q8), so a bf16/q8/ANE image
comparison would be an eyeball check on a path that loses on both axes. **Recommendation: stop the
row-split lane.**

If Scriv wants rung 34 firmed up, the one useful re-run is `bench.py t4 --range-fix --patch-mode
primitive` on a freshly rebooted machine (swap 0.00M first). Note the primitive run's swap growth and
that its per-call stats are empty by design — timing the phases requires the host sync.

Known defect to fix before further primitive work: `PrimitiveSplitFFN` keeps MLX copies of the codes
while `prepare_block_weights` keeps the numpy ones, so the primitive path carries the weights twice
(peak RSS 9.08 GB vs 5.96 GB threaded).

## State after Step 2a

T4 with `--range-fix` (run post-reboot with swap 0.00M at start, `weight_surfaces=per-block`, on
commit `653077d9`):

| rung | gpu_only ms | ANE ms | ANE p90 | speedup | discarded |
| --- | --- | --- | --- | --- | --- |
| 8 | 1689.4 | 1456.4 | 2192.8 | 1.160× | 0 |
| 16 | 3050.4 | 2740.0 | 2967.0 | 1.113× | 0 |
| 34 | 6009.3 | 5068.8 | 5643.9 | **1.186×** | 1 |

Per call at rung 34: `prep 80.06, wait 54.10, epilogue 1.54`. Swap 0 → 203.6 MB, wired +3.08 GB,
peak RSS 5.47 GB. **The range fix costs about 10%**: wait rose 46.18 → 54.10 ms because
`abs`/`reduce_max`/`real_div` on a 2048×10240 `h` is real ANE work the saturating program skipped.
So **1.186× replaces 1.265×** as the honest chained-step number, and 1.265× must never be quoted as
the achievable gain — it measured a program that clamped at the fp16 ceiling.

**Next is Step 2b.** It loads the real ~20 GB model and generates, so it needs Scriv's go-ahead
first. Share caveat, now handled in code: `--ane-seq` sizes the program, but Z-Image's main layers
run on the unified sequence, so the share actually used is `Mpad / S` and moves with prompt token
count. Read `ane_observed.seq_seen` and `actual_share` from the ANE-RESULT line rather than assuming
0.50.

## What Step 1 established (do not redo it)

The old failures were **fp16 saturation**, not the ANE misbehaving. The fix, `MlpSpec(range_fix=True)`:

- **(a)** weights widened by `dequantize(scale = 1/sqrt(K_in))`; the real per-row scale moves to the
  matmul output as `s*sqrt(K)`; `x` is no longer pre-shrunk.
- **(b)** `h` divided by its own per-row max inside the program; that max leaves as a **second
  output** `h_max_out` and is restored in the **fp32 GPU epilogue**. Restoring inside the program is
  what overflows. Floor is `6.1e-5` (fp16's smallest normal) — `1e-8` underflows to zero → `0/0`.

Measured over all 34 real FFN-bearing blocks (30 `layers` + 2 `noise_refiner` + 2 `context_refiner`)
× g{1,2,4} = 102 rows: range gate **102/102** (worst magnitude 5,064 vs limit 32,752), `copybacks=0`,
nonfinite 0, **ANE − mirror cosine delta median −7.4e-8, all 102 within 1e-5**. Worst block
`layers.27`: 0.9996898 ANE vs 0.9996900 mirror — identical, so the residual is per-row int8 and no
further ANE work moves it. Scriv **accepted per-row int8 at ~0.99998**; the q8 comparison gate, `g64h`
and smoothing are **dropped — do not revisit them**.

## Config to use

- `--ane-share 0.50` (best across T1–T4; 0.25 is neutral, 0.40 noisier)
- `--ane-seq 4224` → `Mpad = 2048` at 1024×1024
- `scale_mode="output"`, `range_fix=True`, weight dtype int8
- **g is not a knob in generation.** g{1,2,4} were a stand-in for `1 + scale_mlp` in the mirror.
  Real runs use real modulation, which is exactly what Steps 2–3 measure.

## Pre-flight checks — run all four before any timed command

```bash
cd /Users/ianscrivener/MFlux/mflux-ane-stage3
sysctl -n vm.swapusage              # want used = 0.00M; any nonzero swap invalidates timings
ls /Users/ianscrivener/MFlux/mflux-upstream/spikes/BENCH.lock 2>/dev/null || echo "lock free"
.venv/bin/python -m pytest -q spikes/ane_matmul/stage3d/   # want 13 passed
which macmon && macmon --version    # for the ANE power check below
```

If swap is **not** ~0, stop and tell Scriv — do not run timings. During ANE runs check
`macmon pipe -s 0 -i 500` for `ane_power` ≈ 9 W; ~4 W means the ANE is not saturated and the
timing is not a best case.

## Step 2 — time the arms

`bench.py` acquires `BENCH.lock` itself on timed tests. **Do not also hold it in the launching
shell** — that self-deadlocked a run for 11 minutes at 0.13 s CPU.

```bash
# 2a. chained 34-block step, bf16 vs ANE, range fix on. Run under watchdog.sh.
zsh spikes/ane_matmul/watchdog.sh .venv/bin/python -u spikes/ane_matmul/stage3d/bench.py \
    t4 --backend bridge --scale-mode output --range-fix --shares 0.5 --repeats 5 \
    >| spikes/ane_matmul/stage3d/results/t4_rf.log 2>&1

# 2b. one full image per arm, 1024^2, 8 steps. generate_ane.py does NOT self-lock:
#     hold BENCH.lock in the shell around these three commands.
L=/Users/ianscrivener/MFlux/mflux-upstream/spikes/BENCH.lock
{ set -o noclobber; printf '{"owner":"qwen-3d-step2","pid":%s,"at":%s}\n' "$$" "$(date +%s)" > "$L"; }
trap 'rm -f "$L"' EXIT
D=spikes/ane_matmul/stage3d/images; mkdir -p $D
P="a portrait of an elderly luthier in his workshop, warm rim light, 85mm"
for arm in bf16 q8 ane; do
  case $arm in
    bf16) EXTRA="" ;;
    q8)   EXTRA="--quantize 8" ;;
    ane)  EXTRA="--ane-mlp --ane-share 0.50 --ane-seq 4224" ;;
  esac
  zsh spikes/ane_matmul/watchdog.sh .venv/bin/python -u spikes/ane_matmul/stage3d/generate_ane.py \
      --prompt "$P" --width 1024 --height 1024 --steps 8 --seed 0 $EXTRA \
      --arm-name $arm --output $D/step2_${arm}.png >| spikes/ane_matmul/stage3d/results/step2_${arm}.log 2>&1
  grep -H ANE-RESULT spikes/ane_matmul/stage3d/results/step2_${arm}.log
done
```

Record for each arm: `image_s`, `per_step_s`, `swap_mb_before/after`, `wired_gb_before/after`,
`patch.spec`, `copybacks`. **Swap delta must be 0** — any growth invalidates that row; discard it and
say so. Expect wired to grow by several GB: the memory fix (one shared weight copy) is deliberately
out of scope here and **not** validated — the shared-surface experiment made wired worse (+6.09 GB vs
+4.86 GB) because the per-block int8 copies were never removed. Report the numbers, do not fix.

## Step 3 — images for the eyeball check

4 prompts × 1 seed × 3 arms, then a labelled grid per prompt.

Prompts (write these into the report too):

1. portrait — "a portrait of an elderly luthier in his workshop, warm rim light, 85mm, shallow depth of field"
2. text on a sign — "a neon sign reading OPEN LATE above a rain-slicked diner, legible lettering, night"
3. detailed landscape — "an alpine lake at dawn, mirrored peaks, wildflower meadow foreground, f/11 detail"
4. busy indoor — "a crowded market hall, vendors, hanging lamps, steam, dozens of distinct faces"

```bash
D=spikes/ane_matmul/stage3d/images; SEED=0
# reuse the same $L / trap block from Step 2 (one lock for the whole sweep)
i=1
for P in "<prompt 1>" "<prompt 2>" "<prompt 3>" "<prompt 4>"; do
  for arm in bf16 q8 ane; do
    case $arm in
      bf16) EXTRA="" ;; q8) EXTRA="--quantize 8" ;;
      ane) EXTRA="--ane-mlp --ane-share 0.50 --ane-seq 4224" ;;
    esac
    zsh spikes/ane_matmul/watchdog.sh .venv/bin/python -u spikes/ane_matmul/stage3d/generate_ane.py \
        --prompt "$P" --width 1024 --height 1024 --steps 8 --seed $SEED $EXTRA \
        --arm-name $arm --output $D/${i}_${arm}.png >| spikes/ane_matmul/stage3d/results/${i}_${arm}.log 2>&1
  done
  i=$((i+1))
done
```

Grids and metrics: build `<i>_grid.png` with three labelled panels (bf16 | q8 | ANE). There is no
grid helper committed yet — write one in `stage3d/` using PIL (`ImageDraw.text` for the arm labels),
or ask Scriv first if he would rather skip grids.

```bash
for i in 1 2 3 4; do
  .venv/bin/python -u spikes/ane_matmul/stage3d/image_metrics.py \
      --ref $D/${i}_bf16.png --arms $D/${i}_q8.png $D/${i}_ane.png \
      --out spikes/ane_matmul/stage3d/results/metrics_${i}.json
done
```

`image_metrics.py` prints PSNR and leaves LPIPS null: **`lpips` is not installed** and installing it
pulls VGG weights over the network → **ask Scriv before installing anything.** The script is
self-tested (PSNR 30.19 on a noisy pair, `inf` + `identical=True` on an identical file, non-zero exit
with "shape mismatch" on different sizes, LPIPS null path), so only real images are unverified.

## Then stop

Report image paths + the metrics table, then **wait for Scriv's eyeball verdict.** Do not open a PR,
do not push, do not start the single-weight-copy memory fix — that is the next piece of work and
needs its own go-ahead.

## Gotchas that have already cost time

- **`MLComputePlan.load_from_path` on a `.mlpackage` aborts the process** (uncatchable C++ throw).
  Pass the compiled `.mlmodelc`.
- Run python with `-u`; buffered stdout is lost when a process dies.
- `set -o noclobber` (needed for the lock) also blocks `>` from truncating an existing log — use
  `>|`. This bit me twice.
- `bench.py` self-locks; `generate_ane.py` does not. Never double-acquire.
- A cosine above 1 means float32 accumulation in the denominator — compute norms in float64.
- Finiteness is **not** evidence of range safety: the ANE clamps at 65504 instead of producing inf.
  Check magnitudes.
- `mlx.nn.quantize()` returns None; mflux q8 is `module.to_quantized(group_size=64, bits=8)`.
- Pre-commit `ruff format` rewrites files mid-commit → the commit aborts with "files were modified".
  Re-`git add`, re-run tests, commit again.
- `patch_transformer` swaps `layer.feed_forward` (attribute swap). bench.py's T2/T3/T4 used
  `RowSplitBlock`, which re-implements the block forward. Same math, two code paths — do not present
  the generation-arm numbers and the old T4 numbers as one series without saying which path produced
  each.
- Known leak, fixed in `3f549884`: `prepare_block_weights` used to retain source arrays to make
  `id()` keys safe, pinning ~590 MB per block. It stores a verified content
  fingerprint instead. The old T2/T3/T4 runs used the leaky version, but bench passes MLX params the
  blocks own anyway, so their wired numbers still stand.
- Each patched block spawns one worker thread (30 threads at full depth). Fine for a one-arm process;
  run arms in separate processes as the loops above do.
- `RowSplitFFN` asserts the real sequence is longer than `Mpad`. If a prompt changes the token count,
  `--ane-seq` must match or the assert fires — that is intentional, do not relax it.
