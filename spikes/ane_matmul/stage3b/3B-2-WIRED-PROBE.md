# 3B-2 wired-reduction probe

Date: 2026-11-19. One block, synthetic weights, fresh process per config,
MEMGATE on (free >= 25%, swap <= 2 GB), in-run GPU canary (~55 ms throughout,
so no contention). Purpose: test whether packaging changes can bring wired
memory to <= 250 MB per block at unchanged block speed. The 3b-2 step battery
found ~1000 MB wired per production block, and that is what kills --ane on a
36 GB machine.

## Configs

| Name | Models | Shapes | Precision |
|---|---|---|---|
| W4E4 | 4 separate | 4 enumerated tiers | W8A16 (production today) |
| W4E1 | 4 separate | one fixed shape 4608 | W8A16 |
| W4E1q8 | 4 separate | one fixed shape 4608 | W8A8 |
| W2E1 | 2 pair-fused | one fixed shape 4608 | W8A16 |
| W2E1q8 | 2 pair-fused | one fixed shape 4608 | W8A8 |

Pair A fuses qkv + w1w3 (they share the block input, so one conv). Pair B
fuses o + w2 (one model, two inputs, one concatenated output). Two models per
block is the floor: the SDPA and the silu sit between the call sites, and
porting them into Core ML is out of scope.

## Wired per model, median of three fresh passes

predict_mb is the wired added by one model plus its first predictions. It is
stable across passes, so it is the trustworthy column. The config totals mix
real wiring with system-wide drift, so the table shows both min and median.

| Model | Weight MB | W4E4 | W4E1 | W4E1q8 | W2E1 | W2E1q8 |
|---|---|---|---|---|---|---|
| qkv (5760) | 22 | 474 | 392 | 535 | - | - |
| o (1920) | 7 | 177 | 88 | 71 | - | - |
| w13 (10240) | 39 | 254 | 165 | 206 | - | - |
| w2 (1920) | 20 | 882 | 535 | 780 | - | - |
| pairA (16000) | 61 | - | - | - | 487 | 818 |
| pairB (3840) | 27 | - | - | - | 587 | 817 |

| Config | min MB | median MB | max MB |
|---|---|---|---|
| W4E4 | 999 | 1616 | 1715 |
| W4E1 | 649 | 956 | 1265 |
| W4E1q8 | 990 | 1210 | 1251 |
| W2E1 | 599 | 1054 | 1108 |
| W2E1q8 | 1119 | 1688 | 1900 |

## ANE work per block (latency)

The GPU suffix, SDPA and silu are identical across configs, so the sum of the
ANE calls isolates what the probe changes. Interleaved rounds in one process,
canary 55.4 ms.

| Config | ANE work ms |
|---|---|
| W4E4 | 73.6 / 73.9 |
| W4E1 | 74.9 / 73.9 |
| W4E1q8 | 66.9 / 67.1 |
| W2E1 | 69.6 / 69.6 |
| W2E1q8 | 62.5 / 62.5 |

## Answers to the four probe questions

1. Fixed shape against enumerated: yes, it helps. Per site, the same model
   wires 33 to 55 percent less with one fixed shape (w2 535 against 882, o 88
   against 177). Enumerated tiers keep a scratch pool per shape. Speed is
   unchanged (74.9 against 73.6).
2. W8A16 against W8A8 wired: W8A8 costs wired. Same weight bytes, +40 to 50
   percent wired (pairA 818 against 487). The dequantise and token-scale
   kernels add wired buffers. The ANE expands nothing new: both precisions
   carry the int8 weights. The extra wiring tracks the fp16 intermediates the
   W8A8 plan adds.
3. Two fused models against four separate: modest win. Pair-fusing halves
   instances: fixed W2E1 wires 1074 MB summed against 1180 MB, about 9 percent
   less. It also gives slightly less ANE work (69.6 against 74.9 ms). Instance
   overhead is not the main cost.
4. Best combination: W2E1 (pair-fused, fixed shape, W8A16): 599 MB min, 1054
   MB median, 69.6 ms. Against production W4E4 that is about 35 percent less
   wired and about 5 percent faster.

## Verdict: the gate fails

Gate: <= 250 MB wired per block at block work <= ~120 ms. Speed passes. Wired
fails at every config. The best observed single pass is 599 MB, and the
reliable per-model sum for the best config is about 1.05 GB. Weight-proportional
buffers and conv intermediates set the floor: pairA carries 61 MB of
int8 weights and wires 487 MB. No packaging change reaches 250 MB, because
the per-block dataflow forbids fewer than two models, and the per-model floor
is already above the gate.

Extrapolation: 32 blocks x 0.6-1.05 GB still means 19-34 GB wired on 36 GB,
even with every reduction applied. Model weights and conv intermediates set
the floor, not instance count or enum. Recommendation: freeze the ANE lane,
unless you plan a 64/128 GB machine trial. If one happens, carry these three changes
into it: fixed shapes per resolution (one package per tier set, not four
enumerated), pair-fused blocks (2 models), and W8A16 unless latency matters
more than RAM (W8A8 buys ~10 percent speed for ~50 percent more wired).

What the probe did not test: three or four fused outputs inside one model with
in-graph joins between call sites - that needs the SDPA ported to Core ML, and
it changes the architecture, not the packaging.
