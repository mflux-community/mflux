"""Does wl_fused read inputs that MLX placed on the GPU, and is the CPU-stream choice real?

The primitive's eval_cpu takes inputs[i].data<void>() and hands it to Core ML. If MLX keeps those
arrays in Metal buffers, this only works because Apple Silicon's unified memory maps them
host-side. That may be load-bearing or incidental, so check three things directly:

  1. where the arrays actually live (default device, and after the primitive call);
  2. whether a GPU-resident int8 weight really is host-readable (read the first element from
     Python via numpy and compare against the source values);
  3. whether the result changes when inputs are moved to the CPU device on purpose.

If (3) differs from (1), the device placement is doing something subtle and the wiring has to pin
it rather than rely on it.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mlx.core as mx  # noqa: E402
from mlp_program import MlpSpec  # noqa: E402
from rowsplit import BridgeBackend, prepare_block_weights  # noqa: E402

D, H, M = 384, 1024, 256
rng = np.random.default_rng(0)
w1 = rng.normal(0, 0.05, (H, D)).astype(np.float32)
w3 = rng.normal(0, 0.05, (H, D)).astype(np.float32)
w2 = rng.normal(0, 0.05, (D, H)).astype(np.float32)
x = rng.normal(0, 1.0, (1, M, D)).astype(np.float16)

spec = MlpSpec(mpad=M, dim=D, hidden=H, scale_mode="output", range_fix=True)
bb = BridgeBackend(HERE / "cache")
bw = prepare_block_weights(spec, w1, w3, w2)
y_ptr, hm_ptr = bb.run(spec, bw, x)
y_ptr = np.array(y_ptr, dtype=np.float32).copy()
hm_ptr = np.array(hm_ptr, dtype=np.float32).copy()

model = bb._models[spec.name]
ext = bb.ext


def to_mx(device):
    # MLX 0.32.4 has no mx.to_device; placement is decided by the default device at creation
    prev = mx.default_device()
    mx.set_default_device(device)
    try:
        return [
            mx.array(x),
            mx.array(bw.w1),
            mx.array(bw.w3),
            mx.array(bw.w2),
            mx.array(bw.scales["s1"]),
            mx.array(bw.scales["s3"]),
            mx.array(bw.scales["s2"]),
        ]
    finally:
        mx.set_default_device(prev)


print(f"default device: {mx.default_device()}")
for label, dev in (("gpu (default)", mx.gpu), ("cpu", mx.cpu)):
    ins = to_mx(dev)
    try:
        y, hm = ext.wl_fused(*ins, model)
        mx.eval(y, hm)
        ya = np.array(y.astype(mx.float32))
        ha = np.array(hm.astype(mx.float32))
        d = float(np.abs(ya - y_ptr).max())
        print(
            f"  inputs on {label:>13}: maxdiff vs pointer {d:.6f} identical={d == 0.0} "
            f"h_max maxdiff {float(np.abs(ha - hm_ptr).max()):.6f}"
        )
    except Exception as exc:  # noqa: BLE001 -- reporting which placement fails is the result
        print(f"  inputs on {label:>13}: FAILED {type(exc).__name__}: {str(exc)[:180]}")

# The identical result above says the numbers come out right on the default device. This checks the
# bytes themselves, so GPU-device int8 is not being quietly re-quantised or re-cast on the way in.
gpu_codes = to_mx(mx.gpu)[1]
mx.eval(gpu_codes)
print(f"\nGPU-device int8 read back through numpy: identical={np.array_equal(np.array(gpu_codes), bw.w1)}")
