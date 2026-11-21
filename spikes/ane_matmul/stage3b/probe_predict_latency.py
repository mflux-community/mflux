"""One-shot probe: per-predict latency with a small live model set, clean process.

Loads a few production-built packages and times single predicts (rows 4608),
the same calls the step battery makes, to compare against the block sweep's
16-26 ms. Answers: is the flat ~280 ms per predict inherent to these packages
or does it appear only inside the big compiled step graph?
"""

import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ane_linear"))
import _ane_linear as ext  # noqa: E402

CACHE = Path("/tmp/ane3a/step_cache")


def one(p: Path, o: int, i: int, n=15):
    m = ext.AneModel(str(p), 6528, i, o)
    x = mx.zeros((4608, i), mx.float16)
    ts = []
    for _ in range(n):
        y = ext.ane_linear(x, m)
        t0 = time.perf_counter()
        mx.eval(y)
        ts.append((time.perf_counter() - t0) * 1e3)
    return round(statistics.median(ts), 1)


if __name__ == "__main__":
    picks = {}
    for p in sorted(CACHE.glob("*w8a16*.mlpackage")):
        parts = p.stem.split("_")
        kind = parts[1]
        od, idd = parts[3].split("x")
        picks.setdefault(kind, (p, int(od), int(idd.split("_")[0])))
    for kind, (p, o, i) in sorted(picks.items()):
        print(kind, o, i, one(p, o, i), flush=True)
    print("now four models interleaved (round robin per call):")
    ms = [(ext.AneModel(str(v[0]), 6528, v[2], v[1]), v[2], v[1], v[0].stem) for v in picks.values()]
    xs = {m[1]: mx.zeros((4608, m[1]), mx.float16) for m in ms}  # keyed by in-features
    for k in range(2):
        for m, i, o, tag in ms:
            y = ext.ane_linear(xs[i], m)
            t0 = time.perf_counter()
            mx.eval(y)
            print(" ", tag[:28], f"{(time.perf_counter() - t0) * 1e3:.1f} ms", flush=True)
