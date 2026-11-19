"""RSS cost per loaded AneModel: load 32 production packages one by one."""

import os
import subprocess
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ane_linear"))
import _ane_linear as ext  # noqa: E402

CACHE = Path("/tmp/ane3a/step_cache")


def rss_mb():
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True, check=True)
    return int(out.stdout.strip()) / 1024.0  # KB -> MB


pkgs = sorted(CACHE.glob("b*_qkv_w8a16*.mlpackage"))[:16] + sorted(CACHE.glob("b*_w13_w8a16*.mlpackage"))[:16]
print(f"{len(pkgs)} packages, rss start {rss_mb():.0f} MB")
base = rss_mb()
last = base
ms = []
for p in pkgs:
    x = mx.zeros((4608, 3840 if "qkv" in p.name else 3840), mx.float16)
    m = ext.AneModel(str(p), 6528, 3840, 5760 if "qkv" in p.name else 10240)
    y = ext.ane_linear(x, m)
    mx.eval(y)
    r = rss_mb()
    ms.append((p.stem.split("_")[0] + ("_qkv" if "qkv" in p.name else "_w13"), round(r - last), round(r - base)))
    last = r
for tag, d, tot in ms:
    print(tag, f"+{d} MB", f"total+{tot} MB")
