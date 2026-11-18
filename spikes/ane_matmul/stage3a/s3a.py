"""Shared helpers for the Stage 3a infrastructure tests.

All tests use synthetic weights at production shapes. Timing runs hold the
BENCH.lock; correctness runs do not. The lock lives in the main checkout so
that every worktree on this machine shares one lock.
"""

import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPIKE = HERE.parent
REPO = SPIKE.parent.parent
PYTHON = str(REPO / ".venv" / "python")
WATCHDOG = str(SPIKE / "watchdog.sh")
ANEBENCH = str(SPIKE / "bench" / "anebench")
ANEPLAN = str(SPIKE / "bench" / "aneplan")
ANERUN = str(SPIKE / "bench" / "anerun")
RESULTS = HERE / "results"
MODELS = Path("/tmp/ane3a/models")
LOCK = Path("/Users/ianscrivener/MFlux/mflux-upstream/spikes/BENCH.lock")
OWNER = f"stage3a-{os.getpid()}"

T_FIXED = 4224  # production token count for 1024x1024


class BenchLock:
    """Cooperative lock that serialises timed runs between agents on this Mac."""

    def __init__(self, owner: str = OWNER, wait_s: int = 3600, poll_s: float = 5.0):
        self.path = LOCK
        self.owner = owner
        self.wait_s = wait_s
        self.poll_s = poll_s
        self.fd = None

    def _holder_alive(self) -> bool:
        try:
            info = json.loads(self.path.read_text())
            os.kill(int(info["pid"]), 0)
            return True
        except (OSError, ValueError, KeyError, FileNotFoundError):
            return False

    def __enter__(self):
        deadline = time.time() + self.wait_s
        while True:
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(
                    self.fd,
                    json.dumps({"owner": self.owner, "pid": os.getpid(), "at": time.time()}).encode(),
                )
                return self
            except FileExistsError:
                if not self._holder_alive():
                    print(
                        f"[lock] removing stale lock: {self.path.read_text() if self.path.exists() else '?'}",
                        flush=True,
                    )
                    try:
                        self.path.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                if time.time() > deadline:
                    raise TimeoutError(f"BENCH.lock held for over {self.wait_s}s: {self.path.read_text()}")
                time.sleep(self.poll_s)

    def __exit__(self, *exc):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        return False


def run(cmd, timeout=1800, **kw):
    """Run a command, raise on failure, return stdout."""
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout, **kw)
    if p.returncode != 0:
        raise RuntimeError(
            f"command failed ({p.returncode}): {' '.join(str(c) for c in cmd)}\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}"
        )
    return p.stdout


def anebench(
    model, shape=None, tokens=None, function=None, layout="conv", iters=50, units="ane", window=None, int8in=False
):
    """Benchmark one model/function; returns the parsed JSON result."""
    cmd = [WATCHDOG, ANEBENCH, model, "--units", units, "--iters", iters, "--layout", layout]
    if shape is not None:
        cmd += ["--shape", *shape]
    if tokens is not None:
        cmd += ["--tokens", tokens]
    if function is not None:
        cmd += ["--function", function]
    if int8in:
        cmd += ["--int8in"]
    if window is not None:
        cmd += ["--window", *window]
    out = run(cmd)
    return json.loads(out.strip().splitlines()[-1])


def aneplan(model, function=None):
    """Return {op_name: device} counts for one model/function."""
    cmd = [ANEPLAN, model]
    if function is not None:
        cmd += ["--function", function]
    out = run(cmd, timeout=600)
    counts = {}
    for line in out.splitlines():
        op, _, dev = line.partition("\t")
        key = f"{op}:{dev}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def emit(test: str, record: dict):
    """Append one JSON result line to results/<test>.jsonl and print it."""
    RESULTS.mkdir(exist_ok=True)
    line = json.dumps({"test": test, **record})
    with open(RESULTS / f"{test}.jsonl", "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def package_bytes(path) -> int:
    path = Path(path)
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def gen_fixed(outdir, T, O, I, variant="c1"):
    """Generate a fixed-shape model with gen_models.py; return the .mlpackage path."""
    outdir = Path(outdir)
    path = outdir / f"{variant}_{T}x{O}x{I}.mlpackage"
    if not path.exists():
        run([PYTHON, SPIKE / "gen_models.py", outdir, T, O, I, "--variant", variant], timeout=3600)
    if not path.exists():
        raise RuntimeError(f"model generation produced no file: {path}")
    return path
