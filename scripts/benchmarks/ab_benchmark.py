"""Compare two or more ways of running the same mflux command, with a fair and repeatable method.

Each arm is a command line that makes one image. The harness runs every arm in a fresh process, one image per
process, and alternates the order of the arms between runs, so that a slow start or a hot Mac does not favour one
arm. It reports, for each run:

    wall_s        the time of the whole process, with the model load
    gen_s         generation_time_seconds from the metadata file, when the command writes one
    first_step_s  the first denoising step. It holds the one-off costs: compile, lazy weight loads, lazy encoding
    steady_step_s the median of the other steps
    swap_grew_mb  how much swap grew during the run

Step times come from the tqdm progress bar that every mflux command prints on stderr.

Safety: the harness kills the whole process group of a run when free memory falls below --min-free-percent, or when
swap grows by more than --swap-limit-mb. A run whose swap grew by more than --discard-swap-mb is shown in the table
but left out of the medians. An exclusive lock file stops two benchmarks from running on the Mac at the same time.

Example:
    uv run python scripts/benchmarks/ab_benchmark.py \\
        --arm "stock=mflux-generate-z-image-turbo --model z-image-turbo -q 8 --steps 8 --seed 1 --output {image}" \\
        --arm "lowram=mflux-generate-z-image-turbo --model z-image-turbo -q 8 --steps 8 --seed 1 --low-ram --output {image}" \\
        --case "fox=--prompt 'a red fox in the snow'" --case "city=--prompt 'a neon street market in the rain'" \\
        --reps 3 --out-dir /tmp/ab

{image} is replaced by the output path. The text of a --case is added to the end of every arm. The first --arm is
the baseline: a ratio above 1 means that an arm is faster than the baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_LOCK = Path.home() / ".cache" / "mflux" / "benchmark.lock"
PROGRESS = re.compile(rb"(\d+)/(\d+) \[")
FREE_PERCENT = re.compile(r"free percentage:\s*(\d+)%")
SWAP_USED = re.compile(r"used\s*=\s*([\d.]+)([MGK])")


# --- Small pure helpers (tested in tests/benchmarks/test_ab_benchmark.py) -------------------------------------


def parse_swap_mb(text: str) -> float:
    """Parse the output of 'sysctl -n vm.swapusage' and return the used swap in MB."""
    found = SWAP_USED.search(text)
    if not found:
        raise ValueError(f"no swap figure in {text!r}")
    value, unit = float(found.group(1)), found.group(2)
    return value * {"K": 1 / 1024, "M": 1.0, "G": 1024.0}[unit]


def parse_free_percent(text: str) -> int | None:
    """Parse the output of memory_pressure and return the free percentage, or None."""
    found = FREE_PERCENT.search(text)
    return int(found.group(1)) if found else None


def swap_mb() -> float:
    return parse_swap_mb(subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout)


def free_percent() -> int | None:
    return parse_free_percent(subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout)


def run_order(arms: list[str], rep: int, case_index: int) -> list[str]:
    """Alternate the arm order from run to run. With two arms, the order flips every rep and every case."""
    shift = (rep + case_index) % len(arms)
    return arms[shift:] + arms[:shift]


def parse_progress(buffer: bytes) -> tuple[list[tuple[int, int]], bytes]:
    """Split a stderr buffer on CR and LF. Return the (n, total) of each complete segment and the unfinished rest."""
    parts = re.split(rb"[\r\n]", buffer)
    rest = parts.pop()
    steps = []
    for part in parts:
        found = PROGRESS.search(part)
        if found:
            steps.append((int(found.group(1)), int(found.group(2))))
    return steps, rest


def step_times(marks: dict[int, float]) -> tuple[float | None, float | None]:
    """From the time at which the bar first showed n steps done, return (first step, median of the other steps)."""
    if 0 not in marks or 1 not in marks:
        return None, None
    first = marks[1] - marks[0]
    later = [marks[n + 1] - marks[n] for n in sorted(marks) if n >= 1 and n + 1 in marks]
    return first, (statistics.median(later) if later else None)


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


@dataclass
class Row:
    arm: str
    case: str
    rep: int
    ok: bool
    wall_s: float
    gen_s: float | None = None
    first_step_s: float | None = None
    steady_step_s: float | None = None
    swap_grew_mb: float = 0.0
    killed: str | None = None
    image: str = ""
    note: str = ""


def summarise(rows: list[Row], arms: list[str], discard_swap_mb: float) -> dict:
    """Medians for each arm and the ratio against the first arm. Failed and swap-grown runs are left out."""
    usable = [r for r in rows if r.ok and r.swap_grew_mb <= discard_swap_mb]
    metrics = ("wall_s", "gen_s", "first_step_s", "steady_step_s")
    medians = {
        arm: {m: median([getattr(r, m) for r in usable if r.arm == arm and getattr(r, m) is not None]) for m in metrics}
        for arm in arms
    }
    base = medians[arms[0]]
    ratios = {}
    for arm in arms[1:]:
        ratios[arm] = {
            m: (base[m] / medians[arm][m] if base[m] and medians[arm][m] else None) for m in metrics
        }  # fmt: skip
    counts = {arm: sum(1 for r in usable if r.arm == arm) for arm in arms}
    return {"medians": medians, "ratios_vs_first_arm": ratios, "runs_used": counts}


# --- Lock -------------------------------------------------------------------------------------------------------


class BenchLock:
    """A cooperative lock file. It stops two benchmarks from timing on the same Mac at once. A dead holder is removed."""

    def __init__(self, path: Path, wait_s: float = 3600, poll_s: float = 5.0):
        self.path, self.wait_s, self.poll_s, self.fd = path, wait_s, poll_s, None

    def _holder_alive(self) -> bool:
        try:
            os.kill(int(json.loads(self.path.read_text())["pid"]), 0)
            return True
        except (OSError, ValueError, KeyError):
            return False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.time() + self.wait_s
        while True:
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self.fd, json.dumps({"pid": os.getpid(), "at": time.time()}).encode())
                return self
            except FileExistsError:
                if not self._holder_alive():
                    self.path.unlink(missing_ok=True)
                    continue
                if time.time() > deadline:
                    raise TimeoutError(f"{self.path} is held by another benchmark: {self.path.read_text()}")
                time.sleep(self.poll_s)

    def __exit__(self, *exc):
        if self.fd is not None:
            os.close(self.fd)
        self.path.unlink(missing_ok=True)
        return False


# --- One run ----------------------------------------------------------------------------------------------------


@dataclass
class Guard:
    min_free_percent: int
    swap_limit_mb: float
    swap_before: float
    poll_s: float = 1.0
    killed: str | None = field(default=None)

    def check(self) -> str | None:
        free = free_percent()
        if free is not None and free < self.min_free_percent:
            return f"free memory {free}% is below {self.min_free_percent}%"
        grown = swap_mb() - self.swap_before
        if grown > self.swap_limit_mb:
            return f"swap grew {grown:.0f} MB, over {self.swap_limit_mb:.0f} MB"
        return None


def run_once(command: list[str], image: Path, guard: Guard) -> tuple[Row, str]:
    """Run one command in its own process group. Return a partly filled Row and the tail of stderr."""
    marks: dict[int, float] = {}
    tail = bytearray()
    start = time.perf_counter()
    proc = subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True
    )  # fmt: skip

    def read_stderr() -> None:
        buffer = b""
        while chunk := os.read(proc.stderr.fileno(), 4096):
            now = time.perf_counter()
            tail.extend(chunk)
            del tail[:-4000]
            steps, buffer = parse_progress(buffer + chunk)
            for n, _ in steps:
                marks.setdefault(n, now)

    reader = threading.Thread(target=read_stderr, daemon=True)
    reader.start()
    killed = None
    while proc.poll() is None:
        time.sleep(guard.poll_s)
        reason = guard.check()
        if reason:
            killed = reason
            os.killpg(proc.pid, signal.SIGKILL)
            break
    proc.wait()
    reader.join(timeout=5)
    wall = time.perf_counter() - start
    first, steady = step_times(marks)
    row = Row(
        "", "", 0, proc.returncode == 0 and killed is None, wall, None, first, steady, killed=killed, image=str(image)
    )
    return row, tail.decode(errors="replace")


def read_generation_time(image: Path) -> float | None:
    sidecar = image.with_suffix(".metadata.json")
    try:
        return float(json.loads(sidecar.read_text()).get("generation_time_seconds"))
    except (OSError, ValueError, TypeError):
        return None


def psnr(a: Path, b: Path) -> float | None:
    try:
        import numpy as np
        from PIL import Image

        x, y = np.asarray(Image.open(a)).astype(np.float64), np.asarray(Image.open(b)).astype(np.float64)
    except (ImportError, OSError):
        return None
    if x.shape != y.shape:
        return None
    mse = float(((x - y) ** 2).mean())
    return float("inf") if mse == 0 else 10 * float(np.log10(255.0**2 / mse))


# --- Report -----------------------------------------------------------------------------------------------------


def fmt(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def report(rows: list[Row], arms: list[str], summary: dict, psnrs: dict, discard_swap_mb: float) -> str:
    lines = [
        "| arm | case | rep | wall s | gen s | first step s | steady step s | swap grew MB | note |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        note = r.killed or (
            "failed" if not r.ok else ("left out: swap grew" if r.swap_grew_mb > discard_swap_mb else "")
        )
        lines.append(
            f"| {r.arm} | {r.case} | {r.rep} | {fmt(r.wall_s, 1)} | {fmt(r.gen_s)} | {fmt(r.first_step_s)} | {fmt(r.steady_step_s, 3)} | {r.swap_grew_mb:.0f} | {note} |"
        )  # fmt: skip
    lines += ["", f"Medians. Runs that failed or grew swap by more than {discard_swap_mb:.0f} MB are left out.", ""]
    lines += ["| arm | runs | wall s | gen s | first step s | steady step s |", "|---|---|---|---|---|---|"]
    for arm in arms:
        m = summary["medians"][arm]
        lines.append(
            f"| {arm} | {summary['runs_used'][arm]} | {fmt(m['wall_s'], 1)} | {fmt(m['gen_s'])} | {fmt(m['first_step_s'])} | {fmt(m['steady_step_s'], 3)} |"
        )  # fmt: skip
    if len(arms) > 1:
        lines += [
            "",
            f"Ratio of the median of '{arms[0]}' to the median of each arm (above 1 means faster than '{arms[0]}'):",
            "",
        ]
        lines += ["| arm | wall | gen | first step | steady step |", "|---|---|---|---|---|"]
        for arm in arms[1:]:
            q = summary["ratios_vs_first_arm"][arm]
            lines.append(f"| {arm} | {fmt(q['wall_s'], 3)} | {fmt(q['gen_s'], 3)} | {fmt(q['first_step_s'], 3)} | {fmt(q['steady_step_s'], 3)} |")  # fmt: skip
    for key, value in psnrs.items():
        lines.append(f"\nImage difference {key}: {fmt(value, 1)} dB PSNR")
    return "\n".join(lines) + "\n"


# --- Main -------------------------------------------------------------------------------------------------------


def parse_named(items: list[str], what: str, allow_empty: bool = False) -> dict[str, str]:
    named = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep or not name or not (value or allow_empty):
            raise SystemExit(f"{what} must look like NAME=VALUE, got {item!r}")
        if name in named:
            raise SystemExit(f"{what} name {name!r} is used twice")
        named[name] = value
    return named


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--arm", action="append", required=True, help="NAME=COMMAND with {image}. The first is the baseline."
    )
    parser.add_argument(
        "--case", action="append", default=[], help="NAME=EXTRA ARGS, added to every arm. Default: one empty case."
    )
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-free-percent", type=int, default=3, help="kill a run when free memory is below this")
    parser.add_argument(
        "--swap-limit-mb", type=float, default=6000.0, help="kill a run when swap grows by more than this"
    )
    parser.add_argument(
        "--discard-swap-mb", type=float, default=64.0, help="leave a run out of the medians above this swap growth"
    )
    parser.add_argument("--lock-file", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--no-lock", action="store_true")
    parser.add_argument("--psnr", action="store_true", help="compare the rep 0 image of each arm with the baseline")
    args = parser.parse_args(argv)

    arm_commands = parse_named(args.arm, "--arm")
    arms = list(arm_commands)
    for name, command in arm_commands.items():
        if "{image}" not in command:
            raise SystemExit(f"--arm {name!r} needs {{image}} in its command")
    cases = parse_named(args.case, "--case", allow_empty=True) if args.case else {"run": ""}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[Row] = []
    lock = None if args.no_lock else BenchLock(args.lock_file)
    if lock:
        lock.__enter__()
    try:
        for rep in range(args.reps):
            for case_index, (case, extra) in enumerate(cases.items()):
                for arm in run_order(arms, rep, case_index):
                    image = args.out_dir / f"{arm}_{case}_r{rep}.png"
                    command = shlex.split(arm_commands[arm].replace("{image}", str(image))) + shlex.split(extra)
                    before = swap_mb()
                    row, tail = run_once(command, image, Guard(args.min_free_percent, args.swap_limit_mb, before))
                    row.arm, row.case, row.rep = arm, case, rep
                    row.swap_grew_mb = swap_mb() - before
                    row.gen_s = read_generation_time(image)
                    if not row.ok:
                        row.note = tail[-600:]
                        print(
                            f"[bench] {arm}/{case}/r{rep} FAILED: {row.killed or tail[-300:]}",
                            file=sys.stderr,
                            flush=True,
                        )
                    rows.append(row)
                    print(f"[bench] {json.dumps(row.__dict__)}", flush=True)
    finally:
        if lock:
            lock.__exit__(None, None, None)

    summary = summarise(rows, arms, args.discard_swap_mb)
    psnrs = {}
    if args.psnr:
        for case in cases:
            for arm in arms[1:]:
                psnrs[f"{case}, {arm} against {arms[0]}, rep 0"] = psnr(
                    args.out_dir / f"{arms[0]}_{case}_r0.png", args.out_dir / f"{arm}_{case}_r0.png"
                )
    (args.out_dir / "rows.json").write_text(json.dumps([r.__dict__ for r in rows], indent=1))
    text = report(rows, arms, summary, psnrs, args.discard_swap_mb)
    (args.out_dir / "report.md").write_text(text)
    print(text)
    return 0 if all(r.ok for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
