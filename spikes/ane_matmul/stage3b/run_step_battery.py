"""3b-2 driver: interleaved whole-step battery runs as alternating subprocesses.

BENCH.lock + watchdog for the whole battery, macmon sampling throughout. Arms
cannot share a process (the ANE patch replaces weights in place), so passes
alternate one subprocess per (arm, res) and pairing happens across passes.
First pass pays the Core ML package builds (cached under /tmp/ane3a/step_cache).
"""

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "stage3a"))
from s3a import WATCHDOG, BenchLock  # noqa: E402

PYTHON = str(HERE.parents[2] / ".venv" / "bin" / "python")
RESULTS = HERE / "results"
PASS_LIST = [1, 2]
ARMS = ["base", "ane16", "ane8"]
RES = [1024, 512]


def main():
    out_path = RESULTS / "step_battery.jsonl"
    with BenchLock(owner="3b2-step-battery", wait_s=7200):
        # Start the sampler only after the lock: an earlier start would truncate
        # the active battery's telemetry file.
        macmon = subprocess.Popen(
            ["macmon", "pipe", "-i", "1000"],
            stdout=open(RESULTS / "step_battery_macmon.jsonl", "w"),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        try:
            for p in PASS_LIST:
                for res in RES:
                    for arm in ARMS:
                        t0 = time.perf_counter()
                        cmd = [
                            WATCHDOG,
                            PYTHON,
                            str(HERE / "step_battery.py"),
                            "--arm",
                            arm,
                            "--res",
                            str(res),
                            "--pass",
                            str(p),
                        ]
                        try:
                            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3400)
                            out, err, timed_out = proc.stdout, proc.stderr, False
                        except subprocess.TimeoutExpired as exc:
                            out, err, timed_out = (exc.stdout or b"").decode(), "timeout after 3400 s", True
                        line = (out or "").strip().splitlines()
                        rec = json.loads(line[-1]) if line and line[-1].startswith("{") else None
                        if rec is None:
                            rec = dict(arm=arm, res=res, passno=p, error=(err or "")[-400:])
                        if timed_out:
                            rec["error"] = "TIMEOUT_3400S"
                        rec["wall_s"] = round(time.perf_counter() - t0, 1)
                        with open(out_path, "a") as f:
                            f.write(json.dumps(rec) + "\n")
                        print(json.dumps(rec), flush=True)
        finally:
            macmon.terminate()
            macmon.wait(timeout=10)


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    main()
