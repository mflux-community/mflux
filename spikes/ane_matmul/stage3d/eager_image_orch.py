import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage3a"))
from s3a import BenchLock

PY, WD, CHILD, OUT = sys.argv[1:5]
REPS = int(os.environ.get("REPS", "3"))
os.makedirs(OUT, exist_ok=True)


def swap():
    return subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.strip()


with BenchLock(owner="eager-image-%d" % os.getpid(), wait_s=3600):
    print("SWAP_BEFORE", swap(), flush=True)
    for rep in range(REPS):
        arms = os.environ.get("ARMS", "A,C").split(",")
        if rep % 2 == 1:  # alternate the order each repetition so no arm always runs first
            arms = arms[::-1]
        print("ORDER", rep, ",".join(arms), flush=True)
        for arm in arms:
            r = subprocess.run([WD, PY, CHILD, arm, str(rep), OUT], capture_output=True, text=True)
            line = [l for l in r.stdout.splitlines() if l.startswith("RESULT")]
            print(line[0] if line else "FAIL " + arm + " " + r.stderr[-1500:], flush=True)
    print("SWAP_AFTER", swap(), flush=True)
