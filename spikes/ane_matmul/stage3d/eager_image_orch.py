import os, sys, json, time, subprocess
sys.path.insert(0, "/Users/ianscrivener/MFlux/mflux-ane-stage3/spikes/ane_matmul/stage3a")
from s3a import BenchLock
PY, WD, CHILD, OUT = sys.argv[1:5]
REPS = int(os.environ.get("REPS", "3"))
os.makedirs(OUT, exist_ok=True)
def swap(): return subprocess.run(["sysctl","-n","vm.swapusage"],capture_output=True,text=True).stdout.strip()
with BenchLock(owner="eager-image-%d" % os.getpid(), wait_s=3600):
    print("SWAP_BEFORE", swap(), flush=True)
    for rep in range(REPS):
        for arm in ("A", "C"):
            r = subprocess.run([WD, PY, CHILD, arm, str(rep), OUT], capture_output=True, text=True)
            line = [l for l in r.stdout.splitlines() if l.startswith("RESULT")]
            print(line[0] if line else "FAIL " + arm + " " + r.stderr[-1500:], flush=True)
    print("SWAP_AFTER", swap(), flush=True)
