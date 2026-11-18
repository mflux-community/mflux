"""System sampler for the slow-state probe.

Usage: slow_state_sampler.py DURATION_S OUT.jsonl
One JSON line every ~2 s: timestamp, thermal limit, memory free %, swap used,
and the top CPU consumers. Runs alongside the ANE loop.
"""

import json
import re
import subprocess
import sys
import time
from pathlib import Path


def sh(cmd, timeout=10):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.stdout
    except (subprocess.TimeoutExpired, OSError):
        return ""


def sample():
    rec = {"ts": time.time()}
    therm = sh(["pmset", "-g", "therm"])
    m = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", therm)
    rec["cpu_speed_limit"] = int(m.group(1)) if m else None
    rec["thermal_pressure"] = (re.search(r"Thermal Pressure\s*=\s*(\S+)", therm) or [None, None])[1]
    mp = sh(["memory_pressure"], timeout=15)
    m = re.search(r"free percentage\s*:\s*(\d+)", mp) or re.search(r"(\d+)%", mp)
    rec["mem_free_pct"] = int(m.group(1)) if m else None
    sw = sh(["sysctl", "vm.swapusage"])
    m = re.search(r"used\s*=\s*([\d.]+)M", sw)
    rec["swap_used_mb"] = float(m.group(1)) if m else None
    ps = sh(["ps", "aux", "-r"])
    procs = []
    busy_total = 0.0
    for line in ps.splitlines()[1:]:
        f = line.split(None, 10)
        if len(f) < 11:
            continue
        try:
            pct = float(f[2])
        except ValueError:
            continue
        busy_total += pct
        if pct > 8 and len(procs) < 4:
            name = f[10].split("/")[-1].split(" ")[0][:40]
            procs.append([name, pct])
    rec["top_cpu"] = procs
    rec["cpu_busy_sum_pct"] = round(busy_total, 1)
    # Scheduling state of the benchmark process, to test the QoS-throttle hypothesis.
    bench = sh(["pgrep", "-x", "anebench"]).strip().split()
    if bench:
        info = sh(["ps", "-p", bench[0], "-o", "pri=", "ni=", "flags="]).split()
        rec["anebench_pri"] = int(info[0]) if info and info[0].isdigit() else None
        rec["anebench_nice"] = int(info[1]) if len(info) > 1 and info[1].lstrip("-").isdigit() else None
        rec["anebench_flags"] = info[2] if len(info) > 2 else None
    return rec


def main():
    dur = float(sys.argv[1])
    out = Path(sys.argv[2])
    end = time.time() + dur
    with open(out, "a") as f:
        while time.time() < end:
            t0 = time.time()
            rec = sample()
            f.write(json.dumps(rec) + "\n")
            f.flush()
            time.sleep(max(0.0, 2.0 - (time.time() - t0)))


if __name__ == "__main__":
    main()
