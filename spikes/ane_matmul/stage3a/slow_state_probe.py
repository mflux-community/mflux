"""Slow-state probe: loop the w1w3 ANE call for ~10 minutes and correlate with system signals.

Under BENCH.lock + watchdog. Starts the system sampler and `macmon pipe` (ANE
power, GPU freq, CPU activity — sudoless), runs anebench in window mode with
per-sample logging, then analyses: slow-state frequency, run lengths, and which
sampled signal predicts slow samples.

Usage: slow_state_probe.py [run_tag]   (default tag: run2; the untagged files
from the first probe stay untouched.)
Summary lines go to results/slow_state.jsonl with a "run" field. Raw logs:
results/slow_state_percall[_tag].jsonl, results/slow_state_sys[_tag].jsonl,
results/slow_state_macmon[_tag].jsonl.
"""

import json
import os
import subprocess
import sys
import threading
import time

import s3a
from s3a import HERE, MODELS, BenchLock, anebench, emit

DURATION_S = 600
MODEL = MODELS / "compiled" / "sa_w1w3.mlmodelc"  # hash recorded in recompile_w1w3.jsonl
SHAPE = (4224, 20480, 3840)
SLOW_MS = 70.0  # fast state ~53 ms, slow state ~90 ms
TAG = sys.argv[1] if len(sys.argv) > 1 else "run2"
SUF = "" if TAG in ("", "run1") else f"_{TAG}"
PERCALL = HERE / "results" / f"slow_state_percall{SUF}.jsonl"
SYSLOG = HERE / "results" / f"slow_state_sys{SUF}.jsonl"
MACLOG = HERE / "results" / f"slow_state_macmon{SUF}.jsonl"


def check_powermetrics():
    p = subprocess.run(
        ["sudo", "-n", "powermetrics", "--samplers", "ane_power", "-n", "1", "-i", "100"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if p.returncode == 0:
        return "available (passwordless sudo)"
    return f"unavailable: sudo -n powermetrics returned {p.returncode} ({p.stderr.strip()[:120]})"


def start_macmon_logger(stop_event):
    """Run `macmon pipe -i 1000` and stamp each JSON line with wall time."""
    proc = subprocess.Popen(
        ["macmon", "pipe", "-i", "1000"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
    )
    fh = open(MACLOG, "a")

    def pump():
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec["ts"] = time.time()
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if stop_event.is_set():
                break
        fh.close()

    t = threading.Thread(target=pump, daemon=True)
    t.start()
    return proc, t


def main():
    PERCALL.unlink(missing_ok=True)
    SYSLOG.unlink(missing_ok=True)
    MACLOG.unlink(missing_ok=True)
    pm = check_powermetrics()
    emit(
        "slow_state",
        dict(phase="setup", run=TAG, powermetrics=pm, model=str(MODEL), duration_s=DURATION_S, macmon="pipe -i 1000"),
    )

    with BenchLock(owner=f"stage3a-slowstate-{os.getpid()}"):
        stop = threading.Event()
        macmon_proc, macmon_thread = start_macmon_logger(stop)
        sampler = subprocess.Popen(
            [s3a.PYTHON, str(HERE / "slow_state_sampler.py"), str(DURATION_S + 40), str(SYSLOG)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            start = time.time() + 8
            end = start + DURATION_S
            r = anebench(MODEL, shape=SHAPE, units="ane", window=(start, end), percall=str(PERCALL), repeat=1)
            emit(
                "slow_state",
                dict(
                    phase="loop_summary",
                    run=TAG,
                    median_ms=r["median_ms"],
                    p90_ms=r["p90_ms"],
                    tflops=r["tflops"],
                    units=r["units"],
                ),
            )
        finally:
            sampler.wait(timeout=120)
            stop.set()
            macmon_proc.terminate()
            macmon_thread.join(timeout=10)

    analyse()


def _load(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def analyse():
    samples = _load(PERCALL)
    sysrecs = _load(SYSLOG)
    macrecs = _load(MACLOG)
    if not samples:
        emit("slow_state", dict(phase="analysis", run=TAG, error="no per-call samples"))
        return
    slow = [s for s in samples if s["ms"] >= SLOW_MS]
    runs, cur = [], 0
    for s in samples:
        if s["ms"] >= SLOW_MS:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    gaps = sorted(b["t"] - a["t"] for a, b in zip(samples, samples[1:]))
    gap = gaps[len(gaps) // 2] if gaps else 0.0
    fast_ms = sorted(s["ms"] for s in samples if s["ms"] < SLOW_MS)
    emit(
        "slow_state",
        dict(
            phase="distribution",
            run=TAG,
            n_samples=len(samples),
            n_slow=len(slow),
            slow_frac=round(len(slow) / len(samples), 3),
            median_sample_gap_ms=round(gap * 1e3, 1),
            slow_runs_s=[round(r * gap, 1) for r in sorted(runs, reverse=True)[:10]],
            slow_runs_note="durations in seconds, longest first",
            fast_median_ms=fast_ms[len(fast_ms) // 2] if fast_ms else None,
            slow_median_ms=sorted(s["ms"] for s in slow)[len(slow) // 2] if slow else None,
        ),
    )

    # --- macmon join: each macmon record covers [ts, ts + ~1 s] of wall time.
    if macrecs:
        buckets = []
        for rec in macrecs:
            t0, t1 = rec["ts"], rec["ts"] + 1.0
            win = [s for s in samples if t0 <= s["t"] < t1]
            if len(win) < 3:
                continue
            frac = sum(1 for s in win if s["ms"] >= SLOW_MS) / len(win)
            buckets.append((frac, rec, win))

        def group_stats(pred, key):
            vals = []
            for frac, rec, _ in buckets:
                if pred(frac) and rec.get(key) is not None:
                    vals.append(rec[key])
            return round(sum(vals) / len(vals), 3) if vals else None

        for name, pred in [("fast_buckets", lambda f: f < 0.2), ("slow_buckets", lambda f: f > 0.8)]:
            n = sum(1 for f, _, _ in buckets if pred(f))
            emit(
                "slow_state",
                dict(
                    phase="macmon_join",
                    run=TAG,
                    group=name,
                    n_buckets=n,
                    ane_power_w=group_stats(pred, "ane_power"),
                    gpu_freq_mhz=group_stats(pred, "gpu_freq_mhz"),
                    gpu_active_ratio=group_stats(pred, "gpu_active_ratio"),
                    cpu_active_ratio=group_stats(pred, "cpu_active_ratio"),
                    all_power_w=group_stats(pred, "all_power"),
                ),
            )

    # --- sys sampler join (same method as run 1).
    def sysrec_for(ts):
        best = None
        for r in sysrecs:
            if r["ts"] <= ts + 2.0 and (best is None or r["ts"] > best["ts"]):
                best = r
        return best

    fast_buckets, slow_buckets = {}, {}
    for s in samples:
        r = sysrec_for(s["t"])
        if r is None:
            continue
        key = int(r["ts"] // 4)
        tgt = slow_buckets if s["ms"] >= SLOW_MS else fast_buckets
        tgt.setdefault(key, r)

    def summarise(buckets):
        recs = list(buckets.values())
        if not recs:
            return None

        def mean(key):
            vals = [r[key] for r in recs if r.get(key) is not None]
            return round(sum(vals) / len(vals), 2) if vals else None

        names = {}
        for r in recs:
            for name, pct in r.get("top_cpu", []):
                names[name] = max(names.get(name, 0), pct)
        return dict(
            n_buckets=len(recs),
            cpu_busy_sum_pct=mean("cpu_busy_sum_pct"),
            cpu_speed_limit=mean("cpu_speed_limit"),
            mem_free_pct=mean("mem_free_pct"),
            swap_used_mb=mean("swap_used_mb"),
            anebench_pri=mean("anebench_pri"),
            top_cpu_peaks=sorted(names.items(), key=lambda kv: -kv[1])[:4],
        )

    emit("slow_state", dict(phase="join", run=TAG, when="fast_samples", stats=summarise(fast_buckets)))
    emit("slow_state", dict(phase="join", run=TAG, when="slow_samples", stats=summarise(slow_buckets)))


if __name__ == "__main__":
    main()
