"""Slow-state probe: loop the w1w3 ANE call for ~10 minutes and correlate with system signals.

Under BENCH.lock + watchdog. Starts the system sampler, runs anebench in window
mode with per-sample logging, then analyses: slow-state frequency, run lengths,
and which sampled signal (thermal, memory, swap, CPU load) predicts slow samples.
Emits to results/slow_state.jsonl. Raw logs: results/slow_state_percall.jsonl,
results/slow_state_sys.jsonl.
"""

import json
import os
import subprocess
import time

import s3a
from s3a import HERE, MODELS, BenchLock, anebench, emit

DURATION_S = 600
MODEL = MODELS / "compiled" / "sa_w1w3.mlmodelc"  # hash recorded in recompile_w1w3.jsonl
SHAPE = (4224, 20480, 3840)
PERCALL = HERE / "results" / "slow_state_percall.jsonl"
SYSLOG = HERE / "results" / "slow_state_sys.jsonl"
SLOW_MS = 70.0  # fast state ~53 ms, slow state ~90 ms


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


def main():
    PERCALL.unlink(missing_ok=True)
    SYSLOG.unlink(missing_ok=True)
    pm = check_powermetrics()
    emit("slow_state", dict(phase="setup", powermetrics=pm, model=str(MODEL), duration_s=DURATION_S))

    with BenchLock(owner=f"stage3a-slowstate-{os.getpid()}"):
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
                    median_ms=r["median_ms"],
                    p90_ms=r["p90_ms"],
                    tflops=r["tflops"],
                    units=r["units"],
                ),
            )
        finally:
            sampler.wait(timeout=120)

    analyse()


def analyse():
    samples = [json.loads(l) for l in PERCALL.read_text().splitlines() if l.strip()]
    sysrecs = [json.loads(l) for l in SYSLOG.read_text().splitlines() if l.strip()]
    if not samples:
        emit("slow_state", dict(phase="analysis", error="no per-call samples"))
        return
    slow = [s for s in samples if s["ms"] >= SLOW_MS]
    # Runs of consecutive slow samples.
    runs, cur = [], 0
    for s in samples:
        if s["ms"] >= SLOW_MS:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    # Duration estimate: median sample interval x run length.
    gaps = [b["t"] - a["t"] for a, b in zip(samples, samples[1:])]
    gaps.sort()
    gap = gaps[len(gaps) // 2] if gaps else 0.0
    emit(
        "slow_state",
        dict(
            phase="distribution",
            n_samples=len(samples),
            n_slow=len(slow),
            slow_frac=round(len(slow) / len(samples), 3),
            median_sample_gap_ms=round(gap * 1e3, 1),
            slow_runs=[round(r * gap, 1) for r in sorted(runs, reverse=True)[:10]],
            slow_runs_note="durations in seconds, longest first",
            fast_median_ms=sorted(s["ms"] for s in samples if s["ms"] < SLOW_MS)[
                max(0, (len(samples) - len(slow)) // 2)
            ]
            if len(slow) < len(samples)
            else None,
            slow_median_ms=sorted(s["ms"] for s in slow)[len(slow) // 2] if slow else None,
        ),
    )

    # Join: bucket per-call samples by the sys sample that covers their timestamp.
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
        key = int(r["ts"] // 4)  # 4-second buckets
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
        top = sorted(names.items(), key=lambda kv: -kv[1])[:4]
        return dict(
            n_buckets=len(recs),
            cpu_busy_sum_pct=mean("cpu_busy_sum_pct"),
            cpu_speed_limit=mean("cpu_speed_limit"),
            mem_free_pct=mean("mem_free_pct"),
            swap_used_mb=mean("swap_used_mb"),
            top_cpu_peaks=top,
        )

    emit("slow_state", dict(phase="join", when="fast_samples", stats=summarise(fast_buckets)))
    emit("slow_state", dict(phase="join", when="slow_samples", stats=summarise(slow_buckets)))


if __name__ == "__main__":
    main()
