"""3b-1 extension tests: enumerated tiers, W8A8 pre form, fail-loud errors.

Smoke + numerics + a quick timing pass, all synthetic weights. Emits to
results/extension_test.jsonl.
"""

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

HERE = Path(__file__).resolve().parent
SPIKE = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SPIKE / "ane_linear"))
import _ane_linear as ext  # noqa: E402
import ane_models as AM  # noqa: E402

CACHE = Path("/tmp/ane3a/cache_ext")
RESULTS = HERE / "results"
TIERS = [4608, 6528]
O_ANE, I = 1920, 3840
ROWS = 4128
# Production call sites: (tag, O, I, ksplit). w1w3 and QKV are the fused arms.
PROD = [("qkv", (11520, 3840, 1)), ("o", (3840, 3840, 1)), ("w13", (20480, 3840, 1)), ("w2", (3840, 10240, 4))]
PROD_ROWS = 4128  # padded to tier 4608


def emit(rec):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "extension_test.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def cosine(a, b):
    an, bn = mx.flatten(a.astype(mx.float32)), mx.flatten(b.astype(mx.float32))
    return float(mx.sum(an * bn) / (mx.sqrt(mx.sum(an * an)) * mx.sqrt(mx.sum(bn * bn))))


def main():
    rng = np.random.default_rng(7)
    w = (rng.standard_normal((O_ANE, I)) * 0.02).astype(np.float32)
    wq, sc = AM.quantize_perchannel(w)
    CACHE.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    p_enum, in_enum = AM.build_enum_w8a16(wq, sc, ksplit=1, tiers=TIERS)
    AM.convert(p_enum, in_enum, CACHE / "w8a16_enum.mlpackage")
    p_q8, in_q8 = AM.build_enum_q8pre(wq, sc, ksplit=1, tiers=TIERS)
    AM.convert(p_q8, in_q8, CACHE / "q8pre_enum.mlpackage")
    emit(dict(phase="build", seconds=round(time.perf_counter() - t0, 1)))

    x = mx.array(rng.standard_normal((ROWS, I)).astype(np.float32))
    tier = AM.pick_tier(ROWS, TIERS)
    assert tier == 4608

    # T1: W8A16 enumerated, padded rows.
    m16 = ext.AneModel(str(CACHE / "w8a16_enum.mlpackage"), TIERS[-1], I, O_ANE)
    xf = x.astype(mx.float16)
    xp = mx.pad(xf, [(0, tier - ROWS), (0, 0)])
    y16 = ext.ane_linear(xp, m16)
    mx.eval(y16)
    y16 = y16[:ROWS]
    ref = x.astype(mx.float32) @ mx.array(w).astype(mx.float32).T
    c16 = cosine(y16.astype(mx.float32), ref.astype(mx.float32))
    emit(
        dict(
            phase="w8a16_enum",
            cosine_vs_fp32=round(c16, 7),
            finite=bool(mx.isfinite(y16).all()),
            err=repr(m16.last_error()),
        )
    )

    # T2: W8A8 pre enumerated.
    mq8 = ext.AneModel(str(CACHE / "q8pre_enum.mlpackage"), TIERS[-1], I, O_ANE)
    x32 = x.astype(mx.float32)
    amax = mx.max(mx.abs(x32), axis=1, keepdims=False)
    tok = mx.maximum(amax / 127.0, mx.array(1e-6))
    q = mx.clip(mx.round(x32 / tok[:, None]), -127, 127).astype(mx.int8)
    qp = mx.pad(q, [(0, tier - ROWS), (0, 0)])
    tokp = mx.pad(tok, [(0, tier - ROWS)])
    yq8 = ext.ane_linear_q8(qp, tokp.astype(mx.float32), mq8)
    mx.eval(yq8)
    yq8 = yq8[:ROWS]
    # fp64 emulation of the same quantised math.
    qn = np.array(q.astype(mx.float32)).astype(np.float64)
    tn = np.array(tok).astype(np.float64)
    wdeq = wq.astype(np.float64) * sc.astype(np.float64)[:, None]
    emul = (qn * tn[:, None]) @ wdeq.T
    y2 = np.array(yq8.astype(mx.float32)).astype(np.float64)
    cerr = float(np.max(np.abs(y2 - emul)) / np.max(np.abs(emul)))
    emit(
        dict(
            phase="q8pre_enum",
            max_rel_vs_emulation=round(cerr, 6),
            finite=bool(mx.isfinite(yq8).all()),
            cos_vs_w8a16=round(cosine(yq8.astype(mx.float32), y16.astype(mx.float32)), 7),
            err=repr(mq8.last_error()),
        )
    )

    # T3: fail loud on a non-tier row count.
    raised = None
    try:
        bad = ext.ane_linear(mx.zeros((5000, I), mx.float16), m16)
        mx.eval(bad)
    except Exception as e:  # noqa: BLE001
        raised = type(e).__name__ + ": " + str(e)[:160]
    emit(dict(phase="fail_loud", raised=raised, recorded=repr(m16.last_error())[:160]))
    assert raised, "non-tier rows must raise"

    # T3b: the same 5000 rows padded to tier 6528 must succeed.
    tier2 = AM.pick_tier(5000, TIERS)
    assert tier2 == 6528
    x2 = mx.array(rng.standard_normal((5000, I)).astype(np.float32))
    x2p = mx.pad(x2.astype(mx.float16), [(0, tier2 - 5000), (0, 0)])
    y2p = ext.ane_linear(x2p, m16)
    mx.eval(y2p)
    ref2 = x2.astype(mx.float32) @ mx.array(w).astype(mx.float32).T
    emit(
        dict(
            phase="tier6528_pad",
            cosine_vs_fp32=round(cosine(y2p[:5000].astype(mx.float32), ref2), 7),
            finite=bool(mx.isfinite(y2p).all()),
        )
    )

    # T5: the four production call sites, W8A16 and W8A8, padded rows.
    for tag, (Op, Ip, ks) in PROD:
        wp = (rng.standard_normal((Op, Ip)) * 0.02).astype(np.float32)
        wqp, scp = AM.quantize_perchannel(wp)
        p16, i16 = AM.build_enum_w8a16(wqp, scp, ksplit=ks, tiers=AM.TIERS)
        AM.convert(p16, i16, CACHE / f"prod_{tag}_w8a16.mlpackage")
        pq8, iq8 = AM.build_enum_q8pre(wqp, scp, ksplit=ks, tiers=AM.TIERS)
        AM.convert(pq8, iq8, CACHE / f"prod_{tag}_q8pre.mlpackage")
        xpr = mx.array(rng.standard_normal((PROD_ROWS, Ip)).astype(np.float32))
        xpad = mx.pad(xpr.astype(mx.float16), [(0, 4608 - PROD_ROWS), (0, 0)])
        mm16 = ext.AneModel(str(CACHE / f"prod_{tag}_w8a16.mlpackage"), 4608, Ip, Op)
        y16p = ext.ane_linear(xpad, mm16)
        mx.eval(y16p)
        refp = xpr.astype(mx.float32) @ mx.array(wp).astype(mx.float32).T
        cos16 = cosine(y16p[:PROD_ROWS].astype(mx.float32), refp)
        x32p = xpr.astype(mx.float32)
        am = mx.max(mx.abs(x32p), axis=1, keepdims=False)
        tk5 = mx.maximum(am / 127.0, mx.array(1e-6))
        q5 = mx.clip(mx.round(x32p / tk5[:, None]), -127, 127).astype(mx.int8)
        mmq8 = ext.AneModel(str(CACHE / f"prod_{tag}_q8pre.mlpackage"), 4608, Ip, Op)
        yq8p = ext.ane_linear_q8(
            mx.pad(q5, [(0, 4608 - PROD_ROWS), (0, 0)]), mx.pad(tk5, [(0, 4608 - PROD_ROWS)]), mmq8
        )
        mx.eval(yq8p)
        cosq8 = cosine(yq8p[:PROD_ROWS].astype(mx.float32), refp)
        emit(
            dict(
                phase=f"prod_{tag}",
                O=Op,
                I=Ip,
                ksplit=ks,
                cos_w8a16=round(cos16, 7),
                cos_q8_vs_fp32=round(float(cosq8), 7),
                finite=bool(mx.isfinite(y16p).all() and mx.isfinite(yq8p).all()),
            )
        )
        del mm16, mmq8

    # T4: timing (quick, best of 3): enum tier 4608 vs earlier fixed 4224 numbers.
    med = []
    for _ in range(3):
        ts = []
        for _ in range(30):
            b = ext.ane_linear(xp, m16)
            t0 = time.perf_counter()
            mx.eval(b)
            ts.append(time.perf_counter() - t0)
        med.append(sorted(ts)[15])
    emit(dict(phase="timing_w8a16_enum4608", median_ms=round(min(med) * 1e3, 2)))
    med = []
    for _ in range(3):
        ts = []
        for _ in range(30):
            b = ext.ane_linear_q8(qp, tokp.astype(mx.float32), mq8)
            t0 = time.perf_counter()
            mx.eval(b)
            ts.append(time.perf_counter() - t0)
        med.append(sorted(ts)[15])
    emit(dict(phase="timing_q8pre_enum4608", median_ms=round(min(med) * 1e3, 2)))


if __name__ == "__main__":
    main()
