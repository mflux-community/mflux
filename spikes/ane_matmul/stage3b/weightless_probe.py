"""3b-2 dynamic-weights probe, weightless form (per DRAWTHINGS-ANE-FINDINGS.md, ccv BSD-3).

Program per (M, N, K): int8 inputs w[1,1,N,K] and x[1,1,K,M], mb.dequantize
with the CONSTANT scale 1/sqrt(K) on both, fp16 mb.matmul, K statically
sliced above the Draw Things cliff (limit 2688 for M>1024). Real row and
token scales live in the epilogue, outside the program - not modelled here.
M = 4352 (256-aligned from the real 4128). Feeding and timing run through
bench/dynrun (owned-copy mode for now; IOSurface handoff is the next step).

Gate input: per-call ms and cosine vs int64 emulation at the four call sites,
plus the first-predict (cold) time per shape.
"""

import json
import shutil
import sys
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import wired_probe_models as WPM
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dyn_ops_probe2 as D  # TARGET, CACHE-style helpers

HERE = Path(__file__).resolve().parent
CACHE = Path("/tmp/ane3a/cache_wl")
RESULTS = HERE / "results"
M = 4352
KLIMIT = 2688
SITES = [("qkv", 5760, 3840), ("o", 1920, 3840), ("w13", 10240, 3840), ("w2", 1920, 10240)]


def _convert(prog, ci, path):
    if path.exists():
        shutil.rmtree(path)
    m = ct.convert(
        prog,
        convert_to="mlprogram",
        inputs=ci,
        minimum_deployment_target=D.TARGET,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    m.save(str(path))


def cos_of(y, ref):
    yy = y.astype(np.float64).reshape(ref.shape)
    return float(yy.ravel() @ ref.ravel() / (np.linalg.norm(yy.ravel()) * np.linalg.norm(ref.ravel()) + 1e-30))


def emit(rec):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "weightless_probe.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def k_slices(K):
    if K <= KLIMIT:
        return 1
    for n in (2, 4):
        if K / n <= KLIMIT:
            return n
    return 4


def build_weightless(N, K):
    ns = k_slices(K)
    ks = K // ns
    c = np.float16(1.0 / np.sqrt(K))

    @mb.program(
        input_specs=[mb.TensorSpec((1, 1, N, K), dtype=types.int8), mb.TensorSpec((1, 1, K, M), dtype=types.int8)],
        opset_version=D.TARGET,
    )
    def prog(w, x):
        wd = mb.dequantize(input=w, scale=c)
        xd = mb.dequantize(input=x, scale=c)
        y = None
        for i in range(ns):
            if ns == 1:
                wm, xm = wd, xd
            else:
                wm = mb.slice_by_size(x=wd, begin=[0, 0, 0, i * ks], size=[1, 1, N, ks])
                xm = mb.slice_by_size(x=xd, begin=[0, 0, i * ks, 0], size=[1, 1, ks, M])
            part = mb.matmul(x=wm, y=xm)
            y = part if y is None else mb.add(x=y, y=part)
        return mb.identity(x=y, name="y")

    return prog, [ct.TensorType(name=n) for n in ("w", "x")]


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    for site, N, K in SITES:
        rec = {"site": site, "M": M, "N": N, "K": K, "kslices": k_slices(K)}
        rng = np.random.default_rng(hash(site) % 2**31)
        wq = rng.integers(-127, 128, (N, K)).astype(np.int8)
        sw = np.maximum(np.abs(wq.astype(np.float32)).max(axis=1) / 127.0, np.float32(1e-8))
        xq = rng.integers(-127, 128, (K, M)).astype(np.int8)
        # tokens are the columns of xq; per-token scale over axis 0
        tx2 = np.maximum(np.abs(xq.astype(np.float32)).max(axis=0) / 127.0, np.float32(1e-8))
        path = CACHE / f"wl_{site}.mlpackage"
        if path.exists():
            shutil.rmtree(path)
        t0 = time.perf_counter()
        try:
            prog, ci = build_weightless(N, K)
            m = ct.convert(
                prog,
                convert_to="mlprogram",
                inputs=ci,
                minimum_deployment_target=D.TARGET,
                compute_precision=ct.precision.FLOAT16,
                compute_units=ct.ComputeUnit.CPU_AND_NE,
            )
            m.save(str(path))
        except Exception as e:  # noqa: BLE001
            rec.update(convert="fail", err=str(e)[:220])
            emit(rec)
            continue
        rec["convert_s"] = round(time.perf_counter() - t0, 1)
        arrs = {"w": np.ascontiguousarray(wq), "x": np.ascontiguousarray(xq)}
        res, err, y, peak = D.run_mode(path, arrs, [])
        if res is None:
            rec.update(predict="fail", err=(err or "")[:220])
            emit(rec)
            continue
        c2 = 1.0 / K
        ref = (wq.astype(np.float64) @ xq.astype(np.float64)) * c2
        yy = y.astype(np.float64).reshape(N, M)
        if yy.shape != ref.shape:
            yy = yy.reshape(ref.shape)
        cos = float(yy.ravel() @ ref.ravel() / (np.linalg.norm(yy.ravel()) * np.linalg.norm(ref.ravel()) + 1e-30))
        rec.update(
            ms=round(res["med_ms"], 2),
            cold_ms=round(res.get("cold_ms", 0), 1),
            cos_vs_int64=round(cos, 7),
            tflops=round(2 * M * N * K / 1e12 / (res["med_ms"] / 1e3), 1),
            ane_peak_w=round(peak, 1),
        )
        # baked controls: same int weights, token-major boundary (M, K) / cf (1,K,1,M)
        pb = CACHE / f"baked16_{site}.mlpackage"
        prog_b, ci_b = WPM.build_fixed(wq, sw.astype(np.float16), "w8a16", ksplit=1, rows=M)
        _convert(prog_b, ci_b, pb)
        a_codes = xq.T.astype(np.float32)  # (M, K) token-major
        xdq = (a_codes * tx2[:, None]).astype(np.float16)
        res, err, yb, pk = D.run_mode(pb, {"x": np.ascontiguousarray(xdq)}, [])
        if res:
            rec["baked16_ms"] = round(res["med_ms"], 2)
            rec["baked16_cos"] = round(cos_of(yb, ref), 7)
        pb8 = CACHE / f"baked8_{site}.mlpackage"
        prog8, ci8 = WPM.build_fixed(wq, sw.astype(np.float16), "w8a8", ksplit=1, rows=M)
        _convert(prog8, ci8, pb8)
        res8, _, y8, _ = D.run_mode(
            pb8, {"x": np.ascontiguousarray(xq), "tok": np.ascontiguousarray(tx2.astype(np.float16))}, []
        )
        if res8:
            rec["baked8_ms"] = round(res8["med_ms"], 2)
            rec["baked8_cos"] = round(cos_of(y8, ref), 7)
        emit(rec)


if __name__ == "__main__":
    main()
