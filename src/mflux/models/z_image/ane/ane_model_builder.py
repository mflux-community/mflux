"""Core ML builders for the enumerated ANE linear packages (W8A16 and W8A8 pre).

coremltools is needed only when a package is BUILT; loading a cached package
needs just the native extension. 1x1 Conv2d channels-first (1, C, 1, T) with
baked int8 weights, the Stage 1-3a winning form. Enumerated int8 inputs must
sit at a channels-first boundary: a token-major enumerated int8 boundary fails
the Core ML plan builder (see spikes/ane_matmul/stage3b/diag_q8plan.py).
"""

import hashlib
import itertools
from pathlib import Path

import numpy as np

_SYM = itertools.count()


class AneModelBuilder:
    TIERS = [4608, 6528, 8320, 16128]

    @staticmethod
    def pick_tier(rows: int, tiers: list[int] | None = None) -> int:
        tiers = tiers or AneModelBuilder.TIERS
        for t in tiers:
            if rows <= t:
                return t
        raise ValueError(f"rows {rows} exceeds the largest ANE tier {tiers[-1]}")

    @staticmethod
    def quantize_perchannel(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sc = (np.abs(w).max(axis=1) / 127.0).astype(np.float32)
        sc = np.maximum(sc, 1e-12)
        wq = np.clip(np.round(w / sc[:, None]), -127, 127).astype(np.int8)
        return wq, sc.astype(np.float16)

    @staticmethod
    def cache_path(w: np.ndarray, prec: str, ksplit: int, tag: str, cache_dir: Path, tiers: list[int]) -> Path:
        digest = hashlib.sha1(np.ascontiguousarray(w, dtype=np.float32).tobytes()).hexdigest()[:12]
        od, idd = w.shape
        return cache_dir / f"{tag}_{prec}_{od}x{idd}_k{ksplit}_{digest}.mlpackage"

    @staticmethod
    def build(w: np.ndarray, prec: str, ksplit: int, path: Path, tiers: list[int] | None = None) -> Path:
        tiers = tiers or AneModelBuilder.TIERS
        if path.exists():
            return path
        import coremltools as ct
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import get_new_symbol, types

        target = ct.target.iOS26 if hasattr(ct.target, "iOS26") else ct.target.iOS18
        od, idd = w.shape
        wq, sc = AneModelBuilder.quantize_perchannel(w)
        w4 = np.ascontiguousarray(wq.reshape(od, idd, 1, 1))
        s4 = sc.reshape(od, 1, 1, 1)
        if idd % ksplit:
            raise ValueError(f"inner dim {idd} not divisible by ksplit {ksplit}")
        kc = idd // ksplit
        ws = [np.ascontiguousarray(w4[:, k * kc : (k + 1) * kc]) for k in range(ksplit)]
        tsym = get_new_symbol(f"ane_t{next(_SYM)}")

        def conv_program(body):
            return mb.program(input_specs=body["specs"], opset_version=target)(body["fn"])

        def chunks(xcf):
            acc = None
            for k in range(ksplit):
                if ksplit == 1:
                    xs = xcf
                else:
                    xs = mb.slice_by_index(
                        x=xcf,
                        begin=[0, k * kc, 0, 0],
                        end=[1, (k + 1) * kc, 1, 0],
                        end_mask=[False, False, False, True],
                    )
                xs = mb.dequantize(input=xs, scale=np.float16(1.0)) if prec == "w8a8" else xs
                wf = mb.constexpr_blockwise_shift_scale(data=ws[k], scale=s4, offset=None)
                y = mb.conv(x=xs, weight=wf, pad_type="valid")
                acc = y if acc is None else mb.add(x=acc, y=y)
            return acc

        def to_token_major(acc):
            return mb.transpose(x=mb.reshape(x=acc, shape=[od, -1]), perm=[1, 0])

        if prec == "w8a16":

            def w8a16_fn(x):
                xcf = mb.reshape(x=mb.transpose(x=x, perm=[1, 0]), shape=[1, idd, 1, -1])
                return mb.identity(x=to_token_major(chunks(xcf)), name="y")

            body = {"specs": [mb.TensorSpec((tsym, idd), dtype=types.fp16)], "fn": w8a16_fn}
            prog = conv_program(body)
            ct_inputs = [
                ct.TensorType(
                    name="x", shape=ct.EnumeratedShapes(shapes=[[t, idd] for t in tiers], default=[tiers[0], idd])
                )
            ]
        elif prec == "w8a8":

            def w8a8_fn(x, tok):
                y = mb.mul(x=chunks(x), y=tok)
                return mb.identity(x=to_token_major(y), name="y")

            body = {
                "specs": [
                    mb.TensorSpec((1, idd, 1, tsym), dtype=types.int8),
                    mb.TensorSpec((1, 1, 1, tsym), dtype=types.fp16),
                ],
                "fn": w8a8_fn,
            }
            prog = conv_program(body)
            ct_inputs = [
                ct.TensorType(
                    name="x",
                    shape=ct.EnumeratedShapes(shapes=[[1, idd, 1, t] for t in tiers], default=[1, idd, 1, tiers[0]]),
                ),
                ct.TensorType(
                    name="tok",
                    shape=ct.EnumeratedShapes(shapes=[[1, 1, 1, t] for t in tiers], default=[1, 1, 1, tiers[0]]),
                ),
            ]
        else:
            raise ValueError(f"unknown ANE precision {prec}")

        m = ct.convert(
            prog,
            convert_to="mlprogram",
            inputs=ct_inputs,
            minimum_deployment_target=target,
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(path))
        return path
